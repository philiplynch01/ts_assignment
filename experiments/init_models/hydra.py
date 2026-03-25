import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV

from experiments.hydra import Hydra, SparseScaler
from experiments.hydra.code.optimised_hydra import UpdatedHydra

from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile


def _to_numpy(t) -> np.ndarray:
    """Safely convert a torch Tensor or any array-like to a numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get_safe_device() -> torch.device:
    """
    Falls back to CPU if CUDA or MPS is unavailable.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HydraModel:
    def __init__(self,
                 input_dim: int,
                 device: torch.device | None = None,
                 k: int = 8,
                 g: int = 64,
                 seed: int = 42,
                 use_latency_optimisation: bool = True,
                 print_profile: bool = False):
        self.device = device or _get_safe_device()
        self.print_profile = print_profile
        self.use_latency_optimisation = use_latency_optimisation
        self.classifier = None
        print(f"[HydraModel] Using device: {self.device}")
        if use_latency_optimisation or self.device.type == "mps":
            print(f"[HydraModel] Using latency optimisation")
            self.transform = UpdatedHydra(input_dim, k=k, g=g, seed=seed).to(self.device)
        else:
            self.transform = Hydra(input_dim, k=k, g=g, seed=seed).to(self.device)
        self.scaler = SparseScaler()
        torch.manual_seed(seed)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert (N, T) numpy array to (N, 1, T) float tensor on the correct device."""
        return torch.from_numpy(x).float().unsqueeze(-2).to(self.device)


    # def __call__(self,
    #              x_train: np.ndarray,
    #              y_train: np.ndarray,
    #              x_test: np.ndarray,
    #              y_test: np.ndarray) -> np.ndarray:

    #     x_train_t = self._to_tensor(x_train)
    #     x_test_t = self._to_tensor(x_test)

    #     try:
    #         if self.print_profile:
    #             x_train_transformed, x_test_transformed = self._print_profile(x_train_t, x_test_t)
    #         else:
    #             if self.use_latency_optimisation or self.device.type == "mps":
    #                 x_train_transformed = self.transform.batch(x_train_t)
    #                 x_test_transformed = self.transform.batch(x_test_t)
    #             else:
    #                 x_train_transformed = self.transform(x_train_t)
    #                 x_test_transformed = self.transform(x_test_t)

    #     except Exception as e:
    #         print(f"[HydraModel] Transformation error: {e}")
    #         raise
    #     try:
    #         print("[HydraModel] Fitting model")
    #         x_train_transformed = self.scaler.fit_transform(x_train_transformed)
    #         x_test_transformed = self.scaler.transform(x_test_transformed)
    #     except Exception as e:
    #         print(f"[HydraModel] Scaling error: {e}")
    #         raise

    #     # Detach from torch graph --> numpy before handing to sklearn
    #     x_train_np = _to_numpy(x_train_transformed)
    #     x_test_np = _to_numpy(x_test_transformed)
    #     y_train_np = y_train.astype(np.int32)

    #     try:
    #         self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    #         self.classifier.fit(x_train_np, y_train_np)
    #     except Exception as e:
    #         print(f"[HydraModel] Classification error: {e}")
    #         raise

    #     return self.classifier.predict(x_test_np)
    

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Fit scaler + ridge classifier on Hydra features."""
        x_train_tensor = self._to_tensor(x_train)
        try:
            if self.print_profile:
                x_train_transformed = self._print_profile(x_train_tensor)
            else:
                x_train_transformed = self._transform_features(x_train_tensor)
        except Exception as e:
            print(f"[HydraModel] Transformation error during fit: {e}")
            raise

        try:
            x_train_transformed = self.scaler.fit_transform(x_train_transformed)
        except Exception as e:
            print(f"[HydraModel] Scaling error during fit: {e}")
            raise

        x_train_np = _to_numpy(x_train_transformed)
        y_train_np = np.asarray(y_train).astype(np.int32)

        try:
            self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            self.classifier.fit(x_train_np, y_train_np)
        except Exception as e:
            print(f"[HydraModel] Classification error during fit: {e}")
            raise

        self.is_fitted = True
        return self
    

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict labels for raw time series."""
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("HydraModel must be fitted before calling predict().")

        x_test_t = self._to_tensor(x_test)

        try:
            x_test_transformed = self._transform_features(x_test_t)
            x_test_transformed = self.scaler.transform(x_test_transformed)
        except Exception as e:
            print(f"[HydraModel] Prediction pipeline error: {e}")
            raise

        x_test_np = _to_numpy(x_test_transformed)
        return self.classifier.predict(x_test_np)
    


    def __call__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray | None = None,) -> np.ndarray:
        """ Compatibility wrapper for pipeline.py."""
        self.fit(x_train, y_train)
        return self.predict(x_test)
    


    def explain(self, x_single: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """
        Compute a saliency map for a single time series.

        Steps:
        1. Ensure model is fitted
        2. Convert input to correct shape (1, 1, T)
        3. Determine which class to explain
        4. Call Hydra's saliency function
        """
        print("\n[EXPLAIN] Starting explanation...")

        # 1. Check model is fitted
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Model must be fitted before calling explain().")

        # 2. Ensure correct input shape
        x_single = np.asarray(x_single)
        print(f"[EXPLAIN] Original input shape: {x_single.shape}")

        if x_single.ndim != 1:
            raise ValueError(f"Expected shape (T,), got {x_single.shape}")

        # Add batch dimension --> (1, T)
        x_batched = x_single[None, :]
        print(f"[EXPLAIN] After adding batch dim: {x_batched.shape}")

        # Convert to tensor + add channel --> (1, 1, T)
        x_single_t = self._to_tensor(x_batched)
        print(f"[EXPLAIN] Tensor shape (for Hydra): {x_single_t.shape}")

        # 3. Determine class index
        if class_index is None:
            print("[EXPLAIN] No class_index provided --> using model prediction")

            pred = self.predict(x_batched)[0]
            print(f"[EXPLAIN] Predicted class label: {pred}")

            if hasattr(self.classifier, "classes_"):
                print(f"[EXPLAIN] Available classes: {self.classifier.classes_}")

            # Multiclass case
            if hasattr(self.classifier, "classes_") and len(self.classifier.classes_) > 2:
                class_matches = np.where(self.classifier.classes_ == pred)[0]
                if len(class_matches) == 0:
                    raise ValueError(f"Predicted class {pred} not found in classes_")
                class_index = int(class_matches[0])
                print(f"[EXPLAIN] Mapped class_index: {class_index}")

            # Binary case
            else:
                class_index = 0
                print(f"[EXPLAIN] Binary classification --> using class_index = 0")

        else:
            print(f"[EXPLAIN] Using provided class_index: {class_index}")

        # --- 4. Check saliency function exists ---
        if not hasattr(self.transform, "get_saliency_map"):
            raise AttributeError("Hydra transform does not implement get_saliency_map()")

        print("[EXPLAIN] Calling Hydra saliency function...")

        # --- 5. Compute saliency ---
        saliency = self.transform.get_saliency_map(
            x_single_t,
            self.classifier,
            self.scaler,
            class_index=class_index,
        )

        print(f"[EXPLAIN] Saliency shape: {saliency.shape}")
        print(f"[EXPLAIN] Saliency stats --> min: {saliency.min()}, max: {saliency.max()}")

        print("[EXPLAIN] Done.\n")

        return saliency






    def _print_profile(self, x_train_t, x_test_t):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                if self.use_latency_optimisation or self.device.type == "mps":
                    x_train_transformed = self.transform.batch(x_train_t)
                    x_test_transformed = self.transform.batch(x_test_t)
                else:
                    x_train_transformed = self.transform(x_train_t)
        print(prof.key_averages())
        return x_train_transformed, x_test_transformed
    
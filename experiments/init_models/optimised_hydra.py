import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

from experiments.hydra import SparseScaler
from experiments.hydra.code.optimised_hydra import UpdatedHydra
from experiments.utils.model_utils import get_torch_device


def _to_numpy(t) -> np.ndarray:
    """Safely convert a torch Tensor or any array-like to a numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get_safe_device() -> torch.device:
    """
    Falls back to CPU if CUDA is unavailable.
    """
    return torch.device(get_torch_device())


class LOHydraModel:
    def __init__(self,
                 input_dim: int,
                 device: torch.device | None = None,
                 k: int = 8,
                 g: int = 64,
                 seed: int = 42):
        self.device = device or _get_safe_device()
        print(f"[HydraModel] Using device: {self.device}")
        self.transform = UpdatedHydra(input_dim, k=k, g=g, seed=seed).to(self.device)
        self.scaler = SparseScaler()
        torch.manual_seed(seed)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert (N, T) numpy array to (N, 1, T) float tensor on the correct device."""
        return torch.from_numpy(x).float().unsqueeze(-2).to(self.device)

    def __call__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray) -> np.ndarray:

        x_train_t = self._to_tensor(x_train)
        x_test_t = self._to_tensor(x_test)

        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    x_train_transformed = self.transform.batch(x_train_t)
                    x_test_transformed = self.transform.batch(x_test_t)

            print(prof.key_averages())
        except Exception as e:
            print(f"[HydraModel] Transformation error: {e}")
            raise

        try:
            print("[HydraModel] Fitting model")
            x_train_transformed = self.scaler.fit_transform(x_train_transformed)
            x_test_transformed = self.scaler.transform(x_test_transformed)
        except Exception as e:
            print(f"[HydraModel] Scaling error: {e}")
            raise

        # Detach from torch graph → numpy before handing to sklearn
        x_train_np = _to_numpy(x_train_transformed)
        x_test_np = _to_numpy(x_test_transformed)
        y_train_np = y_train.astype(np.int32)

        try:
            print("[HydraModel] Predicting model")
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(x_train_np, y_train_np)
        except Exception as e:
            print(f"[HydraModel] Classification error: {e}")
            raise

        return classifier.predict(x_test_np)

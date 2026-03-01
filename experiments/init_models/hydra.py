from functorch.dim import Tensor

from experiments.hydra import Hydra, SparseScaler
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import torch
from experiments.utils.model_utils import get_torch_device

def _to_numpy(t) -> np.ndarray:
    """Safely convert a torch Tensor or any array-like to a numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get_safe_device() -> torch.device:
    """
    Return the best available device. MPS is excluded because Hydra's
    convolution ops are not fully supported on MPS (weight/input type mismatch).
    Falls back to CPU if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HydraModel:
    def __init__(self,
                 input_dim: int,
                 device: torch.device | None = None,
                 k: int = 8,
                 g: int = 64,
                 seed: int = 42):
        self.device = device or _get_safe_device()
        self.transform = Hydra(input_dim, k=k, g=g, seed=seed).to(self.device)
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
            x_train_transformed = self.transform(x_train_t)
            x_test_transformed = self.transform(x_test_t)
        except Exception as e:
            print(f"[HydraModel] Transformation error: {e}")
            raise

        try:
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
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(x_train_np, y_train_np)
        except Exception as e:
            print(f"[HydraModel] Classification error: {e}")
            raise

        return classifier.predict(x_test_np)


class MrSQMModel:
    def __init__(self,
                 strat: str = 'RS',
                 features_per_rep: int = 500,
                 selection_per_rep: int = 2000,
                 nsax: int = 0,
                 nsfa: int = 5,
                 random_state: int = 42,
                 sfa_norm: bool = True,
                 first_diff: bool = True):

        self.random_state = random_state
        self.transformer_kwargs = dict(
            strat=strat,
            features_per_rep=features_per_rep,
            selection_per_rep=selection_per_rep,
            nsax=nsax,
            nsfa=nsfa,
            random_state=random_state,
            sfa_norm=sfa_norm,
            first_diff=first_diff
        )
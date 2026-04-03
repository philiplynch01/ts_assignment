import numpy as np
import os
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass


def get_data_dir() -> str:
    current_dir: str = os.path.dirname(__file__)
    data_dir: str = os.path.join(current_dir, "..", "..", "data")
    return data_dir

def get_cmj_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir: str = get_data_dir()
    cmj_dir: str = os.path.join(data_dir, "cmj")
    x_train: np.ndarray = np.load(os.path.join(cmj_dir, "X_train_magnitude.npy"))
    y_train: np.ndarray = np.load(os.path.join(cmj_dir, "CMJ_y_train.npy"))
    x_test: np.ndarray = np.load(os.path.join(cmj_dir, "X_test_magnitude.npy"))
    y_test: np.ndarray = np.load(os.path.join(cmj_dir, "CMJ_y_test.npy"))
    return x_train, y_train, x_test, y_test

@dataclass
class RunResult:
    dataset: str
    model: str
    seed: int
    # Dataset metadata
    n_train: int = 0
    n_test: int = 0
    series_length: int = 0
    n_classes: int = 0
    # Timing (seconds)
    # fit_time: float = 0.0
    # predict_time: float = 0.0
    total_time: float = 0.0
    # Memory (MB)
    peak_memory_mb: float = 0.0
    # Metrics
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    # Status
    status: str = "ok"          # ok | error
    error_msg: str = ""
    sequence_length: int = None


def load_dataset(name: str):
    """Load UCR dataset using canonical train/test splits."""
    x_train, y_train = load_classification(name, split="train")
    x_test, y_test = load_classification(name, split="test")

    # aeon returns shape (N, n_channels, T) — squeeze to (N, T) for univariate
    if x_train.ndim == 3 and x_train.shape[1] == 1:
        x_train = x_train[:, 0, :]
        x_test = x_test[:, 0, :]

    # Encode string labels to integers
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return x_train, y_train, x_test, y_test, le


def dataset_meta(x_train, x_test, y_train) -> dict:
    return {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "series_length": x_train.shape[-1],
        "n_classes": len(np.unique(y_train)),
    }
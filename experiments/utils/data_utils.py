import numpy as np
import os

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
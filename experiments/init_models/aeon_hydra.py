import numpy as np
from aeon.classification.convolution_based import HydraClassifier

class AeonHydraModel:
    def __init__(self, n_jobs = -1, random_state=42) -> None:
        self.classifier = HydraClassifier(n_jobs=n_jobs, random_state=random_state)

    def __call__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray) -> np.ndarray:
        try:
            self.classifier.fit(x_train, y_train)
            predictions = self.classifier.predict(x_test)
            return predictions
        except Exception as e:
            print(f"Unexpected error during fitting/prediction: {e}")
            raise e
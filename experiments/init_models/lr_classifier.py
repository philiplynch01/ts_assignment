import numpy as np
from dtaidistance import dtw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


class LRRawModel:
    """Logistic Regression on raw time series features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            multi_class='auto',
            C=1.0
        )

    def __call__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray):

        # Flatten to 2D if needed: (n_samples, n_timesteps)
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        x_test_2d = x_test.reshape(x_test.shape[0], -1)

        try:
            x_train_scaled = self.scaler.fit_transform(x_train_2d)
        except Exception as e:
            print(f"Unexpected error during scaling: {e}")
            raise e

        try:
            self.classifier.fit(x_train_scaled, y_train)
        except Exception as e:
            print(f"Unexpected error during classification: {e}")
            raise e

        try:
            x_test_scaled = self.scaler.transform(x_test_2d)
            predictions = self.classifier.predict(x_test_scaled)
        except Exception as e:
            print(f"Unexpected error during prediction: {e}")
            raise e

        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        try:
            accuracy = accuracy_score(y_test, predictions)
            print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Unexpected error during accuracy calculation: {e}, {y_test.shape}, {predictions.shape}")
            raise e

        return predictions

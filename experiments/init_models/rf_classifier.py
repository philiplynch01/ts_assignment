import numpy as np
from dtaidistance import dtw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class RandomForestModel:
    """Random Forest classifier on raw time series features."""

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1
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
            self.classifier.fit(x_train_2d, y_train)
        except Exception as e:
            print(f"Unexpected error during classification: {e}")
            raise e

        try:
            predictions = self.classifier.predict(x_test_2d)
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


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mrsqm.mrsqm_wrapper import MrSQMTransformer


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

        self.transformer = MrSQMTransformer(
            strat=strat,
            features_per_rep=features_per_rep,
            selection_per_rep=selection_per_rep,
            nsax=nsax,
            nsfa=nsfa,
            random_state=random_state,
            sfa_norm=sfa_norm,
            first_diff=first_diff
        )

    def __call__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray):

        try:
            train_x = self.transformer.fit_transform(x_train, y_train)
        except Exception as e:
            print(f"Unexpected error during transformation: {e}")
            raise e

        try:
            classifier = LogisticRegression(
                solver='newton-cg',
                class_weight='balanced',
                random_state=0,
                max_iter=1000
            )
            classifier.fit(train_x, y_train)
        except Exception as e:
            print(f"Unexpected error during classification: {e}")
            raise e

        try:
            test_x = self.transformer.transform(x_test)
            predictions = classifier.predict(test_x)
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
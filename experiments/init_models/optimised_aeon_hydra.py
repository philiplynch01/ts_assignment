import numpy as np
import torch
import torch.nn.functional as F
from aeon.classification.convolution_based import HydraClassifier
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import HydraTransformer
from aeon.transformations.collection.convolution_based._hydra import _HydraInternal
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline


class BatchOptimisedHydraInternal(_HydraInternal):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X, batch_size=256):
        seq_length = X.shape[-1]
        target_elements = 5_000
        batch_size = max(1, min(256, target_elements // seq_length))

        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self._forward(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self._forward(X[batch]))
            return torch.cat(Z)

    def _forward(self, X):
        n_examples, n_channels, _ = X.shape

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                if n_channels > 1:  # Multivariate
                    _Z = F.conv1d(
                        (
                            X[:, self.idx[dilation_index][diff_index]].sum(2)
                            if diff_index == 0
                            else diff_X[
                                :, self.idx[dilation_index][diff_index]
                            ].sum(2)
                        ),
                        self.W[dilation_index][diff_index],
                        dilation=d,
                        padding=p,
                        groups=self.h,
                    ).view(n_examples, self.h, self.k, -1)
                else:  # Univariate
                    _Z = F.conv1d(
                        X if diff_index == 0 else diff_X,
                        self.W[dilation_index, diff_index],
                        dilation=d,
                        padding=p,
                    ).view(n_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(n_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(n_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(n_examples, -1)

        return Z

class BatchOptimisedHydraTransformer(HydraTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def _fit(self, X, y=None):
        self._n_jobs = check_n_jobs(self.n_jobs)
        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        torch.set_num_threads(self._n_jobs)
        try:
            self._hydra = BatchOptimisedHydraInternal(
                X.shape[2],
                X.shape[1],
                k=self.n_kernels,
                g=self.n_groups,
                max_num_channels=self.max_num_channels,
            )
        except Exception as e:
            print(f"Unexpected error during fitting/prediction: {e}")

class BatchOptimisedHydraClassifier(HydraClassifier):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        transform = BatchOptimisedHydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )

        self._clf = make_pipeline(
            transform,
            _SparseScaler(),
            RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
            ),
        )
        self._clf.fit(X, y)

        return self


class BatchedOptimisedAeonHydraModel:
    def __init__(self, n_jobs = -1, random_state=42) -> None:
        self.classifier = BatchOptimisedHydraClassifier(n_jobs=n_jobs, random_state=random_state)

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

if __name__ == "__main__":
    print("Running Aeon Hydra Model")
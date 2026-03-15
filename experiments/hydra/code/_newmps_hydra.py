# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
import torch

from experiments.hydra import Hydra
import torch.nn.functional as F

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

# Implementation of Hydra which utilises Macs MPS GPU
# TODO: Should we try test it on CUDA?
class MPSHydra(Hydra):
    def __init__(self, input_length, k=8, g=64, seed=42):
        # 1. Run the parent initialization (this sets up self.W automatically!)
        super().__init__(input_length=input_length, k=k, g=g, seed=seed)


    def forward(self, X):
        # Overridden forward pass using our python lists to prevent .item() stalls
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            # Using our lists instead of tensor.item()!
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                _Z = F.conv1d(X if diff_index == 0 else diff_X,
                              self.W[dilation_index, diff_index],
                              dilation=d, padding=p) \
                    .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device=X.device)
                count_max.scatter_add_(-1, max_indices, max_values)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device=X.device)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)
        return Z
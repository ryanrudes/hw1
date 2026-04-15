from torch import nn, Tensor

import torch
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(W)
        
        std = math.sqrt(2 / (in_features + out_features))
        lo = -3 * std
        hi = 3 * std
        nn.init.trunc_normal_(self.weight, std=std, a=lo, b=hi)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
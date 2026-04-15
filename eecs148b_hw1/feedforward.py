from eecs148b_hw1.linear import Linear

from torch import nn, Tensor

import torch

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.device = device
        self.dtype = dtype

        self.fc1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = torch.clamp(x, min=0)
        x = self.fc2(x)
        return x
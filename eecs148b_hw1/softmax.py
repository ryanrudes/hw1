from torch import Tensor

import torch

def softmax(x: Tensor, dim: int) -> Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / x.sum(dim=dim, keepdim=True)
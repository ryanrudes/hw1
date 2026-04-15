from eecs148b_hw1.softmax import softmax

from torch import Tensor

import torch
import math

def sdpa(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    # Q = (B, ..., T, d_k)
    # K = (B, ..., T, d_k)
    # V = (B, ..., T, d_v)
    # mask = (T, T)
    # return (B, ..., T, d_v)
    d_k = Q.shape[-1]
    x = Q @ K.transpose(-2, -1) # (B, ..., T, T)
    x = x / math.sqrt(d_k)
    if mask is not None:
        x = x.masked_fill(~mask, float('-inf'))
    x = softmax(x, dim=-1)
    return x @ V
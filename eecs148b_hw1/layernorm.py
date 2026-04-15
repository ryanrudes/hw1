from torch import nn, Tensor

import torch

class LayerNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        self.eps = eps
        #self.register_buffer("eps", torch.tensor(eps, device=device, dtype=dtype))
        
    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1, keepdim=True) # (N, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # (N, T, 1)
        result = (x - mean) / torch.sqrt(var + self.eps)
        result = result * self.weight + self.bias
        return result.to(in_dtype)
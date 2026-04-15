from torch import nn, Tensor

import torch

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        position_indices = torch.arange(max_seq_len, device=device, dtype=dtype)
        embedding_indices = torch.arange(d_model // 2, device=device, dtype=dtype)

        i = embedding_indices.unsqueeze(0)
        p = position_indices.unsqueeze(1)
        PE = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        PE[:, 0::2] = torch.sin(p / (10000 ** (2 * i / d_model)))
        PE[:, 1::2] = torch.cos(p / (10000 ** (2 * i / d_model)))
        self.register_buffer("PE", PE, persistent=False)

    def forward(self, token_positions: Tensor) -> Tensor:
        return self.PE[token_positions]
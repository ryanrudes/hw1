from eecs148b_hw1.layernorm import LayerNorm
from eecs148b_hw1.feedforward import FeedForward
from eecs148b_hw1.attention import CausalMultiheadSelfAttention

from torch import nn, Tensor

import torch

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        ln_eps: float = 1e-5,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln_eps = ln_eps
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.ln1 = LayerNorm(d_model, eps=ln_eps, device=device, dtype=dtype)
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, eps=ln_eps, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attn(self.ln1(x))
        y = y + self.ffn(self.ln2(y))
        return y
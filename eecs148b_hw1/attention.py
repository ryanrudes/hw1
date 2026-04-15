from eecs148b_hw1.sdpa import sdpa
from eecs148b_hw1.linear import Linear

from torch import nn, Tensor

import torch

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}")
        
        self.d_k = self.d_v = d_model // num_heads

        # Just one matrix multiplication for all heads
        #self.qkv = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len is not None:
            J = torch.ones(max_seq_len, max_seq_len, device=device, dtype=bool)
            mask = torch.tril(J)
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, d_model = x.shape

        #qkv = self.qkv(x) # (B, T, 3 * d_model)
        #qkv = qkv.reshape(B, T, self.num_heads, 3 * self.d_k) # (B, T, num_heads, 3 * d_k)
        #qkv = qkv.transpose(1, 2) # (B, num_heads, T, 3 * d_k)
        #q, k, v = qkv.chunk(3, dim=-1) # (B, num_heads, T, d_k), (B, num_heads, T, d_k), (B, num_heads, T, d_v)
        
        q = self.q_proj(x) # (B, T, d_model)
        k = self.k_proj(x) # (B, T, d_model)
        v = self.v_proj(x) # (B, T, d_model)
        q = q.reshape(B, T, self.num_heads, self.d_k) # (B, T, num_heads, d_k)
        k = k.reshape(B, T, self.num_heads, self.d_k) # (B, T, num_heads, d_k)
        v = v.reshape(B, T, self.num_heads, self.d_v) # (B, T, num_heads, d_v)
        q = q.transpose(1, 2) # (B, num_heads, T, d_k)
        k = k.transpose(1, 2) # (B, num_heads, T, d_k)
        v = v.transpose(1, 2) # (B, num_heads, T, d_v)

        if self.mask is None:
            J = torch.ones(T, T, device=self.device, dtype=bool)
            mask = torch.tril(J)
        else:
            mask = self.mask[:T, :T]

        x = sdpa(q, k, v, mask) # (B, num_heads, T, d_v)
        x = x.transpose(1, 2) # (B, T, num_heads, d_v)
        x = x.reshape(B, T, d_model) # (B, T, d_model)
        o = self.output_proj(x) # (B, T, d_model)
        return o
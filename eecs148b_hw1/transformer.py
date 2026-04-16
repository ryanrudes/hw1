from eecs148b_hw1.embedding import Embedding
from eecs148b_hw1.positional_embedding import SinusoidalPositionalEncoding
from eecs148b_hw1.transformer_block import TransformerBlock
from eecs148b_hw1.layernorm import LayerNorm
from eecs148b_hw1.linear import Linear

from torch import nn, Tensor

import torch

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        ln_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln_eps = ln_eps
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.positional_embedding = SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff=d_ff, ln_eps=ln_eps, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = LayerNorm(d_model, eps=ln_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.token_positions = torch.arange(context_length, device=device, dtype=torch.long)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape

        # x = (B, T)
        x = self.token_embeddings(x) # (B, T, d_model)
        
        token_positions = self.token_positions[:T] # (T,)
        pos_embeds = self.positional_embedding(token_positions) # (T, d_model)
        x = x + pos_embeds # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x) # (B, T, d_model)
        return self.lm_head(x) # (B, T, vocab_size)
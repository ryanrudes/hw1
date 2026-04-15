import numpy as np
import torch

def sample_batch(
    token_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample a batch of sequences from the token_ids array
    indices = np.random.randint(0, len(token_ids) - context_length, batch_size, dtype=np.int64)
    offset = np.arange(context_length + 1, dtype=np.int64)
    # (B, 1) + (1, T + 1) = (B, T + 1)
    idx_matrix = indices[:, None] + offset[None, :]
    # (B, T + 1)
    batch = token_ids[idx_matrix]
    # (B, T)
    inputs = torch.from_numpy(batch[:, :-1]).to(device)
    targets = torch.from_numpy(batch[:, 1:]).to(device)
    return inputs, targets
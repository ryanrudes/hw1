import torch

from torch import Tensor

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # logits = (..., vocab_size)
    # targets = (...,)
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    exp_logits_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    neg_log_probs = max_logits + torch.log(exp_logits_sum) - logits.gather(dim=-1, index=targets.unsqueeze(-1))
    return torch.mean(neg_log_probs)
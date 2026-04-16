from eecs148b_hw1.transformer import Transformer
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.softmax import softmax

from torch import Tensor

import torch

def generate_text(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> str:
    """
    Generate text using the model.
    """
    # Get previous model eval/train state
    previous_mode = model.training
    model.eval()
    device = model.device
    with torch.inference_mode():
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        while input_ids.shape[-1] < max_tokens:
            output = model(input_ids)
            logits = output[0, -1] / temperature
            probs = softmax(logits, dim=-1)

            # Do top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_remove_mask = cumulative_probs > top_p

            # Shift right by 1 to keep the first token where cumulative >= p
            sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
            sorted_remove_mask[..., 0] = False

            filtered_sorted_probs = sorted_probs.masked_fill(sorted_remove_mask, 0.0)

            # Renormalize over the kept set
            denom = filtered_sorted_probs.sum(dim=-1, keepdim=True)

            # Denominator shouldn't be zero in theory but floating-point issues
            # can happen, so be defensive
            fallback = torch.zeros_like(filtered_sorted_probs)
            fallback[..., 0] = 1.0

            normalized_sorted_probs = torch.where(
                denom > 0,
                filtered_sorted_probs / denom,
                fallback,
            )

            # Sample in sorted space
            sampled_sorted_pos = torch.multinomial(
                normalized_sorted_probs.reshape(-1, normalized_sorted_probs.shape[-1]),
                num_samples=1,
            ).reshape(normalized_sorted_probs.shape[:-1])

            # Map sampled sorted positions back to original vocab indices
            sampled_token = torch.gather(
                sorted_indices,
                dim=-1,
                index=sampled_sorted_pos.unsqueeze(-1),
            ).squeeze(-1)

            # Terminate if we've generated the end of text token
            if sampled_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, sampled_token.unsqueeze(0).unsqueeze(0)], dim=-1)

    # Restore previous model eval/train state
    model.train(previous_mode)
    return tokenizer.decode(input_ids[0].tolist())

def stream_text(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> str:
    """
    Generate text using the model.
    """
    # Get previous model eval/train state
    previous_mode = model.training
    model.eval()
    device = model.device
    with torch.inference_mode():
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        while input_ids.shape[-1] < max_tokens:
            output = model(input_ids)
            logits = output[0, -1] / temperature
            probs = softmax(logits, dim=-1)

            # Do top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_remove_mask = cumulative_probs > top_p

            # Shift right by 1 to keep the first token where cumulative >= p
            sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
            sorted_remove_mask[..., 0] = False

            filtered_sorted_probs = sorted_probs.masked_fill(sorted_remove_mask, 0.0)

            # Renormalize over the kept set
            denom = filtered_sorted_probs.sum(dim=-1, keepdim=True)

            # Denominator shouldn't be zero in theory but floating-point issues
            # can happen, so be defensive
            fallback = torch.zeros_like(filtered_sorted_probs)
            fallback[..., 0] = 1.0

            normalized_sorted_probs = torch.where(
                denom > 0,
                filtered_sorted_probs / denom,
                fallback,
            )

            # Sample in sorted space
            sampled_sorted_pos = torch.multinomial(
                normalized_sorted_probs.reshape(-1, normalized_sorted_probs.shape[-1]),
                num_samples=1,
            ).reshape(normalized_sorted_probs.shape[:-1])

            # Map sampled sorted positions back to original vocab indices
            sampled_token = torch.gather(
                sorted_indices,
                dim=-1,
                index=sampled_sorted_pos.unsqueeze(-1),
            ).squeeze(-1)

            # Terminate if we've generated the end of text token
            if sampled_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, sampled_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            yield tokenizer.decode([sampled_token.item()])

    # Restore previous model eval/train state
    model.train(previous_mode)
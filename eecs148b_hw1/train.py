from eecs148b_hw1.transformer import Transformer
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.data_loading import sample_batch
from eecs148b_hw1.cross_entropy import cross_entropy

import typer
from typing_extensions import Annotated

import numpy as np
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEFAULT_TRAIN_PATH = "tokenizer/train_encodings.npy"
DEFAULT_VALID_PATH = "tokenizer/valid_encodings.npy"
DEFAULT_MERGES = "tokenizer/merges.json"
DEFAULT_VOCAB = "tokenizer/vocab.json"
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_D_MODEL = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_LN_EPS = 1e-5
DEFAULT_LR = 3e-4
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.98
DEFAULT_OPTIMIZER_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_DEVICE = get_device()
DEFAULT_DTYPE_STR = "float32"
DEFAULT_NUM_STEPS = 1000
DEFAULT_BATCH_SIZE = 32

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _parse_dtype(name: str) -> torch.dtype:
    key = name.lower().removeprefix("torch.")
    if key not in _DTYPE_BY_NAME:
        allowed = ", ".join(sorted(_DTYPE_BY_NAME))
        raise typer.BadParameter(f"Unknown dtype {name!r}; expected one of: {allowed}")
    return _DTYPE_BY_NAME[key]


def train(
    train_path: Annotated[str, typer.Option(help="Path to the training data")] = DEFAULT_TRAIN_PATH,
    valid_path: Annotated[str, typer.Option(help="Path to the validation data")] = DEFAULT_VALID_PATH,
    context_length: Annotated[int, typer.Option(help="Context length")] = DEFAULT_CONTEXT_LENGTH,
    num_layers: Annotated[int, typer.Option(help="Number of layers")] = DEFAULT_NUM_LAYERS,
    d_model: Annotated[int, typer.Option(help="Size of the model embeddings")] = DEFAULT_D_MODEL,
    num_heads: Annotated[int, typer.Option(help="Number of heads")] = DEFAULT_NUM_HEADS,
    d_ff: Annotated[int | None, typer.Option(help="Size of the feedforward layer")] = None,
    ln_eps: Annotated[float, typer.Option(help="Epsilon for the layer norm")] = DEFAULT_LN_EPS,
    lr: Annotated[float, typer.Option(help="Learning rate")] = DEFAULT_LR,
    beta1: Annotated[float, typer.Option(help="Beta1 for the optimizer")] = DEFAULT_BETA1,
    beta2: Annotated[float, typer.Option(help="Beta2 for the optimizer")] = DEFAULT_BETA2,
    weight_decay: Annotated[float, typer.Option(help="Weight decay")] = DEFAULT_WEIGHT_DECAY,
    optimizer_eps: Annotated[float, typer.Option(help="Optimizer epsilon")] = DEFAULT_OPTIMIZER_EPS,
    num_steps: Annotated[int, typer.Option(help="Number of steps")] = DEFAULT_NUM_STEPS,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = DEFAULT_BATCH_SIZE,
    device: Annotated[str, typer.Option(help="Device to use (cpu, cuda, mps)")] = DEFAULT_DEVICE,
    dtype: Annotated[str, typer.Option(help="Data type: float32, float16, bfloat16, float64")] = DEFAULT_DTYPE_STR,
):
    torch_dtype = _parse_dtype(dtype)

    tokenizer = Tokenizer.from_files(DEFAULT_VOCAB, DEFAULT_MERGES)
    
    vocab_size = len(tokenizer.vocab)
    
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        ln_eps=ln_eps,
        device=device,
        dtype=torch_dtype,
    )

    train_data = np.memmap(train_path, dtype=np.uint32)
    valid_data = np.memmap(valid_path, dtype=np.uint32)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=optimizer_eps,
        weight_decay=weight_decay,
    )

    model.to(device)
    model.train()

    for step in range(num_steps):
        inputs, targets = sample_batch(train_data, batch_size, context_length, device)

        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}, Loss: {loss.item()}")
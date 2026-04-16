from eecs148b_hw1.transformer import Transformer
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.data_loading import sample_batch
from eecs148b_hw1.cross_entropy import cross_entropy
from eecs148b_hw1.decoding import generate_text

import typer
from typing_extensions import Annotated

import numpy as np
import torch
import os

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEFAULT_TRAIN_PATH = "tokenizer/train_encodings.npy"
DEFAULT_VALID_PATH = "tokenizer/valid_encodings.npy"
DEFAULT_MERGES = "tokenizer/merges.pkl"
DEFAULT_VOCAB = "tokenizer/vocab.pkl"
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_GENERATION_LENGTH = 256
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
DEFAULT_NUM_STEPS = 100000
DEFAULT_BATCH_SIZE = 32

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}

EVAL_PROMPTS = [
    "Once upon a time",
    "There once was a man who",
    "In the year 1999, a new virus was discovered",
    "To be or not to be,",
    "Large language models (LLMs) began being developed in",
    "The quick brown fox",
]


def _parse_dtype(name: str) -> torch.dtype:
    key = name.lower().removeprefix("torch.")
    if key not in _DTYPE_BY_NAME:
        allowed = ", ".join(sorted(_DTYPE_BY_NAME))
        raise typer.BadParameter(f"Unknown dtype {name!r}; expected one of: {allowed}")
    return _DTYPE_BY_NAME[key]


def train(
    train_path: Annotated[str, typer.Option(help="Path to the training data")] = DEFAULT_TRAIN_PATH,
    valid_path: Annotated[str, typer.Option(help="Path to the validation data")] = DEFAULT_VALID_PATH,
    vocab_path: Annotated[str, typer.Option(help="Path to the vocabulary")] = DEFAULT_VOCAB,
    merges_path: Annotated[str, typer.Option(help="Path to the merges")] = DEFAULT_MERGES,
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
    wandb: Annotated[bool, typer.Option(help="Use wandb")] = False,
    val_every: Annotated[int, typer.Option(help="Validate every n steps")] = 100,
    generation_length: Annotated[int, typer.Option(help="Generation length")] = DEFAULT_GENERATION_LENGTH,
    device: Annotated[str, typer.Option(help="Device to use (cpu, cuda, mps)")] = DEFAULT_DEVICE,
    dtype: Annotated[str, typer.Option(help="Data type: float32, float16, bfloat16, float64")] = DEFAULT_DTYPE_STR,
):
    if val_every <= 0:
        raise typer.BadParameter("val_every must be greater than 0")

    if wandb:
        import wandb

        run = wandb.init(project="eecs148b_hw1", config=wandb.config)

        run.config.update({
            "train_path": train_path,
            "valid_path": valid_path,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "ln_eps": ln_eps,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "optimizer_eps": optimizer_eps,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
        })

        run.define_metric(
            name="loss/train",
            step_metric="step",
            step_sync=True,
        )

        run.define_metric(
            name="loss/val",
            step_metric="step",
            step_sync=True,
        )

        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training on {device}")

    torch_dtype = _parse_dtype(dtype)

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    
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

    train_data = np.memmap(train_path, dtype=np.uint16).astype(np.long)
    valid_data = np.memmap(valid_path, dtype=np.uint16).astype(np.long)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=optimizer_eps,
        weight_decay=weight_decay,
    )

    model.to(device)
    model.train()

    if wandb:
        wandb.watch(model, log="all", log_freq=100)

    for step in range(num_steps):
        inputs, targets = sample_batch(train_data, batch_size, context_length, device)

        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}, Loss: {loss.item()}")

        if wandb:
            wandb.log({
                "loss/train": loss.item(),
                "step": step,
            })

        if step == 0 or (step + 1) % val_every == 0:
            model.eval()

            with torch.inference_mode():
                val_inputs, val_targets = sample_batch(valid_data, batch_size, context_length, device)
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_targets)
                print(f"Validation Loss: {val_loss.item()}")
                if wandb:
                    wandb.log({
                        "loss/val": val_loss.item(),
                        "step": step,
                    })

            table = wandb.Table(columns=["prompt", "generation"])

            for prompt in EVAL_PROMPTS:
                generation = generate_text(model, tokenizer, prompt, generation_length)
                table.add_data(prompt, generation)

            wandb.log({
                "eval/table": table,
                "step": step,
            })

            model.train()

            torch.save(model.state_dict(), f"{checkpoint_dir}/model_{step}.pt")
            #wandb.save(f"{checkpoint_dir}/model_{step}.pth")

    if wandb:
        torch.save(model.state_dict(), f"{checkpoint_dir}/final.pt")
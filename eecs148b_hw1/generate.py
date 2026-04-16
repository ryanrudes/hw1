from eecs148b_hw1.transformer import Transformer
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.decoding import stream_text
from eecs148b_hw1.train import get_device, _parse_dtype

import typer
from typing_extensions import Annotated

import torch

from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.console import Console
from rich.theme import Theme

console = Console(theme=Theme({
    "info": "dim cyan",
    "warning": "dim yellow",
    "danger": "bold red",
}))

DEFAULT_MERGES = "tokenizer/merges.pkl"
DEFAULT_VOCAB = "tokenizer/vocab.pkl"
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_GENERATION_LENGTH = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_D_MODEL = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_LN_EPS = 1e-5
DEFAULT_DEVICE = get_device()
DEFAULT_DTYPE_STR = "float32"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95

def generate(
    model_path: Annotated[str, typer.Option(help="Path to the model")],
    vocab_path: Annotated[str, typer.Option(help="Path to the tokenizer")] = DEFAULT_VOCAB,
    merges_path: Annotated[str, typer.Option(help="Path to the merges")] = DEFAULT_MERGES,
    generation_length: Annotated[int, typer.Option(help="Generation length")] = DEFAULT_GENERATION_LENGTH,
    temperature: Annotated[float, typer.Option(help="Temperature")] = DEFAULT_TEMPERATURE,
    top_p: Annotated[float, typer.Option(help="Top-p")] = DEFAULT_TOP_P,
    context_length: Annotated[int, typer.Option(help="Context length")] = DEFAULT_CONTEXT_LENGTH,
    num_layers: Annotated[int, typer.Option(help="Number of layers")] = DEFAULT_NUM_LAYERS,
    d_model: Annotated[int, typer.Option(help="Size of the model embeddings")] = DEFAULT_D_MODEL,
    num_heads: Annotated[int, typer.Option(help="Number of heads")] = DEFAULT_NUM_HEADS,
    d_ff: Annotated[int | None, typer.Option(help="Size of the feedforward layer")] = None,
    ln_eps: Annotated[float, typer.Option(help="Epsilon for the layer norm")] = DEFAULT_LN_EPS,
    device: Annotated[str, typer.Option(help="Device to use (cpu, cuda, mps)")] = DEFAULT_DEVICE,
    dtype: Annotated[str, typer.Option(help="Data type: float32, float16, bfloat16, float64")] = DEFAULT_DTYPE_STR,
):
    """
    Generate text using the model.
    """
    console.print("[info]Loading model...")
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

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    model.float()
    model.eval()
    model.to(device)
    
    console.print("[info]Model loaded successfully")
    
    console.print("[info]Generating text...")
    prompt = Prompt.ask("[info]Enter a prompt")

    # First print the prompt in bold
    console.print(f"[bold]{prompt}[/bold]", end="")

    # Then print the text stream in italic
    for token in stream_text(model, tokenizer, prompt, generation_length, temperature, top_p):
        console.print(f"[italic]{token}[/italic]", end="")
    
    console.print()
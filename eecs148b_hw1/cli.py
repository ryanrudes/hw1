import typer

from eecs148b_hw1.train import train

app = typer.Typer(help="EECS 148B homework 1 CLI.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Entry point; use a subcommand such as ``train``."""
    pass


app.command()(train)


if __name__ == "__main__":
    app()
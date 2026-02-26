"""Main entrypoint for the CLI application."""

import logging
import warnings

import typer
from hydra.core.utils import simple_stdout_log_config

from icenet_mp.plugins import register_plugins

from .datasets import datasets_cli
from .evaluate import evaluation_cli
from .train import training_cli

# Configure logging
simple_stdout_log_config()
logger = logging.getLogger(__name__)

# Register all plugins
register_plugins()

# Ignore warnings about known PyTorch issues
warnings.filterwarnings(
    "ignore",
    message=".*Using padding='same' with even kernel lengths and odd dilation.*",
)

# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for imp CLI application.",
    no_args_is_help=True,
)
app.add_typer(datasets_cli, name="datasets")
app.add_typer(evaluation_cli)
app.add_typer(training_cli)


def main() -> None:
    """Initialise and run the CLI application."""
    # Run the app
    try:
        app()
    except NotImplementedError as exc:
        # Catch MPS-not-implemented errors
        if "not currently implemented for the MPS device" in str(exc):
            msg = (
                "WARNING: job failed due to running on MPS without CPU fallback enabled.\n"
                "Please rerun after setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`. "
                "This *must* be set before starting the Python interpreter. "
                "It will be slower than running natively on MPS."
            )
            logger.error(msg)  # noqa: TRY400
            typer.Exit(1)


if __name__ == "__main__":
    main()

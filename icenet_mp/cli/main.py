"""Main entrypoint for the CLI application."""

import logging

import typer

from icenet_mp.compatibility import configure_external_libraries

from .datasets import datasets_cli
from .eof import eof_app
from .evaluate import evaluation_cli
from .input_diagnostics import input_diag_app
from .input_explainability import input_exp_app
from .pca import pca_app
from .pre_feature_analysis import pre_feature_app
from .rf import rf_app
from .train import training_cli
from .vif import vif_app

# Configure logging
logging.basicConfig(
    format="😈 [%(asctime)s] %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)
log = logging.getLogger(__name__)

# Configure external libraries
configure_external_libraries()


# Create the typer app
app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Entrypoint for imp CLI application.",
    no_args_is_help=True,
)
app.add_typer(datasets_cli, name="datasets")
app.add_typer(evaluation_cli)
app.add_typer(training_cli)
app.add_typer(vif_app)
app.add_typer(pre_feature_app, name="pre-feature-analysis")
app.add_typer(input_diag_app, name="input-diagnostics")
app.add_typer(input_exp_app, name="input-explainability")
app.add_typer(pca_app)
app.add_typer(eof_app)
app.add_typer(rf_app)


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
            log.error(msg)  # noqa: TRY400
            typer.Exit(1)


if __name__ == "__main__":
    main()

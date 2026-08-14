import inspect
import itertools
from collections.abc import Callable
from typing import Annotated, ParamSpec, TypeVar

from hydra import compose, initialize
from omegaconf import DictConfig
from typer import Argument, Option

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def hydra_adaptor(function: Callable) -> Callable[Param, RetType]:
    """Replace a function that takes a Hydra config with one that takes string arguments.

    Args:
        function: Callable(*args, config: DictConfig, **kwargs). If the function also
            declares a plain ``config_name`` parameter (not annotated as a typer Option),
            it receives the resolved config name string alongside ``config`` — useful for
            deriving output paths from the config being run — without adding a second
            ``--config-name`` option to the CLI.

    Returns:
        Callable(*args, config_name: str, **kwargs, overrides: list[str])

    """
    wants_config_name = "config_name" in inspect.signature(function, eval_str=True).parameters

    def wrapper(
        overrides: Annotated[
            list[str] | None,
            Argument(
                help="One or more space-separated Hydra config overrides (https://hydra.cc/docs/advanced/override_grammar/basic/)"
            ),
        ] = None,
        config_name: Annotated[
            str | None,
            Option(help="Name of a file to load from the config directory"),
        ] = "sample",
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        with initialize(config_path="../config", version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
        if wants_config_name:
            kwargs["config_name"] = config_name
        return function(*args, config=config, **kwargs)

    # Separate parameters by kind
    positional_params = []
    keyword_only_params = []

    # Remove the DictConfig parameter from the function signature
    fn_signature = inspect.signature(function, eval_str=True)
    for param in fn_signature.parameters.values():
        if param.annotation == DictConfig:
            continue  # skip config param
        if param.name == "config_name":
            continue  # forwarded directly above, not exposed as a second CLI option
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword_only_params.append(param)
        else:
            positional_params.append(param)

    # Take only the overrides and config_name names from the function signature
    additional_params = (
        param
        for param in inspect.signature(wrapper, eval_str=True).parameters.values()
        if param.name in ("overrides", "config_name")
    )

    # Combine in correct order: positional, then additional, then keyword-only
    combined_parameters = list(
        itertools.chain(positional_params, additional_params, keyword_only_params)
    )
    wrapper.__signature__ = fn_signature.replace(parameters=combined_parameters)  # type: ignore[attr-defined]
    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper  # type: ignore[return-value]

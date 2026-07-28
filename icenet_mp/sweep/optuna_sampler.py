from pathlib import Path
from typing import Any, ClassVar

import yaml
from optuna import Trial, create_study
from optuna.samplers import BaseSampler, QMCSampler, RandomSampler, TPESampler


class OptunaSampler:
    sampler_map: ClassVar[dict[str, type[BaseSampler]]] = {
        "qmc": QMCSampler,
        "tpe": TPESampler,
        "random": RandomSampler,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize an OptunaSampler from a parsed YAML dict."""
        self.name: str = config["name"]
        self.n_trials: int = config["n_trials"]
        self.seed: int = config.get("seed", 0)
        self.metric = {
            "name": config.get("metric", {}).get("name", "validation_loss"),
            "goal": config.get("metric", {}).get("goal", "minimize"),
        }
        self.parameters: dict[str, Any] = config["parameters"]

        try:
            self.sampler = self.sampler_map[config["sampler"]]
        except KeyError as exc:
            msg = (
                f"Unknown sampler '{config['sampler']}', expected one of "
                f"{self.sampler_map.keys()}"
            )
            raise ValueError(msg) from exc

    @classmethod
    def from_yaml(cls, path: Path) -> "OptunaSampler":
        """Load a sweep config from a YAML file."""
        return cls(yaml.safe_load(path.read_text()))

    def generate_sweep_config(
        self, trials: list[dict[str, int | float | str]]
    ) -> dict[str, Any]:
        """Generate a W&B sweep config over a fixed batch of trials.

        Args:
            trials: The sampled hyperparameter combinations fro `generate_trials`.

        Returns:
            A dict suitable for writing to a W&B sweep YAML file.

        """
        return {
            "program": "imp",
            "method": "grid",
            "metric": self.metric,
            "parameters": {"trial-number": {"values": list(range(len(trials)))}},
        }

    def generate_trials(self) -> list[dict[str, int | float | str]]:
        """Sample a fixed batch of hyperparameter combinations from the search-space.

        Returns:
            One dict of {override_path: value} per trial.

        """
        sampler = self.sampler(seed=self.seed)
        study = create_study(sampler=sampler, direction=self.metric["goal"])
        trials = []
        for _ in range(self.n_trials):
            trial = study.ask()
            trials.append(
                {
                    name: self.suggest_param(trial, name, param_spec)
                    for name, param_spec in self.parameters.items()
                }
            )
        return trials

    def suggest_param(
        self, trial: Trial, name: str, param_spec: dict[str, Any]
    ) -> int | float | str:
        """Sample a single parameter value from a trial according to its search-space spec."""
        param_type = param_spec["type"]
        log = param_spec.get("log", False)
        if param_type == "categorical":
            return trial.suggest_categorical(name, param_spec["choices"])
        if param_type == "float":
            return trial.suggest_float(
                name,
                param_spec["low"],
                param_spec["high"],
                log=log,
                step=param_spec.get("step"),
            )
        if param_type == "int":
            return trial.suggest_int(
                name,
                param_spec["low"],
                param_spec["high"],
                log=log,
                step=param_spec.get("step", 1),
            )
        msg = f"Unknown parameter type '{param_type}' for parameter '{name}'"
        raise ValueError(msg)

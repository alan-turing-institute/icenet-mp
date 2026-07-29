import logging
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from optuna import Study, create_study
from optuna.samplers import BaseSampler, QMCSampler, RandomSampler, TPESampler
from optuna.trial import FrozenTrial, Trial, TrialState

from .parameters import build_parameter

log = logging.getLogger(__name__)


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
        self.sampler_cls = config["sampler"]
        self._entity: str | None = config.get("entity")
        self._study: Study | None = None
        self._study_name: str | None = None
        self._study_path: Path | None = None

    @classmethod
    def from_path(cls, study_path: Path) -> "OptunaSampler":
        """Load a sweep config from a study path."""
        yaml_path = study_path / "optuna.yaml"
        instance = cls.from_yaml(yaml_path)
        instance._study_path = study_path
        instance._study_name = study_path.name
        return instance

    @classmethod
    def from_yaml(cls, path: Path) -> "OptunaSampler":
        """Load a sweep config from a YAML file."""
        return cls(yaml.safe_load(path.read_text()))

    @property
    def entity(self) -> str:
        if self._entity is None:
            msg = "W&B entity has not been set."
            raise ValueError(msg)
        return self._entity

    @property
    def study_name(self) -> str:
        if self._study_name is None:
            msg = "Study name has not been set."
            raise ValueError(msg)
        return self._study_name

    @property
    def study_path(self) -> Path:
        if self._study_path is None:
            msg = "Study path has not been set."
            raise ValueError(msg)
        return self._study_path

    @property
    def study(self) -> Study:
        if self._study is None:
            # Create or load the sampler
            sampler_path = self.study_path / "sampler.pkl"
            try:
                with sampler_path.open("rb") as f_sampler:
                    sampler = pickle.load(f_sampler)  # noqa: S301
            except (FileNotFoundError, EOFError):
                log.debug("Sampler could not be loaded from %s.", sampler_path)
                sampler_cls = self.sampler_map.get(self.sampler_cls)
                if sampler_cls is None:
                    msg = (
                        f"Unknown sampler '{self.sampler_cls}', expected one of "
                        f"{self.sampler_map.keys()}"
                    )
                    raise ValueError(msg) from None
                sampler = sampler_cls(seed=self.seed)  # type: ignore[call-arg]

            # Create or load the study
            self._study = create_study(
                direction=self.metric["goal"],
                load_if_exists=True,
                sampler=sampler,
                storage=f"sqlite:///{self.study_path / 'optuna.db'}",
                study_name=self.study_name,
            )
        return self._study

    def _save_sampler(self) -> None:
        """Persist a study's sampler state to disk.

        This must be called after each `ask()` or `tell()` operation on the underlying
        Optuna study, since these calls trigger changes on the sampler. Without this,
        every call to `ask()` will return the same trial.
        """
        sampler_path = self.study_path / "sampler.pkl"
        with sampler_path.open("wb") as f_sampler:
            pickle.dump(self.study.sampler, f_sampler)

    def ask(self) -> Trial:
        """Ask the study for a new trial."""
        trial = self.study.ask()
        self._save_sampler()
        return trial

    def initialise_study(self, model_cfg: DictConfig, sweep_id: str) -> None:
        """Create an Optuna study and save the sampled trials to a JSON file."""
        # Generate study and storage paths
        sweep_base = (Path(model_cfg.get("base_path"))).resolve() / "sweeps"
        self._study_name = sweep_id
        self._study_path = (sweep_base / self.study_name).resolve()
        self.study_path.mkdir(parents=True, exist_ok=True)

        # Write the Optuna config to disk
        optuna_cfg = {
            "entity": self.entity,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "name": self.name,
            "parameters": self.parameters,
            "sampler": self.sampler_cls,
            "seed": self.seed,
        }
        with (self.study_path / "optuna.yaml").open("w") as f_yaml:
            yaml.dump(optuna_cfg, f_yaml, default_flow_style=False)

        # Save the model config to the study path
        OmegaConf.save(model_cfg, self.study_path / "model_config.yaml")

    def initialise_sweep(self, model_cfg: DictConfig) -> str:
        """Generate a new W&B sweep."""
        # Generate the W&B sweep config
        sweep_config = {
            "program": "imp",
            "method": "random",
            "metric": self.metric,
            "parameters": {
                k: v
                for name, param_spec in self.parameters.items()
                for k, v in build_parameter(name, param_spec).sweep_cfg().items()
            },
        }
        self._entity = (
            model_cfg.get("loggers", {})
            .get("wandb", {})
            .get("entity", os.environ.get("WANDB_ENTITY", None))
        )
        # Start the sweep
        return wandb.sweep(sweep_config, entity=self.entity, project="train")

    def generate_trial_config(self, trial: Trial) -> DictConfig:
        return OmegaConf.from_dotlist(
            [
                f"{name}={build_parameter(name, param_spec).suggest(trial)}"
                for name, param_spec in self.parameters.items()
            ]
        )

    def tell(
        self,
        trial: Trial | int,
        values: float | Sequence[float] | None = None,
        *,
        state: TrialState | None = None,
        skip_if_finished: bool = False,
    ) -> FrozenTrial:
        """Ask the study for a new trial."""
        result = self.study.tell(trial, values, state, skip_if_finished)
        self._save_sampler()
        return result

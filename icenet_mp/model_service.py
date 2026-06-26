import logging
import os
from pathlib import Path
from typing import cast

import hydra
import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id

from icenet_mp.callbacks import PlottingCallback, UnconditionalCheckpoint
from icenet_mp.compatibility.torch import patch_interpolate_antialias
from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.models import BaseModel, EncodeProcessDecode
from icenet_mp.models.stages import DecoderStage, EncoderStage, ProcessorStage
from icenet_mp.types import SupportsMetadata
from icenet_mp.utils import get_device_name, get_timestamp, get_wandb_run

log = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config: DictConfig) -> None:
        """Initialize the model service."""
        self.config_ = config

        random_config = config.get("random", {})
        seed = random_config.get("seed", None)
        fully_deterministic = random_config.get("fully_deterministic", False)

        if seed is not None:
            seed = int(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            seed_everything(seed, workers=True)

        # If we are in fully deterministic mode, enable deterministic algorithms and
        # patch any known issues with them. We use warn_only=True to avoid segfaults on
        # unsupported operations.
        if fully_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)  # noqa: FBT003
            patch_interpolate_antialias()
            log.warning(
                "Fully deterministic mode enabled and anti-aliasing disabled. This may "
                "produce different results compared to non-deterministic mode and may "
                "also impact performance. Ensure this is intended before proceeding."
            )

        # Apply any patches necessary for MPS compatibility if appropriate.
        configured_accelerator = (
            config.get("train", {}).get("trainer", {}).get("accelerator", "auto")
        )
        if (
            configured_accelerator in ("mps", "auto")
            and torch.backends.mps.is_available()
        ):
            patch_interpolate_antialias()
            log.warning(
                "Anti-aliasing disabled to avoid known segmentation faults on MPS."
            )

        self.data_module_: CommonDataModule | None = None
        self.model_: BaseModel | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "ModelService":
        """Build a new ModelService by instantiating a model from a configuration."""
        # Load the model configuration
        builder = cls(config)

        # Construct the model
        OmegaConf.resolve(config["train"])  # resolve training config interpolations
        log.info("Building a new '%s' model...", builder.config["model"]["_target_"])
        builder.model_ = hydra.utils.instantiate(
            config["model"],
            hemisphere=builder.data_module.hemisphere,
            input_spaces=[s.to_dict() for s in builder.data_module.input_spaces],
            latitudes_fn=lambda: builder.data_module.latitudes,
            longitudes_fn=lambda: builder.data_module.longitudes,
            n_forecast_steps=builder.data_module.n_forecast_steps,
            n_history_steps=builder.data_module.n_history_steps,
            output_space=builder.data_module.output_space.to_dict(),
            optimizer=config["train"]["optimizer"],
            scheduler=config["train"]["scheduler"],
            loss=config["loss"],
            _recursive_=False,
            _convert_="object",
        )

        return builder

    @classmethod
    def from_checkpoint(
        cls, config: DictConfig, checkpoint_path: Path
    ) -> "ModelService":
        """Build a new ModelService by loading a model from a checkpoint."""
        # Verify the checkpoint path
        if checkpoint_path.is_file():
            log.debug("Found checkpoint at %s.", checkpoint_path)
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Build a combined model configuration where the command line config takes
        # precedence except for the "model", "predict" and "train" keys which are
        # related to training the model.
        config_path = checkpoint_path.parent.parent / "files" / "model_config.yaml"
        try:
            # Load the model configuration from the checkpoint directory
            ckpt_config = DictConfig(OmegaConf.load(config_path))
            log.debug("Loaded checkpoint configuration from %s.", config_path)
            combined_cfg = DictConfig(OmegaConf.merge(ckpt_config, config))
            for key in ("model", "predict", "train"):
                combined_cfg[key] = OmegaConf.merge(
                    combined_cfg.get(key, {}), ckpt_config.get(key, {})
                )
        except (NotADirectoryError, FileNotFoundError):
            combined_cfg = config
            log.debug("Could not load checkpoint configuration from %s.", config_path)

        # Load the model from checkpoint
        builder = cls(combined_cfg)
        model_cls: type[BaseModel] = hydra.utils.get_class(
            builder.config["model"]["_target_"]
        )
        log.info("Loading a trained %s model...", builder.config["model"]["name"])
        builder.model_ = model_cls.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            latitudes_fn=lambda: builder.data_module.latitudes,
            longitudes_fn=lambda: builder.data_module.longitudes,
        )

        return builder

    @property
    def config(self) -> DictConfig:
        """Get the full configuration."""
        if not self.config_:
            msg = "Model config has not been initialised."
            raise AttributeError(msg)
        return self.config_

    @property
    def data_module(self) -> CommonDataModule:
        """Get the data module instance."""
        if not self.data_module_:
            self.data_module_ = CommonDataModule(self.config)
        return self.data_module_

    @property
    def model(self) -> BaseModel:
        """Get the model instance."""
        if not self.model_:
            msg = "Model has not been initialised."
            raise AttributeError(msg)
        return self.model_

    def _fit(
        self,
        *,
        model: LightningModule | None = None,
        config: DictConfig,
        job_stage: str | None = None,
    ) -> Trainer:
        """Build a trainer and run trainer.fit() for the given config and stage.

        Args:
            model: Model to train. Defaults to ``self.model`` if not provided.
            config: Job-specific config section (e.g. ``self.config["train"]``).
            job_stage: Label passed to ``PlottingCallback.prefix`` and used in log messages.

        Returns:
            The trainer after fitting, so callers can save checkpoints or inspect
            training state (e.g. ``trainer.current_epoch``, ``trainer.global_step``).

        """
        log.info("Configuring model for %s.", job_stage or "training")
        current_model = model or self.model
        current_model.optimizer_cfg = config["optimizer"]
        current_model.scheduler_cfg = config["scheduler"]
        trainer = self.build_trainer(
            config=config, job_stage=job_stage, project="train"
        )
        log.info(
            "Starting %s for %d epochs using %d threads across %d %s device(s).",
            f"training {job_stage}" if job_stage else "training",
            trainer.max_epochs,
            torch.get_num_threads(),
            trainer.num_devices,
            get_device_name(trainer.accelerator.name()),
        )
        trainer.fit(model=current_model, datamodule=self.data_module)
        return trainer

    def _merged_config(self, stage_name: str) -> DictConfig:
        """Return train config merged with stage-specific overrides."""
        return cast(
            "DictConfig",
            OmegaConf.merge(
                self.config["train"],
                self.config["train"].get("stages", {}).get(stage_name, {}),
            ),
        )

    def _save_checkpoint(self, trainer: Trainer, stage_name: str) -> None:
        """Save a stage checkpoint at a predictable path."""
        checkpoint_path = (
            self.build_run_directory(trainer)
            / "checkpoints"
            / f"{stage_name}.epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        )
        trainer.save_checkpoint(checkpoint_path)
        log.info("Saved %s checkpoint to %s.", stage_name, checkpoint_path)

    def build_run_directory(self, trainer: Trainer) -> Path:
        """Get run directory from Wandb or generate one in the same format."""
        # Get the run directory from Wandb if it exists
        wandb_run = get_wandb_run(trainer)
        if wandb_run:
            return Path(wandb_run._settings.sync_dir)

        # Otherwise generate a new run directory
        return (
            self.data_module.base_path
            / "training"
            / "local"
            / f"run-{get_timestamp()}-{generate_id()}"
        )

    def build_trainer(
        self,
        *,
        config: DictConfig,
        project: str,
        job_stage: str | None = None,
    ) -> Trainer:
        """Configure the trainer with callbacks and loggers.

        Args:
            config: Job-specific config section (e.g. ``self.config["train"]``).
            project: W&B project name (one of "train" or "evaluate").
            job_stage: Optional label passed to ``PlottingCallback.prefix`` and used
                in log messages. Also sets the W&B ``job_type`` to ``"multi-stage"``
                when provided, or ``"single-stage"`` otherwise.

        """
        # Setup callbacks first
        callback_configs = config.get("callbacks", {}).values()
        extra_callbacks = [
            hydra.utils.instantiate(callback_config)
            for callback_config in callback_configs
        ]
        if not extra_callbacks:
            log.warning("No callbacks have been set for the trainer.")

        # Setup lightning loggers
        extra_loggers = [
            hydra.utils.instantiate(
                logger_config,
                job_type="multi-stage" if job_stage else "single-stage",
                project=project,
            )
            for logger_config in self.config.get("loggers", {}).values()
        ]
        if not extra_loggers:
            log.warning("No loggers have been set for the trainer.")

        # Create a new trainer
        log.debug("Instantiating lightning trainer.")
        trainer = cast(
            "Trainer",
            hydra.utils.instantiate(
                config["trainer"],
                callbacks=extra_callbacks,
                deterministic=self.config.get("random", {}).get(
                    "fully_deterministic", False
                ),
                logger=extra_loggers,
            ),
        )
        # Check warn_only survived Lightning's deterministic setup
        log.debug(
            "deterministic_algorithms_enabled: %s",
            torch.are_deterministic_algorithms_enabled(),
        )
        log.debug(
            "warn_only_enabled: %s",
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

        # Assign workers for data loading
        self.data_module.assign_workers(
            min(8, suggested_max_num_workers(trainer.num_devices))
        )

        # Ensure the run directory exists
        run_directory = self.build_run_directory(trainer)
        log.debug("Set run directory to %s.", run_directory)
        run_directory.mkdir(parents=True, exist_ok=True)

        # Save model config to the run directory
        model_config_path = run_directory / "files" / "model_config.yaml"
        if trainer.is_global_zero:
            model_config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.config, model_config_path)
            if wandb_run := get_wandb_run(trainer):
                wandb_run.save(model_config_path, base_path=model_config_path.parent)

        # Additional configuration for callbacks
        for callback in cast("list[Callback]", trainer.callbacks):  # type: ignore[attr-defined]
            log.debug("Configuring callback %s.", callback.__class__.__name__)
            # Set metadata for supported callbacks
            if isinstance(callback, SupportsMetadata):
                log.debug("Setting metadata for %s.", callback.__class__.__name__)
                callback.set_metadata(self.config, self.model.__class__.__name__)
            # Set plotting stage
            if isinstance(callback, PlottingCallback):
                log.debug(
                    "Setting plotting prefix for %s to %s.",
                    callback.__class__.__name__,
                    job_stage,
                )
                callback.prefix = job_stage
            # Set checkpoint run directory for supported callbacks
            if isinstance(callback, (ModelCheckpoint, UnconditionalCheckpoint)):
                log.debug(
                    "Setting run_directory for %s to %s.",
                    callback.__class__.__name__,
                    run_directory / "checkpoints",
                )
                callback.dirpath = run_directory / "checkpoints"

        return trainer

    def evaluate(self) -> None:
        """Evaluate a trained model."""
        # Configure the trainer with evaluation callbacks and loggers
        log.info("Configuring model for evaluation.")
        trainer = self.build_trainer(config=self.config["evaluate"], project="evaluate")
        # Log evaluation details
        log.info(
            "Starting evaluation using %d threads across %d %s device(s).",
            torch.get_num_threads(),
            trainer.num_devices,
            get_device_name(trainer.accelerator.name()),
        )

        # Evaluate the model
        trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )

    def train(
        self, *, checkpoint_dir: Path | None = None, multistage: bool = False
    ) -> None:
        """Train a model."""
        if multistage:
            self.train_multistage(checkpoint_dir=checkpoint_dir)
        else:
            self._fit(config=self.config["train"])

    def train_multistage(self, *, checkpoint_dir: Path | None = None) -> None:
        """Train an EncodeProcessDecode model in multiple stages: encoders → decoder → processor → finetuning."""
        if not isinstance(self.model, EncodeProcessDecode):
            msg = (
                "Multistage training is only supported for EncodeProcessDecode models."
            )
            raise TypeError(msg)

        log.info("Preparing to train the encoders...")
        trained_encoders = self.train_stage_encoders(
            config=self._merged_config("encoders"),
            checkpoint_dir=checkpoint_dir,
        )

        log.info("Preparing to train the decoder...")
        trained_decoder = self.train_stage_decoder(
            trained_encoders,
            config=self._merged_config("decoder"),
            checkpoint_dir=checkpoint_dir,
        )

        log.info("Preparing to train the processor...")
        processor_model = self.train_stage_processor(
            trained_decoder,
            config=self._merged_config("processor"),
            checkpoint_dir=checkpoint_dir,
        )

        log.info("Preparing to finetune...")
        self.train_stage_finetune(
            processor_model=processor_model,
            config=self._merged_config("finetuning"),
        )

    def train_stage_decoder(
        self,
        encoder_models: list[EncoderStage],
        *,
        config: DictConfig,
        checkpoint_dir: Path | None = None,
    ) -> DecoderStage:
        """Train a decoder on the combined latent space of all frozen encoders."""
        target_variables = (
            self.data_module.target_variables
            or self.data_module.variable_names[self.data_module.target_group_name]
        )
        target_variable_indices = [
            self.data_module.variable_names[self.data_module.target_group_name].index(v)
            for v in target_variables
        ]
        if checkpoint_dir is not None and (
            matches := sorted(checkpoint_dir.glob("decoder.epoch=*-step=*.ckpt"))
        ):
            checkpoint_path = matches[-1]
            log.info(
                "Skipping training for decoder. Loaded checkpoint from %s.",
                checkpoint_path,
            )
            return DecoderStage.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                decoder=self.config["model"]["decoder"],
                encoders=encoder_models,
                target_dataset_name=self.data_module.target_group_name,
                target_variable_indices=target_variable_indices,
            )

        decoder_model = DecoderStage.from_template(
            decoder=self.config["model"]["decoder"],
            encoders=encoder_models,
            target_dataset_name=self.data_module.target_group_name,
            target_variable_indices=target_variable_indices,
        )
        trainer = self._fit(model=decoder_model, config=config, job_stage="decoder")
        self._save_checkpoint(trainer, "decoder")
        return decoder_model

    def train_stage_encoders(
        self, *, config: DictConfig, checkpoint_dir: Path | None = None
    ) -> list[EncoderStage]:
        """Train each encoder separately with a disposable decoder."""
        if not isinstance(self.model, EncodeProcessDecode):
            msg = (
                "train_stage_encoders is only supported for EncodeProcessDecode models."
            )
            raise TypeError(msg)
        encoder_models = []
        for encoder in self.model.encoders:
            if checkpoint_dir is not None and (
                matches := sorted(
                    checkpoint_dir.glob(f"encoder-{encoder.name}.epoch=*-step=*.ckpt")
                )
            ):
                checkpoint_path = matches[-1]
                log.info(
                    "Skipping training for encoder '%s'. Loaded checkpoint from %s.",
                    encoder.name,
                    checkpoint_path,
                )
                encoder_models.append(
                    EncoderStage.load_from_checkpoint(
                        checkpoint_path,
                        weights_only=False,
                        latitudes_fn=lambda: self.data_module.latitudes,
                        longitudes_fn=lambda: self.data_module.longitudes,
                    )
                )
                continue

            encoder_model = EncoderStage.from_template(
                channel_names=self.data_module.variable_names[encoder.name],
                dataset=encoder.name,
                decoder=self.config["model"]["decoder"],
                encoder=self.config["model"]["encoders"][encoder.name],
                template=self.model,
            )
            trainer = self._fit(
                model=encoder_model,
                config=config,
                job_stage=f"encoder-{encoder.name}",
            )
            self._save_checkpoint(trainer, f"encoder-{encoder.name}")
            encoder_models.append(encoder_model)

        return encoder_models

    def train_stage_finetune(
        self, *, config: DictConfig, processor_model: ProcessorStage
    ) -> None:
        """Load pretrained weights from all stages into the full model and finetune end-to-end."""
        model = cast("EncodeProcessDecode", self.model)
        pretrained_encoders = {e.name: e for e in processor_model.encoders}
        for encoder in model.encoders:
            encoder.load_state_dict(pretrained_encoders[encoder.name].state_dict())
            log.info("Loaded pretrained weights for encoder '%s'.", encoder.name)
        model.processor.load_state_dict(processor_model.processor.state_dict())
        log.info("Loaded pretrained weights for processor.")
        model.decoder.load_state_dict(processor_model.decoder.state_dict())
        log.info("Loaded pretrained weights for decoder.")
        self._fit(config=config, job_stage="finetuning")

    def train_stage_processor(
        self,
        decoder_model: DecoderStage,
        *,
        config: DictConfig,
        checkpoint_dir: Path | None = None,
    ) -> ProcessorStage:
        """Train a processor on the latent space using frozen encoders and decoder."""
        if checkpoint_dir is not None and (
            matches := sorted(checkpoint_dir.glob("processor.epoch=*-step=*.ckpt"))
        ):
            checkpoint_path = matches[-1]
            log.info(
                "Skipping training for processor. Loaded checkpoint from %s.",
                checkpoint_path,
            )
            return ProcessorStage.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                processor=self.config["model"]["processor"],
                decoder_model=decoder_model,
            )

        processor_model = ProcessorStage.from_template(
            processor=self.config["model"]["processor"],
            decoder_model=decoder_model,
        )
        trainer = self._fit(model=processor_model, config=config, job_stage="processor")
        self._save_checkpoint(trainer, "processor")
        return processor_model

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id

from icenet_mp.callbacks import (
    IsolatedEvaluationCallback,
    PlottingCallback,
    UnconditionalCheckpoint,
)
from icenet_mp.compatibility.torch import (
    patch_interpolate_antialias,
    patch_open_file_limit,
)
from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.models import BaseModel, EncodeProcessDecode
from icenet_mp.models.multistage import DecoderStage, EncoderStage, ProcessorStage
from icenet_mp.types import SupportsMetadata
from icenet_mp.utils import get_device_name, get_timestamp, get_wandb_run

log = logging.getLogger(__name__)


def _checkpoint_constructor_overrides(checkpoint_path: Path) -> dict[str, Any]:
    """Recover architecture options omitted by legacy checkpoint hyperparameters."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        log.debug(
            "Could not inspect checkpoint hyperparameters at %s.", checkpoint_path
        )
        return {}

    hyperparameters = checkpoint.get("hyper_parameters", {})
    processor = hyperparameters.get("processor")
    state_dict = checkpoint.get("state_dict", {})
    if (
        isinstance(processor, (dict, DictConfig))
        and processor.get("_target_") == "icenet_mp.models.processors.VitProcessor"
        and "decode_head" not in processor
        and "processor.patch_to_pixels.weight" in state_dict
    ):
        processor = DictConfig(OmegaConf.create(processor))
        processor["decode_head"] = "conv_refine"
        patch_size = int(processor["patch_size"])
        projected_channels = int(
            state_dict["processor.patch_to_pixels.weight"].shape[0]
        )
        processor["refine_channels"] = projected_channels // (patch_size**2)
        processor["refine_kernel_size"] = int(
            state_dict["processor.refine.0.block.0.block.0.weight"].shape[-1]
        )
        log.info(
            "Detected legacy ViT conv-refine checkpoint; restoring omitted decode-head settings."
        )
        return {"processor": processor}
    return {}


class ModelService:
    def __init__(self, config: DictConfig) -> None:
        """Initialize the model service."""
        self.config_ = config

        # If a random seed was specified in the configuration, set it for reproducibility.
        if (seed := config.get("random", {}).get("seed", None)) is not None:
            seed = int(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            seed_everything(seed, workers=True)

        # Determine whether to enable fully deterministic mode
        self.fully_deterministic = config.get("random", {}).get(
            "fully_deterministic", False
        )

        # Apply any necessary compatibility patches
        accelerator = (
            config.get("train", {}).get("trainer", {}).get("accelerator", "auto")
        )
        if self.fully_deterministic or (
            torch.backends.mps.is_available() and accelerator in ("mps", "auto")
        ):
            patch_interpolate_antialias()
            log.warning(
                "Anti-aliasing disabled for compatibility with deterministic running "
                "and/or accelerator architecture."
            )
        patch_open_file_limit()

        self.data_module_: CommonDataModule | None = None
        self.model_: BaseModel | None = None
        self.checkpoint_path_: Path | None = None

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
            loss=config["loss"],
            mask_dir=str(builder.data_module.mask_directory),
            n_forecast_steps=builder.data_module.n_forecast_steps,
            n_history_steps=builder.data_module.n_history_steps,
            optimizer=config["train"]["optimizer"],
            output_space=builder.data_module.output_space.to_dict(),
            scheduler=config["train"]["scheduler"],
            target_variable_indices=builder.data_module.target_variable_indices,
            _convert_="object",
            _recursive_=False,
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
            current_callbacks = config.get("evaluate", {}).get("callbacks", {})
            isolated_evaluation = any(
                callback.get("_target_")
                == "icenet_mp.callbacks.IsolatedEvaluationCallback"
                for callback in current_callbacks.values()
            )
            if isolated_evaluation:
                combined_cfg["evaluate"]["callbacks"] = OmegaConf.create(
                    OmegaConf.to_container(current_callbacks)
                )
                combined_cfg["loggers"] = OmegaConf.create(
                    OmegaConf.to_container(config.get("loggers", {}))
                )
        except (NotADirectoryError, FileNotFoundError):
            combined_cfg = config
            log.debug("Could not load checkpoint configuration from %s.", config_path)

        # Load the model from checkpoint
        builder = cls(combined_cfg)
        builder.checkpoint_path_ = checkpoint_path.resolve()
        model_cls: type[BaseModel] = hydra.utils.get_class(
            builder.config["model"]["_target_"]
        )
        log.info("Loading a trained %s model...", builder.config["model"]["name"])
        constructor_overrides = _checkpoint_constructor_overrides(checkpoint_path)
        builder.model_ = model_cls.load_from_checkpoint(
            checkpoint_path,
            mask_dir=str(builder.data_module.mask_directory),
            latitudes_fn=lambda: builder.data_module.latitudes,
            longitudes_fn=lambda: builder.data_module.longitudes,
            map_location="cpu",  # portability: will be moved to the correct device later
            weights_only=False,
            **constructor_overrides,
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
        model: BaseModel | None = None,
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
        log.info("Configuring fitting for %s.", job_stage or "training")
        current_model = model or self.model
        current_model.optimizer_cfg = config["optimizer"]
        current_model.scheduler_cfg = config["scheduler"]
        if "loss" in config:
            current_model.loss_cfg = config["loss"]
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

        # Explicitly release cached device memory rather than delegating this to the
        # Python garbage collector. Multistage training runs many stages in one
        # long-lived process, so unreleased memory from earlier stages slows or blocks
        # later stages.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        gc.collect()

        return trainer

    def _merged_config(self, stage_name: str) -> DictConfig:
        """Return train config merged with stage-specific overrides."""
        return cast(
            "DictConfig",
            OmegaConf.merge(
                self.config["train"],
                self.config["train"].get("multistage", {}).get(stage_name, {}),
            ),
        )

    def _save_stage_checkpoint(self, trainer: Trainer, stage_name: str) -> Path:
        """Save a stage checkpoint at a predictable path.

        Args:
            trainer: The trainer that was used to train the model.
            stage_name: Name of the training stage (e.g. "encoder-era5").

        Returns:
            The path to the saved checkpoint.

        If a best checkpoint is available, it will be moved to the desired path.

        """
        ckpt_dir = self.build_run_directory(trainer) / "checkpoints"
        ckpt_name = f"{stage_name}.epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        # Check for existing best checkpoints
        best_model_paths = set(
            filter(
                None,
                (
                    str(getattr(callback, "best_model_path", ""))
                    for callback in trainer.checkpoint_callbacks
                ),
            )
        )
        if not best_model_paths:
            # Save a new checkpoint at the desired path
            trainer.save_checkpoint(ckpt_dir / ckpt_name, weights_only=False)
        elif len(best_model_paths) == 1:
            # Move a checkpoint that already exists to the desired path
            best_model_path = Path(best_model_paths.pop())
            ckpt_name = f"{stage_name}.{best_model_path.name}"
            if trainer.is_global_zero:
                shutil.move(best_model_path, ckpt_dir / ckpt_name)
            # Ensure all ranks see the moved file before proceeding
            trainer.strategy.barrier()
        else:
            msg = f"Cannot determine which of {len(best_model_paths)} checkpoints to save."
            raise ValueError(msg)
        if trainer.is_global_zero:
            log.info("Saved %s checkpoint to %s.", stage_name, ckpt_dir / ckpt_name)
        return ckpt_dir / ckpt_name

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

    def build_trainer(  # noqa: C901, PLR0912, PLR0915
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
                in log messages. Also sets the W&B ``job_type`` to ``"multistage"``
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

        # Setup Lightning loggers — only pass job_type/project to W&B loggers.
        extra_loggers = []
        for logger_config in self.config.get("loggers", {}).values():
            is_wandb = logger_config.get("_target_", "").split(".")[-1] == "WandbLogger"
            if is_wandb:
                extra_loggers.append(
                    hydra.utils.instantiate(
                        logger_config,
                        job_type="multistage" if job_stage else "single-stage",
                        project=project,
                    )
                )
            else:
                extra_loggers.append(hydra.utils.instantiate(logger_config))
        if not extra_loggers:
            log.warning("No loggers have been set for the trainer.")

        # Create a new trainer
        log.debug("Instantiating lightning trainer.")
        trainer = cast(
            "Trainer",
            hydra.utils.instantiate(
                config["trainer"],
                callbacks=extra_callbacks,
                deterministic="warn" if self.fully_deterministic else False,
                logger=extra_loggers,
            ),
        )

        # Check that fully_deterministic is set correctly
        if self.fully_deterministic != torch.are_deterministic_algorithms_enabled():
            actual = (
                "enabled"
                if torch.are_deterministic_algorithms_enabled()
                else "disabled"
            )
            desired = "enabled" if self.fully_deterministic else "disabled"
            msg = (
                f"torch deterministic algorithms are {actual}, but the config file "
                f"specifies that they should be {desired}."
            )
            raise ValueError(msg)
        if (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
        ):
            msg = (
                "When running in fully deterministic mode, 'warn_only' must be set to "
                "avoid segmentation faults from unsupported operations."
            )
            raise ValueError(msg)

        # Assign workers for data loading
        self.data_module.assign_workers(
            min(8, suggested_max_num_workers(trainer.num_devices))
        )

        # Ensure the run directory exists
        configured_run_directory = config.get("run_directory")
        run_directory = (
            Path(str(configured_run_directory)).expanduser().resolve()
            if configured_run_directory
            else self.build_run_directory(trainer)
        )
        if configured_run_directory and run_directory.exists():
            msg = (
                f"Configured run directory already exists: {run_directory}. "
                "Select a new evaluation run name."
            )
            raise FileExistsError(msg)
        log.debug("Set run directory to %s.", run_directory)
        run_directory.mkdir(parents=True, exist_ok=True)

        # Save model config to the run directory
        model_config_path = run_directory / "files" / "model_config.yaml"
        if trainer.is_global_zero:
            model_config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.config, model_config_path)
            if wandb_run := get_wandb_run(trainer):
                wandb_run.save(
                    model_config_path, base_path=model_config_path.parent, policy="now"
                )

        # Additional configuration for callbacks
        for callback in cast("list[Callback]", trainer.callbacks):  # type: ignore[attr-defined]
            log.debug("Configuring callback %s.", callback.__class__.__name__)
            # Set metadata for supported callbacks
            if isinstance(callback, SupportsMetadata):
                log.debug("Setting metadata for %s.", callback.__class__.__name__)
                model_name = self.config["model"].get(
                    "name", self.model.__class__.__name__
                )
                callback.set_metadata(self.config, model_name)
            if isinstance(callback, IsolatedEvaluationCallback):
                if self.checkpoint_path_ is None:
                    msg = "Isolated evaluation is only supported for checkpoint evaluation."
                    raise ValueError(msg)
                callback.set_evaluation_context(
                    checkpoint_path=self.checkpoint_path_,
                    config=self.config,
                    target_group_name=self.data_module.target_group_name,
                    target_variable_indices=self.data_module.target_variable_indices,
                )
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
        """Train an EncodeProcessDecode model in multiple stages.

        1. encoders
        2. decoder
        3. processor
        4. finetune

        Args:
            checkpoint_dir: Optional directory to load checkpoints from. If provided,
                training will skip any stages for which a checkpoint exists in this
                directory. Checkpoints are expected to be named in the format
                ``<stage>.epoch=<epoch>-step=<step>.ckpt``.

        Raises:
            TypeError: If the model is not an instance of ``EncodeProcessDecode``.

        """
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
        target_encoder = trained_encoders.pop()  # the decoder must not use this

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
            target_encoder=target_encoder,
        )

        log.info("Preparing to finetune...")
        self.train_stage_finetune(
            processor_model=processor_model,
            config=self._merged_config("finetune"),
        )

    def train_stage_decoder(
        self,
        encoder_models: list[EncoderStage],
        *,
        config: DictConfig,
        checkpoint_dir: Path | None = None,
    ) -> DecoderStage:
        """Train a decoder on the combined latent space of all frozen encoders."""
        if not isinstance(self.model, EncodeProcessDecode):
            msg = (
                "train_stage_decoder is only supported for EncodeProcessDecode models."
            )
            raise TypeError(msg)
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
                map_location="cpu",  # portability: will be moved to the correct device later
                weights_only=False,
                decoder=self.config["model"]["decoder"],
                encoders=encoder_models,
                target_dataset_name=self.data_module.target_group_name,
                target_variable_indices=self.data_module.target_variable_indices,
                mask_dir=str(self.data_module.mask_directory),
            )

        decoder_model = DecoderStage.from_template(
            decoder=self.config["model"]["decoder"],
            encoders=encoder_models,
            target_dataset_name=self.data_module.target_group_name,
            target_variable_indices=self.data_module.target_variable_indices,
            mask_dir=str(self.data_module.mask_directory),
        )
        log.info(
            "Training decoder: latent %s -> output %s",
            decoder_model.decoder.data_space_in.chw,
            decoder_model.decoder.data_space_out.chw,
        )
        trainer = self._fit(model=decoder_model, config=config, job_stage="decoder")
        ckpt_path = self._save_stage_checkpoint(trainer, "decoder")
        # Reload the best weights into the decoder model
        decoder_model.load_state_dict(
            torch.load(ckpt_path, weights_only=False)["state_dict"]
        )
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
        for encoder in [*self.model.encoders, self.model.target_encoder]:
            # The target encoder is named "target" but needs to load data from the real
            # corresponding underlying dataset. However, we need to construct a custom
            # DataSpace since we only want to consider the selected target variables,
            # not the full dataset.
            if encoder is self.model.target_encoder:
                dataset_name = self.data_module.target_group_name
                channel_names = self.data_module.target_variables
            else:
                dataset_name = encoder.name
                channel_names = self.data_module.variable_names[dataset_name]

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
                        map_location="cpu",  # portability: will be moved to the correct device later
                        weights_only=False,
                        latitudes_fn=lambda: self.data_module.latitudes,
                        longitudes_fn=lambda: self.data_module.longitudes,
                    )
                )
                continue

            encoder_model = EncoderStage.from_template(
                channel_names=channel_names,
                data_space_in=encoder.data_space_in,
                dataset=encoder.name,
                decoder=self.config["model"]["decoder"],
                encoder=self.config["model"]["encoders"][dataset_name],
                template=self.model,
            )
            log.info(
                "Training encoder-%s: input %s -> latent %s",
                encoder.name,
                encoder_model.encoder.data_space_in.chw,
                encoder_model.encoder.data_space_out.chw,
            )
            trainer = self._fit(
                model=encoder_model,
                config=config,
                job_stage=f"encoder-{encoder.name}",
            )
            ckpt_path = self._save_stage_checkpoint(trainer, f"encoder-{encoder.name}")
            # Reload the best weights into the encoder model
            encoder_model.load_state_dict(
                torch.load(ckpt_path, weights_only=False)["state_dict"]
            )
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
        trainer = self._fit(config=config, job_stage="finetune")
        self._save_stage_checkpoint(trainer, "finetune")

    def train_stage_processor(
        self,
        decoder_model: DecoderStage,
        target_encoder: EncoderStage,
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
                map_location="cpu",  # portability: will be moved to the correct device later
                weights_only=False,
                processor=self.config["model"]["processor"],
                decoder_model=decoder_model,
                target_encoder=target_encoder,
            )

        processor_model = ProcessorStage.from_template(
            processor=self.config["model"]["processor"],
            decoder_model=decoder_model,
            target_encoder=target_encoder,
        )
        log.info(
            "Training processor: history (%d, %d, %d, %d) -> forecast (%d, %d, %d, %d)",
            processor_model.processor.n_history_steps,
            *processor_model.processor.data_space.chw,
            processor_model.processor.n_forecast_steps,
            *processor_model.processor.data_space.chw,
        )
        trainer = self._fit(model=processor_model, config=config, job_stage="processor")
        ckpt_path = self._save_stage_checkpoint(trainer, "processor")
        # Reload the best weights into the processor model
        processor_model.load_state_dict(
            torch.load(ckpt_path, weights_only=False)["state_dict"]
        )
        return processor_model

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from icenet_mp.data_loaders import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import Metadata, ModelStepOutput, PlotSpec
from icenet_mp.utils import datetime_from_npdatetime
from icenet_mp.visualisations import DEFAULT_SIC_SPEC, Plotter

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        *,
        frequency: dict[str, int] | None = None,
        make_input_plots: bool = False,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        plot_spec: PlotSpec | None = None,
        base_path: str | None = None,
    ) -> None:
        """Create plots during evaluation or training validation.

        Note that we do not plot during training as the data is shuffled so it would be
        difficult to work out which date corresponds to each batch.

        Args:
            frequency: A dictionary specifying how often to make plots, with keys "batch" and/or "epoch".
            make_input_plots: Whether to plot the raw inputs.
            make_static_plots: Whether to create static plots.
            make_video_plots: Whether to create video plots.
            plot_spec: Plotting specification to use (contains difference settings, timestep selection, etc.).
            base_path: Base path for finding land masks.

        """
        super().__init__()
        self.frequency_batch = int((frequency or {}).get("batch", -1))
        self.frequency_epoch = int((frequency or {}).get("epoch", -1))
        self.make_input_plots = make_input_plots
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots

        # Plotter instance
        self.plotter = Plotter(base_path, DEFAULT_SIC_SPEC + plot_spec)
        self.plotter_metadata: Metadata | None = None

        # Cache the most recent batch
        self.cached_batch_idx_: int | None = None
        self.cached_dataloader_idx_: int | None = None
        self.cached_outputs_: ModelStepOutput | None = None

    def cache_batch(
        self,
        batch_idx: int,
        dataloader_idx: int,
        outputs: Tensor | Mapping[str, Any] | None,
    ) -> None:
        """Cache the current batch information for use in epoch-end plotting."""
        if isinstance(outputs, Mapping):
            self.cached_outputs_ = ModelStepOutput(**outputs)
            self.cached_batch_idx_ = batch_idx
            self.cached_dataloader_idx_ = dataloader_idx

    def load_dataset(
        self, dataloader: DataLoader | list[DataLoader] | None
    ) -> tuple[CombinedDataset, int] | None:
        """Load the dataset for the given dataloader index."""
        if dataloader is None or self.cached_dataloader_idx_ is None:
            return None
        dataloader = (
            dataloader[self.cached_dataloader_idx_]
            if isinstance(dataloader, Sequence)
            else dataloader
        )
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        if not isinstance(dataset, CombinedDataset):
            logger.warning("Dataset is of type %s not CombinedDataset", type(dataset))
            return None
        if batch_size is None:
            logger.warning("Dataloader does not have a batch size.")
            return None
        return (dataset, batch_size)

    def make_plots(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        dataset: CombinedDataset,
        batch_size: int,
    ) -> None:
        # Set plotting metadata
        if self.plotter_metadata:
            self.plotter_metadata.current_epoch = trainer.current_epoch
            self.plotter.set_metadata(self.plotter_metadata)

        # Ensure that outputs is a ModelStepOutput
        if self.cached_outputs_ is None or self.cached_batch_idx_ is None:
            logger.warning("Could not load outputs, skipping plotting.")
            return

        # Load dates from the dataset
        start_date = dataset.dates[batch_size * self.cached_batch_idx_]
        dates = list(
            map(datetime_from_npdatetime, dataset.get_forecast_steps(start_date))
        )

        # Set hemisphere for plotting based on dataset
        if not isinstance(pl_module, BaseModel):
            msg = f"Lightning module is of type {type(pl_module)}, skipping plotting."
            logger.warning(msg)
            return
        self.plotter.set_hemisphere(pl_module.hemisphere)

        # Get loggers that support image and video logging
        image_loggers = [ll for ll in trainer.loggers if hasattr(ll, "log_image")]
        video_loggers = [ll for ll in trainer.loggers if hasattr(ll, "log_video")]

        if self.make_static_plots:
            self.plotter.log_static_outputs(self.cached_outputs_, dates, image_loggers)
            if self.make_input_plots:
                self.plotter.log_static_inputs(dataset.inputs, dates, image_loggers)

        if self.make_video_plots:
            self.plotter.log_video_outputs(self.cached_outputs_, dates, video_loggers)
            if self.make_input_plots:
                self.plotter.log_video_inputs(dataset.inputs, dates, video_loggers)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each test batch."""
        # Check whether this is a batch we want to plot based on the frequency settings
        is_per_epoch = trainer.is_last_batch
        is_per_batch = self.frequency_batch > 0 and not batch_idx % self.frequency_batch

        # Cache if this is a batch we want to plot
        if is_per_epoch or is_per_batch:
            self.cache_batch(batch_idx, dataloader_idx, outputs)

        # If this is a selected batch then we will plot here
        if is_per_batch:
            # Load the dataset
            if not (ds_tuple := self.load_dataset(trainer.test_dataloaders)):
                logger.warning("Could not load dataset, skipping plotting.")
                return

            # Make the plots
            self.make_plots(trainer, pl_module, *ds_tuple)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of each test epoch."""
        # Only run plotting if this batch is at the specified frequency
        if self.frequency_epoch < 0 or trainer.current_epoch % self.frequency_epoch:
            return

        # Load the dataset
        if not (ds_tuple := self.load_dataset(trainer.test_dataloaders)):
            logger.warning("Could not load dataset, skipping plotting.")
            return

        # Make the plots
        self.make_plots(trainer, pl_module, *ds_tuple)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each validation batch."""
        # Ignore the initial sanity checking run
        if trainer.sanity_checking:
            return

        # Check whether this is a batch we want to plot based on the frequency settings
        is_per_epoch = trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch
        is_per_batch = self.frequency_batch > 0 and not batch_idx % self.frequency_batch

        # Cache if this is a batch we want to plot
        if is_per_epoch or is_per_batch:
            self.cache_batch(batch_idx, dataloader_idx, outputs)

        # If this is a selected batch then we will plot here
        if is_per_batch:
            # Load the dataset
            if not (ds_tuple := self.load_dataset(trainer.val_dataloaders)):
                logger.warning("Could not load dataset, skipping plotting.")
                return

            # Make the plots
            self.make_plots(trainer, pl_module, *ds_tuple)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called at the end of each validation epoch."""
        # Only run plotting if this batch is at the specified frequency
        if self.frequency_epoch < 0 or trainer.current_epoch % self.frequency_epoch:
            return

        # Load the dataset
        if not (ds_tuple := self.load_dataset(trainer.val_dataloaders)):
            logger.warning("Could not load dataset, skipping plotting.")
            return

        # Make the plots
        self.make_plots(trainer, pl_module, *ds_tuple)

    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        """Set metadata for the plotter."""
        self.plotter_metadata = self.plotter.get_metadata(config, model_name)

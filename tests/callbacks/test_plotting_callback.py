import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from icenet_mp.callbacks.plotting_callback import PlottingCallback
from icenet_mp.data import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import Metadata, ModelStepOutput


def _make_plots_args() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a minimal trainer/pl_module/dataset trio for make_plots tests."""
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 0
    trainer.loggers = []
    trainer.datamodule = None
    pl_module = MagicMock(spec=BaseModel)
    pl_module.hemisphere = "south"
    dataset = MagicMock(spec=CombinedDataset)
    dataset.dates = [np.datetime64("2020-01-01T12:00:00")]
    dataset.get_forecast_steps.return_value = [np.datetime64("2020-01-01T12:00:00")]
    dataset.inputs = []
    return trainer, pl_module, dataset


def _stub_plotter(
    callback: PlottingCallback, monkeypatch: pytest.MonkeyPatch
) -> dict[str, MagicMock]:
    """Replace load_target_uncertainties and all Plotter output methods with spies."""
    mocks = {
        "load_target_uncertainties": MagicMock(return_value={}),
        "log_static_outputs": MagicMock(),
        "log_static_inputs": MagicMock(),
        "log_video_outputs": MagicMock(),
        "log_video_inputs": MagicMock(),
        "set_hemisphere": MagicMock(),
        "set_metadata": MagicMock(),
    }
    monkeypatch.setattr(
        callback, "load_target_uncertainties", mocks["load_target_uncertainties"]
    )
    for name in (
        "log_static_outputs",
        "log_static_inputs",
        "log_video_outputs",
        "log_video_inputs",
        "set_hemisphere",
        "set_metadata",
    ):
        monkeypatch.setattr(callback.plotter, name, mocks[name])
    return mocks


class TestInit:
    """Tests for PlottingCallback construction."""

    def test_defaults_frequency_to_negative_one_when_not_given(self) -> None:
        """Disable all frequency-based triggers when no frequency dict is given."""
        callback = PlottingCallback()

        assert callback.frequency_batch == -1
        assert callback.frequency_epoch == -1
        assert callback.frequency_number == -1

    def test_parses_frequency_dict(self) -> None:
        """Read batch/epoch/number frequencies from the given dict."""
        callback = PlottingCallback(frequency={"batch": 5, "epoch": 2, "number": 3})

        assert callback.frequency_batch == 5
        assert callback.frequency_epoch == 2
        assert callback.frequency_number == 3

    def test_stores_plot_toggles_and_prefix(self) -> None:
        """Store the plot-type toggles and key prefix as configured."""
        callback = PlottingCallback(
            make_input_plots=True,
            make_static_plots=False,
            make_video_plots=False,
            prefix="eval",
        )

        assert callback.make_input_plots is True
        assert callback.make_static_plots is False
        assert callback.make_video_plots is False
        assert callback.prefix == "eval"


class TestCacheBatch:
    """Tests for cache_batch."""

    def test_caches_when_outputs_is_a_mapping(self) -> None:
        """Cache batch index, dataloader index, and outputs when given a mapping."""
        callback = PlottingCallback()
        outputs = {
            "prediction": torch.zeros(1, 1, 1, 2, 2),
            "target": torch.ones(1, 1, 1, 2, 2),
            "loss": torch.tensor(0.0),
        }

        callback.cache_batch(3, 1, outputs)

        assert callback.cached_batch_idx_ == 3
        assert callback.cached_dataloader_idx_ == 1
        assert isinstance(callback.cached_outputs_, ModelStepOutput)

    def test_does_not_cache_when_outputs_is_not_a_mapping(self) -> None:
        """Leave the cache untouched when outputs is not a mapping."""
        callback = PlottingCallback()

        callback.cache_batch(3, 1, torch.tensor(0.0))

        assert callback.cached_batch_idx_ is None
        assert callback.cached_dataloader_idx_ is None
        assert callback.cached_outputs_ is None


class TestIsSampleBatch:
    """Tests for is_sample_batch."""

    def test_returns_false_when_frequency_number_not_positive(self) -> None:
        """Never select a batch when sampling is disabled."""
        callback = PlottingCallback(frequency={"number": 0})

        assert callback.is_sample_batch(0, 10) is False

    @pytest.mark.parametrize(
        "total_batches", [float("inf"), float("nan")], ids=["inf", "nan"]
    )
    def test_returns_false_when_total_batches_not_finite(
        self, total_batches: float
    ) -> None:
        """Never select a batch when the total batch count is not finite."""
        callback = PlottingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(0, total_batches) is False

    def test_returns_false_when_total_batches_not_positive(self) -> None:
        """Never select a batch when there are no batches to sample from."""
        callback = PlottingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(0, 0) is False

    def test_single_target_selects_last_batch(self) -> None:
        """Sample only the final batch when frequency_number resolves to one target."""
        callback = PlottingCallback(frequency={"number": 1})

        assert callback.is_sample_batch(4, 5) is True
        assert callback.is_sample_batch(0, 5) is False

    @pytest.mark.parametrize(
        ("batch_idx", "expected"),
        [(0, True), (4, True), (9, True), (1, False), (5, False)],
        ids=["first", "middle", "last", "not-first", "not-middle"],
    )
    def test_selects_evenly_spaced_targets(
        self, batch_idx: int, *, expected: bool
    ) -> None:
        """Sample evenly-spaced batch indices across the epoch."""
        callback = PlottingCallback(frequency={"number": 3})

        assert callback.is_sample_batch(batch_idx, 10) is expected


class TestLoadDataset:
    """Tests for load_dataset."""

    def test_returns_none_when_dataloader_is_none(self) -> None:
        """Return None when there is no dataloader to inspect."""
        callback = PlottingCallback()
        callback.cached_dataloader_idx_ = 0

        assert callback.load_dataset(None) is None

    def test_returns_none_when_cached_dataloader_idx_is_none(self) -> None:
        """Return None when no batch has been cached yet."""
        callback = PlottingCallback()

        assert callback.load_dataset(MagicMock(spec=DataLoader)) is None

    def test_indexes_into_sequence_of_dataloaders(self) -> None:
        """Select the dataloader at cached_dataloader_idx_ from a sequence."""
        callback = PlottingCallback()
        callback.cached_dataloader_idx_ = 1
        dataset = MagicMock(spec=CombinedDataset)
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = dataset
        dataloader.batch_size = 4
        other_dataloader = MagicMock(spec=DataLoader)

        result = callback.load_dataset([other_dataloader, dataloader])

        assert result == (dataset, 4)

    def test_uses_single_dataloader_directly(self) -> None:
        """Use the dataloader directly when it is not a sequence."""
        callback = PlottingCallback()
        callback.cached_dataloader_idx_ = 0
        dataset = MagicMock(spec=CombinedDataset)
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = dataset
        dataloader.batch_size = 2

        assert callback.load_dataset(dataloader) == (dataset, 2)

    def test_returns_none_and_warns_when_dataset_is_not_combined_dataset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and return None when the dataloader's dataset is the wrong type."""
        callback = PlottingCallback()
        callback.cached_dataloader_idx_ = 0
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = object()
        dataloader.batch_size = 2

        with caplog.at_level(logging.WARNING):
            result = callback.load_dataset(dataloader)

        assert result is None
        assert "not CombinedDataset" in caplog.text

    def test_returns_none_and_warns_when_batch_size_is_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and return None when the dataloader has no batch size."""
        callback = PlottingCallback()
        callback.cached_dataloader_idx_ = 0
        dataloader = MagicMock(spec=DataLoader)
        dataloader.dataset = MagicMock(spec=CombinedDataset)
        dataloader.batch_size = None

        with caplog.at_level(logging.WARNING):
            result = callback.load_dataset(dataloader)

        assert result is None
        assert "does not have a batch size" in caplog.text


class TestSetMetadata:
    """Tests for set_metadata."""

    def test_delegates_to_plotter_get_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Store the plotter's metadata computed from the given config and model name."""
        callback = PlottingCallback()
        metadata = MagicMock(spec=Metadata)
        get_metadata = MagicMock(return_value=metadata)
        monkeypatch.setattr(callback.plotter, "get_metadata", get_metadata)
        config = DictConfig({"train": {}})

        callback.set_metadata(config, "my_model")

        get_metadata.assert_called_once_with(config, "my_model")
        assert callback.plotter_metadata is metadata


class TestMakePlots:
    """Tests for make_plots."""

    def test_warns_and_returns_when_no_cached_outputs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skip plotting entirely when no batch has been cached."""
        callback = PlottingCallback()
        trainer, pl_module, dataset = _make_plots_args()

        with caplog.at_level(logging.WARNING):
            callback.make_plots(trainer, pl_module, dataset, 1)

        assert "Could not load outputs" in caplog.text

    def test_warns_and_returns_when_pl_module_is_not_base_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skip plotting when the module is not a BaseModel (no hemisphere info)."""
        callback = PlottingCallback()
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, _pl_module, dataset = _make_plots_args()
        pl_module = MagicMock(spec=LightningModule)

        with caplog.at_level(logging.WARNING):
            callback.make_plots(trainer, pl_module, dataset, 1)

        assert "skipping plotting" in caplog.text

    def test_sets_plotter_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Push the cached plotter_metadata (with current_epoch) onto the plotter."""
        callback = PlottingCallback()
        stubs = _stub_plotter(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        callback.plotter_metadata = MagicMock(spec=Metadata)
        trainer, pl_module, dataset = _make_plots_args()
        trainer.current_epoch = 7

        callback.make_plots(trainer, pl_module, dataset, 1)

        assert callback.plotter_metadata.current_epoch == 7
        stubs["set_metadata"].assert_called_once_with(callback.plotter_metadata)
        stubs["set_hemisphere"].assert_called_once_with("south")

    def test_caches_land_mask_per_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build a LandMask once per mask directory and reuse it on repeat calls."""
        callback = PlottingCallback()
        _stub_plotter(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = _make_plots_args()
        trainer.datamodule = MagicMock(mask_directory=tmp_path)

        callback.make_plots(trainer, pl_module, dataset, 1)
        first_land_mask = callback.plotter.land_mask
        callback.make_plots(trainer, pl_module, dataset, 1)

        assert callback.plotter.land_mask is first_land_mask
        assert len(callback._land_mask_cache) == 1

    def test_skips_static_and_video_plots_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not call any plotter output methods when both toggles are disabled."""
        callback = PlottingCallback(make_static_plots=False, make_video_plots=False)
        stubs = _stub_plotter(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = _make_plots_args()

        callback.make_plots(trainer, pl_module, dataset, 1)

        stubs["log_static_outputs"].assert_not_called()
        stubs["log_video_outputs"].assert_not_called()

    def test_makes_input_plots_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Also log static and video input plots when make_input_plots is enabled."""
        callback = PlottingCallback(make_input_plots=True)
        stubs = _stub_plotter(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = _make_plots_args()

        callback.make_plots(trainer, pl_module, dataset, 1)

        stubs["log_static_inputs"].assert_called_once()
        stubs["log_video_inputs"].assert_called_once()

    def test_filters_loggers_by_image_and_video_support(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only pass loggers with log_image/log_video to the respective plot calls."""
        callback = PlottingCallback()
        stubs = _stub_plotter(callback, monkeypatch)
        callback.cached_batch_idx_ = 0
        callback.cached_outputs_ = MagicMock(spec=ModelStepOutput)
        trainer, pl_module, dataset = _make_plots_args()

        class ImageLogger:
            def log_image(self, *args: object, **kwargs: object) -> None: ...

        class VideoLogger:
            def log_video(self, *args: object, **kwargs: object) -> None: ...

        class PlainLogger: ...

        image_logger = ImageLogger()
        video_logger = VideoLogger()
        trainer.loggers = [image_logger, video_logger, PlainLogger()]

        callback.make_plots(trainer, pl_module, dataset, 1)

        assert stubs["log_static_outputs"].call_args[0][2] == [image_logger]
        assert stubs["log_video_outputs"].call_args[0][2] == [video_logger]


class TestOnTestBatchEnd:
    """Tests for on_test_batch_end."""

    def test_caches_and_plots_when_per_batch_frequency_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache and plot when the batch index matches the configured frequency."""
        callback = PlottingCallback(frequency={"batch": 2})
        cache_batch = MagicMock()
        dataset = MagicMock(spec=CombinedDataset)
        load_dataset = MagicMock(return_value=(dataset, 2))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.is_last_batch = False
        trainer.num_test_batches = [100]
        pl_module = MagicMock(spec=LightningModule)
        outputs = MagicMock()

        callback.on_test_batch_end(trainer, pl_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        load_dataset.assert_called_once_with(trainer.test_dataloaders)
        make_plots.assert_called_once_with(trainer, pl_module, dataset, 2)

    def test_caches_without_plotting_on_last_batch_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache the final batch of the epoch but defer plotting to epoch end."""
        callback = PlottingCallback()
        cache_batch = MagicMock()
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.is_last_batch = True
        trainer.num_test_batches = [100]
        pl_module = MagicMock(spec=LightningModule)
        outputs = MagicMock()

        callback.on_test_batch_end(trainer, pl_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        make_plots.assert_not_called()

    def test_does_nothing_when_batch_not_targeted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip caching entirely when the batch matches no trigger."""
        callback = PlottingCallback()
        cache_batch = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        trainer = MagicMock(spec=Trainer)
        trainer.is_last_batch = False
        trainer.num_test_batches = [100]
        pl_module = MagicMock(spec=LightningModule)

        callback.on_test_batch_end(trainer, pl_module, MagicMock(), {}, 4, 0)

        cache_batch.assert_not_called()

    def test_warns_when_dataset_cannot_be_loaded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and skip plotting when the dataset cannot be loaded."""
        callback = PlottingCallback(frequency={"batch": 1})
        monkeypatch.setattr(callback, "cache_batch", MagicMock())
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.is_last_batch = False
        trainer.num_test_batches = [100]
        pl_module = MagicMock(spec=LightningModule)

        with caplog.at_level(logging.WARNING):
            callback.on_test_batch_end(trainer, pl_module, MagicMock(), {}, 0, 0)

        assert "Could not load dataset" in caplog.text
        make_plots.assert_not_called()


class TestOnTestEpochEnd:
    """Tests for on_test_epoch_end."""

    def test_skips_when_frequency_epoch_negative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never load the dataset when epoch-based plotting is disabled."""
        callback = PlottingCallback()
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 5

        callback.on_test_epoch_end(trainer, MagicMock(spec=LightningModule))

        load_dataset.assert_not_called()

    def test_skips_when_epoch_not_at_frequency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip plotting on epochs that do not match the configured frequency."""
        callback = PlottingCallback(frequency={"epoch": 2})
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 3

        callback.on_test_epoch_end(trainer, MagicMock(spec=LightningModule))

        load_dataset.assert_not_called()

    def test_plots_when_epoch_at_frequency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot on epochs that match the configured frequency."""
        callback = PlottingCallback(frequency={"epoch": 2})
        dataset = MagicMock(spec=CombinedDataset)
        monkeypatch.setattr(
            callback, "load_dataset", MagicMock(return_value=(dataset, 2))
        )
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 4
        pl_module = MagicMock(spec=LightningModule)

        callback.on_test_epoch_end(trainer, pl_module)

        make_plots.assert_called_once_with(trainer, pl_module, dataset, 2)

    def test_warns_when_dataset_cannot_be_loaded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when the dataset cannot be loaded at epoch end."""
        callback = PlottingCallback(frequency={"epoch": 1})
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 0

        with caplog.at_level(logging.WARNING):
            callback.on_test_epoch_end(trainer, MagicMock(spec=LightningModule))

        assert "Could not load dataset" in caplog.text


class TestOnValidationBatchEnd:
    """Tests for on_validation_batch_end."""

    def test_skips_during_sanity_checking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ignore the initial sanity-checking run entirely."""
        callback = PlottingCallback(frequency={"batch": 1})
        cache_batch = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = True

        callback.on_validation_batch_end(
            trainer, MagicMock(spec=LightningModule), MagicMock(), {}, 0, 0
        )

        cache_batch.assert_not_called()

    def test_caches_and_plots_when_per_batch_frequency_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache and plot when the batch index matches the configured frequency."""
        callback = PlottingCallback(frequency={"batch": 2})
        cache_batch = MagicMock()
        dataset = MagicMock(spec=CombinedDataset)
        load_dataset = MagicMock(return_value=(dataset, 3))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "cache_batch", cache_batch)
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False
        trainer.fit_loop = MagicMock()
        trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch = False
        trainer.num_val_batches = [50]
        pl_module = MagicMock(spec=LightningModule)
        outputs = MagicMock()

        callback.on_validation_batch_end(trainer, pl_module, outputs, {}, 4, 0)

        cache_batch.assert_called_once_with(4, 0, outputs)
        load_dataset.assert_called_once_with(trainer.val_dataloaders)
        make_plots.assert_called_once_with(trainer, pl_module, dataset, 3)

    def test_warns_when_dataset_cannot_be_loaded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn and skip plotting when the dataset cannot be loaded."""
        callback = PlottingCallback(frequency={"batch": 1})
        monkeypatch.setattr(callback, "cache_batch", MagicMock())
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False
        trainer.fit_loop = MagicMock()
        trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch = False
        trainer.num_val_batches = [50]
        pl_module = MagicMock(spec=LightningModule)

        with caplog.at_level(logging.WARNING):
            callback.on_validation_batch_end(trainer, pl_module, MagicMock(), {}, 0, 0)

        assert "Could not load dataset" in caplog.text
        make_plots.assert_not_called()


class TestOnValidationEpochEnd:
    """Tests for on_validation_epoch_end."""

    def test_skips_when_epoch_not_at_frequency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip plotting on epochs that do not match the configured frequency."""
        callback = PlottingCallback(frequency={"epoch": 2})
        load_dataset = MagicMock()
        monkeypatch.setattr(callback, "load_dataset", load_dataset)
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 3

        callback.on_validation_epoch_end(trainer, MagicMock(spec=LightningModule))

        load_dataset.assert_not_called()

    def test_plots_when_epoch_at_frequency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot on epochs that match the configured frequency."""
        callback = PlottingCallback(frequency={"epoch": 2})
        dataset = MagicMock(spec=CombinedDataset)
        monkeypatch.setattr(
            callback, "load_dataset", MagicMock(return_value=(dataset, 2))
        )
        make_plots = MagicMock()
        monkeypatch.setattr(callback, "make_plots", make_plots)
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 4
        pl_module = MagicMock(spec=LightningModule)

        callback.on_validation_epoch_end(trainer, pl_module)

        make_plots.assert_called_once_with(trainer, pl_module, dataset, 2)

    def test_warns_when_dataset_cannot_be_loaded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when the dataset cannot be loaded at epoch end."""
        callback = PlottingCallback(frequency={"epoch": 1})
        monkeypatch.setattr(callback, "load_dataset", MagicMock(return_value=None))
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 0

        with caplog.at_level(logging.WARNING):
            callback.on_validation_epoch_end(trainer, MagicMock(spec=LightningModule))

        assert "Could not load dataset" in caplog.text

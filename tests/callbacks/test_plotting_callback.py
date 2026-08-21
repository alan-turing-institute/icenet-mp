from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import DictConfig

from icenet_mp.callbacks.plotting_callback import PlottingCallback
from icenet_mp.data import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import ModelStepOutput, PlotSpec, TensorNTCHW


class FakeTimeTraceModel(BaseModel):
    """A single-channel, 3-step BaseModel subclass used to drive PlottingCallback."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model with a single input and output channel, and 3 forecast steps."""
        super().__init__(
            hemisphere="north",
            input_spaces=[
                DictConfig({"channels": 1, "name": "input", "shape": (4, 4)})
            ],
            loss=DictConfig({"_target_": "torch.nn.HuberLoss", "delta": 0.5}),
            n_forecast_steps=3,
            n_history_steps=1,
            name="fake",
            optimizer=DictConfig({}),
            output_space=DictConfig({"channels": 1, "name": "target", "shape": (4, 4)}),
            scheduler=DictConfig({}),
            **kwargs,
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        raise NotImplementedError


class TestMakePlotsTimeTrace:
    """Tests for the time-trace plotting path of PlottingCallback.make_plots."""

    @pytest.mark.parametrize(
        ("make_time_trace_plots", "prefix", "expected_key"),
        [
            (True, None, "output_time_trace/sea-ice-concentration-time-trace"),
            (True, "val", "val/output_time_trace/sea-ice-concentration-time-trace"),
            (False, None, None),
        ],
    )
    def test_logs_time_trace_image(
        self,
        *,
        make_time_trace_plots: bool,
        prefix: str | None,
        expected_key: str | None,
    ) -> None:
        """make_plots should log a time trace under the expected key, or skip it."""
        callback = PlottingCallback(
            make_static_plots=False,
            make_time_trace_plots=make_time_trace_plots,
            make_video_plots=False,
            plot_spec=PlotSpec(dpi=72),
            prefix=prefix,
        )
        callback.cache_batch(
            batch_idx=0,
            dataloader_idx=0,
            outputs=ModelStepOutput(
                prediction=torch.rand(1, 3, 1, 4, 4),
                target=torch.rand(1, 3, 1, 4, 4),
                loss=torch.tensor(0.0),
            ),
        )

        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 0
        trainer.loggers = [MagicMock()]
        trainer.datamodule = MagicMock()
        trainer.datamodule.mask_directory = None

        dataset = MagicMock(spec=CombinedDataset)
        dataset.dates = [np.datetime64("2020-01-01")]
        dataset.get_forecast_steps.return_value = [
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-04"),
        ]

        callback.make_plots(trainer, FakeTimeTraceModel(), dataset, batch_size=1)

        image_logger = trainer.loggers[0]
        if expected_key is None:
            image_logger.log_image.assert_not_called()
            return
        image_logger.log_image.assert_called_once()
        call_kwargs = image_logger.log_image.call_args.kwargs
        assert call_kwargs["key"] == expected_key
        assert len(call_kwargs["images"]) == 1

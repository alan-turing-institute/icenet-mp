from datetime import date
from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayTHW
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_timeseries import _spatial_mean, plot_time_trace


class TestSpatialMean:
    def test_ignores_nan_values(self) -> None:
        values = np.array(
            [
                [[1.0, np.nan], [0.0, 1.0]],
                [[0.0, 0.0], [np.nan, 1.0]],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(_spatial_mean(values), [2 / 3, 1 / 3])


class TestPlotTimeTrace:
    def test_returns_image_from_forecast_fixture(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
    ) -> None:
        ground_truth, prediction, dates = sic_pair_3d_stream

        result = plot_time_trace(
            ground_truth,
            prediction,
            dates=dates,
            land_mask=LandMask(None),
            variable_name="sea-ice-concentration",
            dpi=72,
        )

        images = result["sea-ice-concentration-time-trace"]
        assert len(images) == 1
        assert isinstance(images[0], ImageFile)
        assert images[0].width > 0
        assert images[0].height > 0

    def test_rejects_date_count_mismatch(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
    ) -> None:
        ground_truth, prediction, dates = sic_pair_3d_stream

        with pytest.raises(InvalidArrayError, match="dates"):
            plot_time_trace(
                ground_truth,
                prediction,
                dates=dates[:-1],
                land_mask=LandMask(None),
                variable_name="sea-ice-concentration",
            )

    def test_land_mask_excludes_masked_pixels_from_mean(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Masked-out pixels must not skew the plotted spatial mean."""
        ground_truth, prediction, dates = sic_pair_3d_stream
        height, width = ground_truth.shape[1:]

        # Mark a block as land and give it an extreme value; if the land mask were
        # not applied before averaging, this would dominate the spatial mean.
        # LandMask.apply_to keeps True (ocean) pixels and NaNs out False (land) ones.
        land_region = np.s_[:5, :5]
        ground_truth = ground_truth.copy()
        prediction = prediction.copy()
        ground_truth[:, *land_region] = 1000.0
        prediction[:, *land_region] = 1000.0

        land_mask = LandMask(None)
        mask = np.ones((height, width), dtype=bool)
        mask[land_region] = False
        land_mask.add_mask(mask)

        expected_ground_truth_mean = _spatial_mean(land_mask.apply_to(ground_truth))
        expected_prediction_mean = _spatial_mean(land_mask.apply_to(prediction))
        # Sanity check: masking must actually change the result, otherwise this test
        # would pass even if plot_time_trace silently ignored the land mask.
        naive_mean = ground_truth.mean(axis=(1, 2))
        assert not np.allclose(expected_ground_truth_mean, naive_mean)

        plotted_ydata: list[np.ndarray] = []
        original_plot = Axes.plot

        def spy_plot(self: Axes, *args: Any, **kwargs: Any) -> list[Line2D]:
            plotted_ydata.append(np.asarray(args[1]))
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "plot", spy_plot)

        plot_time_trace(
            ground_truth,
            prediction,
            dates=dates,
            land_mask=land_mask,
            variable_name="sea-ice-concentration",
            dpi=72,
        )

        assert len(plotted_ydata) == 2
        np.testing.assert_allclose(plotted_ydata[0], expected_ground_truth_mean)
        np.testing.assert_allclose(plotted_ydata[1], expected_prediction_mean)

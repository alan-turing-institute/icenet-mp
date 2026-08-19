from datetime import date

import numpy as np
import pytest
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

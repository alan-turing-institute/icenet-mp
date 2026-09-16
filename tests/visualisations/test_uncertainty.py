from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.visualisations import (
    DEFAULT_SIC_SPEC,
    compute_standardised_difference,
    plot_static_uncertainty,
)
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_static import UncertaintyArrays


class TestComputeStandardisedDifference:
    def test_basic(self) -> None:
        """Compute signed prediction error in units of uncertainty."""
        ground_truth = np.array([[0.5, 0.8], [0.4, 0.2]], dtype=np.float32)
        prediction = np.array([[0.4, 0.6], [0.3, 0.5]], dtype=np.float32)
        uncertainty = np.array([[0.1, 0.2], [0.05, 0.1]], dtype=np.float32)

        result = compute_standardised_difference(ground_truth, prediction, uncertainty)

        np.testing.assert_allclose(result, [[1.0, 1.0], [2.0, -3.0]])

    def test_masks_invalid_uncertainty(self) -> None:
        """Mask values where uncertainty is zero, negative or non-finite."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)
        uncertainty = np.array([[0.5, 0.0], [-1.0, np.nan]], dtype=np.float32)

        result = compute_standardised_difference(ground_truth, prediction, uncertainty)

        assert result[0, 0] == pytest.approx(2.0)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])

    def test_rejects_shape_mismatch(self) -> None:
        """Reject input arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            compute_standardised_difference(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

    def test_rejects_non_2d_arrays(self) -> None:
        """Reject 1D (or any non-2D) ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            compute_standardised_difference(
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )


class TestPlotStaticUncertainty:
    def test_returns_image(self, no_land_mask: LandMask) -> None:
        """Render an uncertainty plot as an image."""
        ground_truth = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        prediction = np.array([[0.1, 0.5], [0.4, 0.7]], dtype=np.float32)
        uncertainty = np.full((2, 2), 0.1, dtype=np.float32)
        when = date(2026, 8, 21)
        spec = replace(DEFAULT_SIC_SPEC, dpi=50)

        result = plot_static_uncertainty(
            UncertaintyArrays(ground_truth, prediction, uncertainty),
            date=when,
            land_mask=no_land_mask,
            plot_spec=spec,
            variable_name="ice_conc",
        )

        key = "2026-08-21-ice_conc-uncertainty-z"
        assert key in result
        assert len(result[key]) == 1
        assert isinstance(result[key][0], ImageFile)
        assert result[key][0].width > 0
        assert result[key][0].height > 0

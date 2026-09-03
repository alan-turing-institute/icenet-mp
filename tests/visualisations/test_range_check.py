import numpy as np
import pytest

from icenet_mp.visualisations.range_check import compute_range_check_report


class TestMagnitudeMismatch:
    """Tests for the magnitude/units mismatch checks in compute_range_check_report."""

    def test_all_nan_arrays_use_zero_fallback_and_produce_no_magnitude_warning(
        self,
    ) -> None:
        """All-NaN ground truth and prediction should fall back to 0.0 medians/percentiles.

        With both reference values falling back to 0.0, no meaningful magnitude
        comparison can be made, so no MAGNITUDE warning should be produced.
        """
        ground_truth = np.full((3, 3), np.nan, dtype=np.float32)
        prediction = np.full((3, 3), np.nan, dtype=np.float32)

        report = compute_range_check_report(
            ground_truth,
            prediction,
            include_shared_range_mismatch_check=False,
        )

        assert not any(w.startswith("MAGNITUDE:") for w in report.warnings), (
            f"Expected no magnitude warnings, got: {report.warnings}"
        )

    def test_zero_ground_truth_with_substantial_prediction_warns(self) -> None:
        """Ground truth median (and 90th percentile) of zero, but prediction is substantial."""
        ground_truth = np.zeros((4, 4), dtype=np.float32)
        prediction = np.full((4, 4), 0.5, dtype=np.float32)

        report = compute_range_check_report(
            ground_truth,
            prediction,
            include_shared_range_mismatch_check=False,
        )

        magnitude_warnings = [w for w in report.warnings if w.startswith("MAGNITUDE:")]
        assert len(magnitude_warnings) == 1, (
            f"Expected exactly one magnitude warning, got: {report.warnings}"
        )
        assert "Ground truth median is zero" in magnitude_warnings[0]

    def test_zero_ground_truth_and_zero_prediction_produces_no_warning(self) -> None:
        """Both ground truth and prediction all-zero should not trigger a magnitude warning."""
        ground_truth = np.zeros((4, 4), dtype=np.float32)
        prediction = np.zeros((4, 4), dtype=np.float32)

        report = compute_range_check_report(
            ground_truth,
            prediction,
            include_shared_range_mismatch_check=False,
        )

        assert not any(w.startswith("MAGNITUDE:") for w in report.warnings), (
            f"Expected no magnitude warnings, got: {report.warnings}"
        )


class TestDisplayClipping:
    """Tests for the shared-scale clipping checks in compute_range_check_report."""

    def test_all_nan_prediction_reports_no_finite_values(self) -> None:
        """An entirely non-finite prediction array should trigger the no-finite-values warning."""
        ground_truth = np.full((4, 4), 0.5, dtype=np.float32)
        prediction = np.full((4, 4), np.nan, dtype=np.float32)

        report = compute_range_check_report(
            ground_truth,
            prediction,
            vmin=0.0,
            vmax=1.0,
            include_shared_range_mismatch_check=True,
        )

        assert "COLOUR ISSUE: No finite prediction values found." in report.warnings

    @pytest.mark.parametrize(
        ("n_below", "expect_prefix"),
        [(6, "!!!"), (2, "Clipping likely")],
        ids=["severe-below-vmin", "warn-below-vmin"],
    )
    def test_below_vmin_thresholds(self, n_below: int, expect_prefix: str) -> None:
        """Fraction of values below vmin should select the severe vs warn message variant.

        20 elements total: 6/20 = 30% (>= severe_outside=0.20) triggers the severe
        message; 2/20 = 10% (>= outside_warn=0.05, < severe_outside=0.20) triggers
        the milder "Clipping likely" message.
        """
        ground_truth = np.full((4, 5), 0.5, dtype=np.float32)
        prediction = np.full((4, 5), 0.5, dtype=np.float32)
        prediction.ravel()[:n_below] = -0.5

        report = compute_range_check_report(
            ground_truth,
            prediction,
            vmin=0.0,
            vmax=1.0,
            outside_warn=0.05,
            severe_outside=0.20,
            include_shared_range_mismatch_check=True,
        )

        below_warnings = [w for w in report.warnings if "below colour limit" in w]
        assert len(below_warnings) == 1, (
            f"Expected exactly one below-vmin warning, got: {report.warnings}"
        )
        assert expect_prefix in below_warnings[0]


class TestSharedRangeInvalid:
    """Tests for the invalid shared display-range (vmax <= vmin) branch."""

    @pytest.mark.parametrize(
        ("vmin", "vmax"),
        [(1.0, 0.0), (0.5, 0.5)],
        ids=["vmax-below-vmin", "vmax-equals-vmin"],
    )
    def test_invalid_range_reports_single_warning(
        self, vmin: float, vmax: float
    ) -> None:
        """An invalid [vmin, vmax] interval should short-circuit the clipping check."""
        ground_truth = np.full((3, 3), 0.5, dtype=np.float32)
        prediction = np.full((3, 3), 0.5, dtype=np.float32)

        report = compute_range_check_report(
            ground_truth,
            prediction,
            vmin=vmin,
            vmax=vmax,
            include_shared_range_mismatch_check=True,
        )

        colour_warnings = [w for w in report.warnings if "COLOUR ISSUE" in w]
        assert colour_warnings == ["COLOUR ISSUE: Shared display range invalid."]


class TestPrefixDeduplication:
    """Tests for the post-processing step that strips repeated category prefixes."""

    def test_simultaneous_above_and_below_vmax_dedup_colour_prefix(self) -> None:
        """Both above-vmax and below-vmin clipping in one call should dedupe the prefix.

        Only the first COLOUR ISSUE message in the output should retain its
        "COLOUR ISSUE:" prefix; the second should have it stripped.
        """
        ground_truth = np.full((10, 10), 0.5, dtype=np.float32)
        prediction = np.full((10, 10), 0.5, dtype=np.float32)
        flat = prediction.ravel()
        flat[:10] = 2.0  # 10% above vmax
        flat[10:20] = -2.0  # 10% below vmin

        report = compute_range_check_report(
            ground_truth,
            prediction,
            vmin=0.0,
            vmax=1.0,
            outside_warn=0.05,
            severe_outside=0.20,
            include_shared_range_mismatch_check=True,
        )

        colour_warnings = [w for w in report.warnings if "colour limit" in w]
        assert len(colour_warnings) == 2, (
            f"Expected two colour-limit warnings, got: {report.warnings}"
        )
        assert colour_warnings[0].startswith("COLOUR ISSUE:")
        assert not colour_warnings[1].startswith("COLOUR ISSUE:")

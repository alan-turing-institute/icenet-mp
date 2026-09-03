"""Tests for icenet_mp/visualisations/plotting_core.py.

Covers the pure helper functions used to compute colourmaps, normalisations
and differences for sea-ice plots. Functions already well covered by
tests/visualisations/test_colourscale.py, test_uncertainty.py and
test_plotting_static.py (exact/wildcard variable style matching,
compute_standardised_difference, shared/separate display ranges via the
sic_pair_2d fixture, signed-mode difference colourmaps) are only lightly
touched here, if at all -- this file targets the remaining gaps.
"""

from dataclasses import replace
from typing import Any

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm, to_rgba

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import PlotSpec
from icenet_mp.visualisations.plotting_core import (
    colourmap_with_bad,
    compute_difference,
    compute_display_ranges,
    compute_standardised_difference,
    create_normalisation,
    make_diff_colourmap,
    prepare_difference_stream,
    safe_nanmax,
    safe_nanmin,
    style_for_variable,
)


class TestColourmapWithBad:
    def test_none_defaults_to_viridis(self) -> None:
        """A None cmap_name falls back to the viridis colourmap."""
        cmap = colourmap_with_bad(None)

        assert isinstance(cmap, Colormap)
        assert cmap.name == "viridis"

    def test_named_cmap_is_used(self) -> None:
        """A named colourmap is looked up and returned by that name."""
        cmap = colourmap_with_bad("magma")

        assert cmap.name == "magma"

    def test_bad_color_is_configured(self) -> None:
        """The bad (NaN) colour is set to the requested colour."""
        cmap = colourmap_with_bad("viridis", bad_color="#ff00ff")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#ff00ff"))

    def test_default_bad_color(self) -> None:
        """With no bad_color argument, the light grey default is used."""
        cmap = colourmap_with_bad("viridis")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#dcdcdc"))

    def test_copy_fallback_on_uncopyable_colourmap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fall back to a fresh lookup when the returned colourmap can't be copied.

        Some matplotlib versions can hand back a colourmap object whose
        .copy() raises AttributeError/TypeError; the function should recover
        by re-fetching a fresh (copyable) colourmap by name rather than
        propagating the error.
        """
        real_get_cmap = mpl.colormaps.get_cmap
        calls = {"n": 0}

        class _UncopyableCmap:
            name = "viridis"

            def copy(self) -> Colormap:
                msg = "this colourmap cannot be copied"
                raise AttributeError(msg)

        def fake_get_cmap(name: str) -> Colormap:
            calls["n"] += 1
            if calls["n"] == 1:
                return _UncopyableCmap()  # type: ignore[return-value]
            return real_get_cmap(name)

        monkeypatch.setattr(mpl.colormaps, "get_cmap", fake_get_cmap)

        cmap = colourmap_with_bad("viridis")

        assert isinstance(cmap, Colormap)
        assert calls["n"] == 2


class TestSafeNanmin:
    def test_normal_array(self) -> None:
        """Return the true minimum for a fully finite array."""
        result = safe_nanmin(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(1.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = safe_nanmin(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(2.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = safe_nanmin(np.array([np.nan, np.nan]), default=-9.0)

        assert result == pytest.approx(-9.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = safe_nanmin(np.array([]), default=7.0)

        assert result == pytest.approx(7.0)

    def test_all_infinite_returns_default(self) -> None:
        """An array of only +/-inf falls back to the default value."""
        result = safe_nanmin(np.array([np.inf, -np.inf]), default=3.0)

        assert result == pytest.approx(3.0)


class TestSafeNanmax:
    def test_normal_array(self) -> None:
        """Return the true maximum for a fully finite array."""
        result = safe_nanmax(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(3.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = safe_nanmax(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(5.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = safe_nanmax(np.array([np.nan, np.nan]), default=42.0)

        assert result == pytest.approx(42.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = safe_nanmax(np.array([]), default=8.0)

        assert result == pytest.approx(8.0)


class TestStyleForVariable:
    def test_none_styles_returns_empty_style(self) -> None:
        """A None styles mapping returns an empty VariableStyle."""
        style = style_for_variable("era5:2t", None)

        assert style.cmap is None
        assert style.units is None

    def test_empty_styles_returns_empty_style(self) -> None:
        """An empty styles mapping returns an empty VariableStyle."""
        style = style_for_variable("era5:2t", {})

        assert style.cmap is None

    def test_non_mapping_styles_returns_empty_style(self) -> None:
        """A styles value that is not a Mapping (e.g. a list) is ignored."""
        style = style_for_variable(
            "era5:2t",
            ["not", "a", "mapping"],  # type: ignore[arg-type]
        )

        assert style.cmap is None

    def test_double_underscore_normalises_to_colon(self) -> None:
        """'era5__2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = style_for_variable("era5__2t", styles)

        assert style.cmap == "RdBu_r"
        assert style.units == "K"

    def test_hyphen_normalises_to_colon(self) -> None:
        """'era5-2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = style_for_variable("era5-2t", styles)

        assert style.cmap == "RdBu_r"

    def test_repeated_colons_collapse(self) -> None:
        """A variable name normalising to repeated ':' collapses to a single ':'."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = style_for_variable("era5__-2t", styles)

        assert style.cmap == "RdBu_r"

    def test_default_fallback(self) -> None:
        """An unmatched variable name falls back to the '_default' style."""
        styles = {"_default": {"cmap": "grey"}}

        style = style_for_variable("totally:unmatched", styles)

        assert style.cmap == "grey"

    def test_no_match_no_default_returns_empty_style(self) -> None:
        """No exact/wildcard/_default match returns an empty VariableStyle."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = style_for_variable("osisaf:ice_conc", styles)

        assert style.cmap is None

    def test_bare_wildcard_key_is_skipped(self) -> None:
        """A wildcard key of just '*' (empty prefix) is skipped, not treated as catch-all."""
        styles = {"*": {"cmap": "ignored"}, "_default": {"cmap": "fallback"}}

        style = style_for_variable("anything:at_all", styles)

        assert style.cmap == "fallback"

    def test_wildcard_candidate_not_a_dict_is_skipped(self) -> None:
        """A matching wildcard key whose value isn't a Mapping is logged and skipped."""
        styles: dict[str, Any] = {"era5:*": "not-a-mapping"}

        style = style_for_variable("era5:2t", styles)

        assert style.cmap is None


class TestCreateNormalisation:
    def test_no_centre_infers_from_data(self) -> None:
        """With no vmin/vmax/centre, the normalisation range comes from the data."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm, vmin, vmax = create_normalisation(data)

        assert type(norm) is Normalize
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(2.0)

    def test_no_centre_explicit_vmin_vmax(self) -> None:
        """Explicit vmin/vmax override the inferred data range."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm, vmin, vmax = create_normalisation(data, vmin=0.0, vmax=1.0)

        assert type(norm) is Normalize
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

    def test_no_centre_all_nan_data_uses_zero_one_fallback(self) -> None:
        """An all-NaN array with no explicit bounds falls back to [0, 1]."""
        data = np.full((2, 2), np.nan)

        _, vmin, vmax = create_normalisation(data)

        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

    def test_centre_symmetric_span_from_data(self) -> None:
        """A centred normalisation is symmetric around the centre value."""
        data = np.array([[0.0, 10.0]])

        norm, vmin, vmax = create_normalisation(data, centre=5.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == pytest.approx(5.0)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(10.0)

    def test_centre_asymmetric_data_expands_symmetrically(self) -> None:
        """An asymmetric data range around the centre still yields a symmetric norm."""
        data = np.array([[-2.0, 10.0]])

        norm, vmin, vmax = create_normalisation(data, centre=0.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert vmin == pytest.approx(-10.0)
        assert vmax == pytest.approx(10.0)

    def test_centre_explicit_vmin_vmax_used_over_data(self) -> None:
        """Explicit vmin/vmax combine with centre instead of the data range."""
        data = np.array([[100.0, -100.0]])

        norm, vmin, vmax = create_normalisation(data, vmin=-1.0, vmax=1.0, centre=0.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(1.0)


class TestComputeStandardisedDifferenceValidation:
    """Cover the one compute_standardised_difference branch test_uncertainty.py misses.

    test_uncertainty.py already exercises compute_standardised_difference's happy
    path, NaN-masking and shape-mismatch rejection; it does not reach the ndim
    check, which this class covers.
    """

    def test_rejects_non_2d_arrays(self) -> None:
        """Reject 1D (or any non-2D) ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            compute_standardised_difference(
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )


class TestComputeDifference:
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 5.0]])

    def test_signed(self) -> None:
        """Signed difference is ground_truth - prediction."""
        result = compute_difference(self.ground_truth, self.prediction, "signed")

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = compute_difference(self.ground_truth, self.prediction, "absolute")

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = compute_difference(self.ground_truth, self.prediction, "smape")

        expected = np.array(
            [
                [0.5 / 0.75, 0.5 / 2.25],
                [1.0 / 2.5, 1.0 / 4.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_smape_avoids_division_by_zero(self) -> None:
        """A near-zero denominator is clipped rather than dividing by zero."""
        ground_truth = np.array([0.0])
        prediction = np.array([0.0])

        result = compute_difference(ground_truth, prediction, "smape")

        assert np.isfinite(result).all()

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised difference mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid difference mode"):
            compute_difference(
                self.ground_truth,
                self.prediction,
                "bogus",  # type: ignore[arg-type]
            )


class TestMakeDiffColourmap:
    def test_signed_scalar(self) -> None:
        """A scalar sample yields a symmetric TwoSlopeNorm around zero."""
        spec = make_diff_colourmap(2.5, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vcenter == pytest.approx(0.0)
        assert spec.norm.vmin == pytest.approx(-2.5)
        assert spec.norm.vmax == pytest.approx(2.5)
        assert spec.vmin is None
        assert spec.vmax is None
        assert spec.cmap == "RdBu_r"

    def test_signed_scalar_below_one_still_uses_unit_floor(self) -> None:
        """A small scalar sample still gets at least a +/-1 symmetric range."""
        spec = make_diff_colourmap(0.1, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-1.0)
        assert spec.norm.vmax == pytest.approx(1.0)

    def test_signed_array(self) -> None:
        """An array sample uses the largest absolute extreme for a symmetric range."""
        sample = np.array([-2.0, 3.0, 0.5])

        spec = make_diff_colourmap(sample, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-3.0)
        assert spec.norm.vmax == pytest.approx(3.0)
        assert spec.cmap == "RdBu_r"

    def test_absolute_scalar(self) -> None:
        """A scalar sample for absolute mode sets vmax directly."""
        spec = make_diff_colourmap(0.75, mode="absolute")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.75)
        assert spec.cmap == "magma"

    def test_absolute_array(self) -> None:
        """An array sample for absolute mode sets vmax from the array's max."""
        sample = np.array([0.1, 0.9, 0.4])

        spec = make_diff_colourmap(sample, mode="absolute")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.9)
        assert spec.cmap == "magma"

    def test_smape_scalar(self) -> None:
        """SMAPE mode behaves like absolute mode for a scalar sample."""
        spec = make_diff_colourmap(1.5, mode="smape")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(1.5)
        assert spec.cmap == "magma"

    def test_smape_array(self) -> None:
        """SMAPE mode behaves like absolute mode for an array sample."""
        sample = np.array([0.2, 0.6])

        spec = make_diff_colourmap(sample, mode="smape")

        assert spec.vmax == pytest.approx(0.6)
        assert spec.cmap == "magma"

    def test_vmax_floor_avoids_zero_width_range(self) -> None:
        """A zero (or negative) sample still yields a strictly positive vmax."""
        spec = make_diff_colourmap(0.0, mode="absolute")

        assert spec.vmax == pytest.approx(1e-6)

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode"):
            make_diff_colourmap(1.0, mode="bogus")  # type: ignore[arg-type]


class TestPrepareDifferenceStream:
    ground_truth_stream = np.array(
        [
            [[0.5, 0.2], [0.1, 0.9]],
            [[0.4, 0.3], [0.2, 0.6]],
            [[0.7, 0.1], [0.3, 0.5]],
        ]
    )
    prediction_stream = np.array(
        [
            [[0.4, 0.3], [0.2, 0.7]],
            [[0.5, 0.1], [0.3, 0.5]],
            [[0.6, 0.3], [0.1, 0.6]],
        ]
    )

    def test_include_difference_false_short_circuits(self) -> None:
        """include_difference=False returns (None, None) regardless of strategy."""
        difference_stream, colour_scale = prepare_difference_stream(
            include_difference=False,
            diff_mode="signed",
            strategy="precompute",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert difference_stream is None
        assert colour_scale is None

    def test_precompute_strategy_returns_full_difference_stream(self) -> None:
        """Precompute returns the full elementwise difference stream and a colour scale."""
        difference_stream, colour_scale = prepare_difference_stream(
            include_difference=True,
            diff_mode="signed",
            strategy="precompute",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert difference_stream is not None
        np.testing.assert_allclose(
            difference_stream, self.ground_truth_stream - self.prediction_stream
        )
        assert colour_scale is not None
        assert colour_scale.cmap == "RdBu_r"

    def test_two_pass_strategy_scans_for_colour_scale_only(self) -> None:
        """two-pass returns no difference stream but derives the colour scale."""
        difference_stream, colour_scale = prepare_difference_stream(
            include_difference=True,
            diff_mode="absolute",
            strategy="two-pass",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert difference_stream is None
        assert colour_scale is not None
        expected_max = float(
            np.nanmax(np.abs(self.ground_truth_stream - self.prediction_stream))
        )
        assert colour_scale.vmax == pytest.approx(expected_max)
        assert colour_scale.vmin == pytest.approx(0.0)

    def test_two_pass_strategy_signed_takes_absolute_of_extremes(self) -> None:
        """two-pass with signed mode uses the largest absolute per-frame extreme."""
        _, colour_scale = prepare_difference_stream(
            include_difference=True,
            diff_mode="signed",
            strategy="two-pass",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert colour_scale is not None
        assert isinstance(colour_scale.norm, TwoSlopeNorm)
        raw_max = float(
            np.nanmax(np.abs(self.ground_truth_stream - self.prediction_stream))
        )
        # make_diff_colourmap enforces a minimum symmetric span of +/-1.0.
        expected_max = max(1.0, raw_max)
        assert colour_scale.norm.vmax == pytest.approx(expected_max)
        assert colour_scale.norm.vmin == pytest.approx(-expected_max)

    def test_per_frame_strategy_returns_nothing_precomputed(self) -> None:
        """per-frame defers all computation to per-frame calls, returning (None, None)."""
        difference_stream, colour_scale = prepare_difference_stream(
            include_difference=True,
            diff_mode="signed",
            strategy="per-frame",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert difference_stream is None
        assert colour_scale is None

    def test_invalid_strategy_raises(self) -> None:
        """An unrecognised strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DiffStrategy"):
            prepare_difference_stream(
                include_difference=True,
                diff_mode="signed",
                strategy="bogus",  # type: ignore[arg-type]
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )


class TestComputeDisplayRanges:
    ground_truth = np.array([[0.1, 0.5], [0.9, 0.3]], dtype=np.float32)
    prediction = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

    def test_shared_strategy_uses_ground_truth_range_for_both(self) -> None:
        """'shared' uses the ground-truth range for both ground truth and prediction."""
        plot_spec = replace(PlotSpec(), colourbar_strategy="shared")

        gt_range, pred_range = compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.1), pytest.approx(0.9))
        assert pred_range == (pytest.approx(0.1), pytest.approx(0.9))

    def test_separate_strategy_uses_each_panels_own_range(self) -> None:
        """'separate' uses each panel's own data range."""
        plot_spec = replace(PlotSpec(), colourbar_strategy="separate")

        gt_range, pred_range = compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.1), pytest.approx(0.9))
        assert pred_range == (pytest.approx(0.2), pytest.approx(0.8))

    def test_unrecognised_strategy_falls_back_to_plot_spec_vmin_vmax(self) -> None:
        """An unrecognised colourbar_strategy falls back to plot_spec.vmin/vmax.

        PlotSpec.colourbar_strategy is typed as Literal["shared", "separate"],
        but dataclasses do not enforce field types at runtime, so this branch
        is reachable by mutating the field after construction.
        """
        plot_spec = replace(PlotSpec(), vmin=0.25, vmax=0.75)
        plot_spec.colourbar_strategy = "bogus"  # type: ignore[assignment]

        gt_range, pred_range = compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.25), pytest.approx(0.75))
        assert pred_range == (pytest.approx(0.25), pytest.approx(0.75))

    def test_unrecognised_strategy_defaults_when_vmin_vmax_none(self) -> None:
        """The fallback defaults to (0.0, 1.0) when plot_spec.vmin/vmax are None."""
        plot_spec = replace(PlotSpec(), vmin=None, vmax=None)
        plot_spec.colourbar_strategy = "bogus"  # type: ignore[assignment]

        gt_range, pred_range = compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.0), pytest.approx(1.0))
        assert pred_range == (pytest.approx(0.0), pytest.approx(1.0))

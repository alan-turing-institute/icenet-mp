"""Tests for icenet_mp/visualisations/colour_scale.py."""

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm, to_rgba

from icenet_mp.visualisations.colour_scale import ColourScale


class TestColourmapWithBad:
    def test_none_defaults_to_viridis(self) -> None:
        """Omitting cmap_name falls back to the viridis colourmap."""
        cmap = ColourScale("signed").colourmap()

        assert isinstance(cmap, Colormap)
        assert cmap.name == "viridis"

    def test_named_cmap_is_used(self) -> None:
        """A named colourmap is looked up and returned by that name."""
        cmap = ColourScale("signed").colourmap("magma")

        assert cmap.name == "magma"

    def test_bad_color_is_configured(self) -> None:
        """The bad (NaN) colour is set to the requested colour."""
        cmap = ColourScale("signed").colourmap("viridis", bad_color="#ff00ff")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#ff00ff"))

    def test_default_bad_color(self) -> None:
        """With no bad_color argument, the light grey default is used."""
        cmap = ColourScale("signed").colourmap("viridis")

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

        cmap = ColourScale("signed").colourmap("viridis")

        assert isinstance(cmap, Colormap)
        assert calls["n"] == 2


class TestCreateNormalisation:
    def test_no_centre_infers_from_data(self) -> None:
        """With no vmin/vmax/centre, the normalisation range comes from the data."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm = ColourScale("signed").normalisation(data)

        assert type(norm) is Normalize
        assert norm.vmin == pytest.approx(-1.0)
        assert norm.vmax == pytest.approx(2.0)

    def test_no_centre_explicit_vmin_vmax(self) -> None:
        """Explicit vmin/vmax override the inferred data range."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm = ColourScale("signed").normalisation(data, vmin=0.0, vmax=1.0)

        assert type(norm) is Normalize
        assert norm.vmin == pytest.approx(0.0)
        assert norm.vmax == pytest.approx(1.0)

    def test_no_centre_all_nan_data_uses_zero_one_fallback(self) -> None:
        """An all-NaN array with no explicit bounds falls back to [0, 1]."""
        data = np.full((2, 2), np.nan)

        norm = ColourScale("signed").normalisation(data)

        assert norm.vmin == pytest.approx(0.0)
        assert norm.vmax == pytest.approx(1.0)

    def test_centre_symmetric_span_from_data(self) -> None:
        """A centred normalisation is symmetric around the centre value."""
        data = np.array([[0.0, 10.0]])

        norm = ColourScale("signed").normalisation(data, centre=5.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == pytest.approx(5.0)
        assert norm.vmin == pytest.approx(0.0)
        assert norm.vmax == pytest.approx(10.0)

    def test_centre_asymmetric_data_expands_symmetrically(self) -> None:
        """An asymmetric data range around the centre still yields a symmetric norm."""
        data = np.array([[-2.0, 10.0]])

        norm = ColourScale("signed").normalisation(data, centre=0.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vmin == pytest.approx(-10.0)
        assert norm.vmax == pytest.approx(10.0)

    def test_centre_explicit_vmin_vmax_used_over_data(self) -> None:
        """Explicit vmin/vmax combine with centre instead of the data range."""
        data = np.array([[100.0, -100.0]])

        norm = ColourScale("signed").normalisation(
            data, vmin=-1.0, vmax=1.0, centre=0.0
        )

        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vmin == pytest.approx(-1.0)
        assert norm.vmax == pytest.approx(1.0)


class TestMakeDiffColourmap:
    def test_signed_scalar(self) -> None:
        """A scalar sample yields a symmetric TwoSlopeNorm around zero."""
        spec = ColourScale("signed").diff_colourmap(2.5)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vcenter == pytest.approx(0.0)
        assert spec.norm.vmin == pytest.approx(-2.5)
        assert spec.norm.vmax == pytest.approx(2.5)
        assert spec.vmin is None
        assert spec.vmax is None
        assert spec.cmap == "RdBu_r"

    def test_signed_scalar_below_one_still_uses_unit_floor(self) -> None:
        """A small scalar sample still gets at least a +/-1 symmetric range."""
        spec = ColourScale("signed").diff_colourmap(0.1)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-1.0)
        assert spec.norm.vmax == pytest.approx(1.0)

    def test_signed_array(self) -> None:
        """An array sample uses the largest absolute extreme for a symmetric range."""
        sample = np.array([-2.0, 3.0, 0.5])

        spec = ColourScale("signed").diff_colourmap(sample)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-3.0)
        assert spec.norm.vmax == pytest.approx(3.0)
        assert spec.cmap == "RdBu_r"

    def test_absolute_scalar(self) -> None:
        """A scalar sample for absolute mode sets vmax directly."""
        spec = ColourScale("absolute").diff_colourmap(0.75)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.75)
        assert spec.cmap == "magma"

    def test_absolute_array(self) -> None:
        """An array sample for absolute mode sets vmax from the array's max."""
        sample = np.array([0.1, 0.9, 0.4])

        spec = ColourScale("absolute").diff_colourmap(sample)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.9)
        assert spec.cmap == "magma"

    def test_smape_scalar(self) -> None:
        """SMAPE mode behaves like absolute mode for a scalar sample."""
        spec = ColourScale("smape").diff_colourmap(1.5)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(1.5)
        assert spec.cmap == "magma"

    def test_smape_array(self) -> None:
        """SMAPE mode behaves like absolute mode for an array sample."""
        sample = np.array([0.2, 0.6])

        spec = ColourScale("smape").diff_colourmap(sample)

        assert spec.vmax == pytest.approx(0.6)
        assert spec.cmap == "magma"

    def test_vmax_floor_avoids_zero_width_range(self) -> None:
        """A zero (or negative) sample still yields a strictly positive vmax."""
        spec = ColourScale("absolute").diff_colourmap(0.0)

        assert spec.vmax == pytest.approx(1e-6)

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode"):
            ColourScale("bogus").diff_colourmap(1.0)  # type: ignore[arg-type]


class TestBounds:
    def test_reads_bounds_from_norm_when_present(self) -> None:
        """A diverging (signed) colourmap's bounds come from its norm."""
        colour_scale = ColourScale("signed")
        spec = colour_scale.diff_colourmap(2.5)

        assert colour_scale.bounds(spec) == (pytest.approx(-2.5), pytest.approx(2.5))

    def test_reads_explicit_bounds_when_no_norm(self) -> None:
        """A sequential (absolute/smape) colourmap's bounds come from vmin/vmax directly."""
        colour_scale = ColourScale("absolute")
        spec = colour_scale.diff_colourmap(0.75)

        assert colour_scale.bounds(spec) == (pytest.approx(0.0), pytest.approx(0.75))

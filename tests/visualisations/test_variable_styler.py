"""Tests for icenet_mp/visualisations/variable_styler.py."""

from typing import Any

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm, to_rgba

from icenet_mp.visualisations.variable_styler import VariableStyler


class TestColourmapWithBad:
    def test_none_defaults_to_viridis(self) -> None:
        """A None cmap_name falls back to the viridis colourmap."""
        cmap = VariableStyler().colourmap_with_bad(None)

        assert isinstance(cmap, Colormap)
        assert cmap.name == "viridis"

    def test_named_cmap_is_used(self) -> None:
        """A named colourmap is looked up and returned by that name."""
        cmap = VariableStyler().colourmap_with_bad("magma")

        assert cmap.name == "magma"

    def test_bad_color_is_configured(self) -> None:
        """The bad (NaN) colour is set to the requested colour."""
        cmap = VariableStyler().colourmap_with_bad("viridis", bad_color="#ff00ff")

        np.testing.assert_allclose(cmap.get_bad(), to_rgba("#ff00ff"))

    def test_default_bad_color(self) -> None:
        """With no bad_color argument, the light grey default is used."""
        cmap = VariableStyler().colourmap_with_bad("viridis")

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

        cmap = VariableStyler().colourmap_with_bad("viridis")

        assert isinstance(cmap, Colormap)
        assert calls["n"] == 2


class TestSafeNanmin:
    def test_normal_array(self) -> None:
        """Return the true minimum for a fully finite array."""
        result = VariableStyler().safe_nanmin(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(1.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = VariableStyler().safe_nanmin(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(2.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = VariableStyler().safe_nanmin(np.array([np.nan, np.nan]), default=-9.0)

        assert result == pytest.approx(-9.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = VariableStyler().safe_nanmin(np.array([]), default=7.0)

        assert result == pytest.approx(7.0)

    def test_all_infinite_returns_default(self) -> None:
        """An array of only +/-inf falls back to the default value."""
        result = VariableStyler().safe_nanmin(np.array([np.inf, -np.inf]), default=3.0)

        assert result == pytest.approx(3.0)


class TestSafeNanmax:
    def test_normal_array(self) -> None:
        """Return the true maximum for a fully finite array."""
        result = VariableStyler().safe_nanmax(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(3.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = VariableStyler().safe_nanmax(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(5.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = VariableStyler().safe_nanmax(np.array([np.nan, np.nan]), default=42.0)

        assert result == pytest.approx(42.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = VariableStyler().safe_nanmax(np.array([]), default=8.0)

        assert result == pytest.approx(8.0)


class TestStyleForVariable:
    def test_none_styles_returns_empty_style(self) -> None:
        """A None styles mapping returns an empty VariableStyle."""
        style = VariableStyler().style_for_variable("era5:2t", None)

        assert style.cmap is None
        assert style.units is None

    def test_empty_styles_returns_empty_style(self) -> None:
        """An empty styles mapping returns an empty VariableStyle."""
        style = VariableStyler().style_for_variable("era5:2t", {})

        assert style.cmap is None
        assert style.decimals is None

    def test_non_mapping_styles_returns_empty_style(self) -> None:
        """A styles value that is not a Mapping (e.g. a list) is ignored."""
        style = VariableStyler().style_for_variable(
            "era5:2t",
            ["not", "a", "mapping"],  # type: ignore[arg-type]
        )

        assert style.cmap is None

    def test_double_underscore_normalises_to_colon(self) -> None:
        """'era5__2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = VariableStyler().style_for_variable("era5__2t", styles)

        assert style.cmap == "RdBu_r"
        assert style.units == "K"

    def test_hyphen_normalises_to_colon(self) -> None:
        """'era5-2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = VariableStyler().style_for_variable("era5-2t", styles)

        assert style.cmap == "RdBu_r"

    def test_repeated_colons_collapse(self) -> None:
        """A variable name normalising to repeated ':' collapses to a single ':'."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = VariableStyler().style_for_variable("era5__-2t", styles)

        assert style.cmap == "RdBu_r"

    def test_default_fallback(self) -> None:
        """An unmatched variable name falls back to the '_default' style."""
        styles = {"_default": {"cmap": "grey"}}

        style = VariableStyler().style_for_variable("totally:unmatched", styles)

        assert style.cmap == "grey"

    def test_no_match_no_default_returns_empty_style(self) -> None:
        """No exact/wildcard/_default match returns an empty VariableStyle."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = VariableStyler().style_for_variable("osisaf:ice_conc", styles)

        assert style.cmap is None

    def test_bare_wildcard_key_is_skipped(self) -> None:
        """A wildcard key of just '*' (empty prefix) is skipped, not treated as catch-all."""
        styles = {"*": {"cmap": "ignored"}, "_default": {"cmap": "fallback"}}

        style = VariableStyler().style_for_variable("anything:at_all", styles)

        assert style.cmap == "fallback"

    def test_wildcard_candidate_not_a_dict_is_skipped(self) -> None:
        """A matching wildcard key whose value isn't a Mapping is logged and skipped."""
        styles: dict[str, Any] = {"era5:*": "not-a-mapping"}

        style = VariableStyler().style_for_variable("era5:2t", styles)

        assert style.cmap is None

    def test_exact_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test exact variable name matching in styling."""
        style = VariableStyler().style_for_variable("era5:2t", variable_styles)

        assert style.cmap == "RdBu_r"
        assert style.two_slope_centre == 273.15
        assert style.units == "K"
        assert style.decimals == 1

    def test_wildcard_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test wildcard pattern matching in styling."""
        # Add wildcard pattern
        styles_with_wildcard = {
            **variable_styles,
            "era5:q_*": {"cmap": "viridis", "decimals": 4},
        }

        style = VariableStyler().style_for_variable("era5:q_500", styles_with_wildcard)

        assert style.cmap == "viridis"
        assert style.decimals == 4

    def test_scientific_notation(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test scientific notation option in styling."""
        # Add style with scientific notation
        styles_with_scientific = {
            **variable_styles,
            "era5:q_10": {
                "cmap": "viridis",
                "decimals": 2,
                "units": "kg/kg",
                "use_scientific_notation": True,
            },
        }

        style = VariableStyler().style_for_variable("era5:q_10", styles_with_scientific)

        assert style.cmap == "viridis"
        assert style.decimals == 2
        assert style.units == "kg/kg"
        assert style.use_scientific_notation is True


class TestCreateNormalisation:
    def test_no_centre_infers_from_data(self) -> None:
        """With no vmin/vmax/centre, the normalisation range comes from the data."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm, vmin, vmax = VariableStyler().create_normalisation(data)

        assert type(norm) is Normalize
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(2.0)

    def test_no_centre_explicit_vmin_vmax(self) -> None:
        """Explicit vmin/vmax override the inferred data range."""
        data = np.array([[-1.0, 0.5], [2.0, 0.1]])

        norm, vmin, vmax = VariableStyler().create_normalisation(
            data, vmin=0.0, vmax=1.0
        )

        assert type(norm) is Normalize
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

    def test_no_centre_all_nan_data_uses_zero_one_fallback(self) -> None:
        """An all-NaN array with no explicit bounds falls back to [0, 1]."""
        data = np.full((2, 2), np.nan)

        _, vmin, vmax = VariableStyler().create_normalisation(data)

        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

    def test_centre_symmetric_span_from_data(self) -> None:
        """A centred normalisation is symmetric around the centre value."""
        data = np.array([[0.0, 10.0]])

        norm, vmin, vmax = VariableStyler().create_normalisation(data, centre=5.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == pytest.approx(5.0)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(10.0)

    def test_centre_asymmetric_data_expands_symmetrically(self) -> None:
        """An asymmetric data range around the centre still yields a symmetric norm."""
        data = np.array([[-2.0, 10.0]])

        norm, vmin, vmax = VariableStyler().create_normalisation(data, centre=0.0)

        assert isinstance(norm, TwoSlopeNorm)
        assert vmin == pytest.approx(-10.0)
        assert vmax == pytest.approx(10.0)

    def test_centre_explicit_vmin_vmax_used_over_data(self) -> None:
        """Explicit vmin/vmax combine with centre instead of the data range."""
        data = np.array([[100.0, -100.0]])

        norm, vmin, vmax = VariableStyler().create_normalisation(
            data, vmin=-1.0, vmax=1.0, centre=0.0
        )

        assert isinstance(norm, TwoSlopeNorm)
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(1.0)

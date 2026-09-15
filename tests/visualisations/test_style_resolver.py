"""Tests for icenet_mp/visualisations/variable_style_resolver.py."""

from typing import Any

from icenet_mp.visualisations.style_resolver import StyleResolver

DEFAULT_CMAP = "viridis"


class TestStyleForVariable:
    def test_none_styles_falls_back_to_default_cmap(self) -> None:
        """A None styles mapping resolves cmap to the bound default."""
        style = StyleResolver(None, DEFAULT_CMAP).style_for_variable("era5:2t")

        assert style.cmap == DEFAULT_CMAP
        assert style.units is None

    def test_empty_styles_falls_back_to_default_cmap(self) -> None:
        """An empty styles mapping resolves cmap to the bound default."""
        style = StyleResolver({}, DEFAULT_CMAP).style_for_variable("era5:2t")

        assert style.cmap == DEFAULT_CMAP
        assert style.vmin is None

    def test_non_mapping_styles_falls_back_to_default_cmap(self) -> None:
        """A styles value that is not a Mapping (e.g. a list) is ignored."""
        style = StyleResolver(
            ["not", "a", "mapping"],  # type: ignore[arg-type]
            DEFAULT_CMAP,
        ).style_for_variable("era5:2t")

        assert style.cmap == DEFAULT_CMAP

    def test_double_underscore_normalises_to_colon(self) -> None:
        """'era5__2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable("era5__2t")

        assert style.cmap == "RdBu_r"
        assert style.units == "K"

    def test_hyphen_normalises_to_colon(self) -> None:
        """'era5-2t' normalises to 'era5:2t' and matches that style key."""
        styles = {"era5:2t": {"cmap": "RdBu_r", "units": "K"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable("era5-2t")

        assert style.cmap == "RdBu_r"

    def test_repeated_colons_collapse(self) -> None:
        """A variable name normalising to repeated ':' collapses to a single ':'."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable("era5__-2t")

        assert style.cmap == "RdBu_r"

    def test_default_fallback(self) -> None:
        """An unmatched variable name falls back to the '_default' style."""
        styles = {"_default": {"cmap": "grey"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable(
            "totally:unmatched"
        )

        assert style.cmap == "grey"

    def test_no_match_no_default_falls_back_to_default_cmap(self) -> None:
        """No exact/wildcard/_default match resolves cmap to the bound default."""
        styles = {"era5:2t": {"cmap": "RdBu_r"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable(
            "osisaf:ice_conc"
        )

        assert style.cmap == DEFAULT_CMAP

    def test_bare_wildcard_key_is_skipped(self) -> None:
        """A wildcard key of just '*' (empty prefix) is skipped, not treated as catch-all."""
        styles = {"*": {"cmap": "ignored"}, "_default": {"cmap": "fallback"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable(
            "anything:at_all"
        )

        assert style.cmap == "fallback"

    def test_wildcard_candidate_not_a_dict_falls_back_to_default_cmap(self) -> None:
        """A matching wildcard key whose value isn't a Mapping is logged and skipped."""
        styles: dict[str, Any] = {"era5:*": "not-a-mapping"}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable("era5:2t")

        assert style.cmap == DEFAULT_CMAP

    def test_exact_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test exact variable name matching in styling."""
        style = StyleResolver(variable_styles, DEFAULT_CMAP).style_for_variable(
            "era5:2t"
        )

        assert style.cmap == "RdBu_r"
        assert style.units == "K"

    def test_wildcard_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test wildcard pattern matching in styling."""
        # Add wildcard pattern
        styles_with_wildcard = {
            **variable_styles,
            "era5:q_*": {"cmap": "viridis", "units": "kg/kg"},
        }

        style = StyleResolver(styles_with_wildcard, DEFAULT_CMAP).style_for_variable(
            "era5:q_500"
        )

        assert style.cmap == "viridis"
        assert style.units == "kg/kg"


class TestDefaultCmapFallback:
    def test_matched_style_without_cmap_falls_back_to_default(self) -> None:
        """A matched style that omits cmap still resolves to the bound default."""
        styles = {"era5:2t": {"units": "K"}}

        style = StyleResolver(styles, DEFAULT_CMAP).style_for_variable("era5:2t")

        assert style.cmap == DEFAULT_CMAP
        assert style.units == "K"

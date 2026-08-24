from omegaconf import DictConfig

from icenet_mp.types import DataSpace, PlotSpec


def test_data_space_properties() -> None:
    """Expose the expected DataSpace helper properties."""
    space = DataSpace(channels=3, name="sic", shape=(16, 24))

    assert space.channels == 3
    assert space.name == "sic"
    assert space.shape == (16, 24)
    assert space.area == 384
    assert space.chw == (3, 16, 24)


def test_data_space_coerces_numeric_values_to_ints() -> None:
    """Coerce numeric string values to integer dimensions."""
    space = DataSpace(channels="2", name="sic", shape=("8", "12"))  # type: ignore[arg-type]

    assert space.channels == 2
    assert space.shape == (8, 12)


def test_data_space_round_trip_from_dictconfig() -> None:
    """Round-trip DataSpace values through DictConfig."""
    config = DictConfig({"channels": 4, "name": "weather", "shape": [32, 48]})

    space = DataSpace.from_dict(config)
    result = space.to_dict()

    assert isinstance(result, DictConfig)
    assert result.channels == 4
    assert result.name == "weather"
    assert tuple(result.shape) == (32, 48)


def test_plot_spec_dict_override_preserves_other_values() -> None:
    """Apply dict overrides without changing unspecified PlotSpec values."""
    spec = PlotSpec(variable="sic", colourmap="viridis", video_fps=2)

    result = spec + {"colourmap": "magma", "video_fps": 5}

    assert result.variable == "sic"
    assert result.colourmap == "magma"
    assert result.video_fps == 5
    assert result.include_difference is True


def test_plot_spec_accepts_dictconfig_override() -> None:
    """Apply PlotSpec overrides supplied as DictConfig."""
    spec = PlotSpec(variable="sic")
    override = DictConfig(
        {
            "include_difference": False,
            "colourbar_location": "vertical",
        }
    )

    result = spec + override

    assert result.variable == "sic"
    assert result.include_difference is False
    assert result.colourbar_location == "vertical"


def test_plot_spec_default_styles_are_not_shared() -> None:
    """Keep default per-variable style dictionaries independent."""
    first = PlotSpec()
    second = PlotSpec()

    first.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] = "magma"

    assert second.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] == "Blues_r"


def test_plot_spec_add_none_returns_same_spec() -> None:
    """Return the same PlotSpec when merging with None."""
    spec = PlotSpec(variable="sic")

    assert spec + None is spec

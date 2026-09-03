import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.types.complex_datatypes import DataSpace, ModelStepOutput, PlotSpec


class TestDataSpace:
    """Tests for DataSpace."""

    def test_coerces_numeric_values_to_ints(self) -> None:
        """Coerce numeric string values to integer dimensions."""
        space = DataSpace(channels="2", name="sic", shape=("8", "12"))  # type: ignore[arg-type]

        assert space.channels == 2
        assert space.shape == (8, 12)

    def test_properties(self) -> None:
        """Expose the expected DataSpace helper properties."""
        space = DataSpace(channels=3, name="sic", shape=(16, 24))

        assert space.channels == 3
        assert space.name == "sic"
        assert space.shape == (16, 24)
        assert space.area == 384
        assert space.chw == (3, 16, 24)

    def test_round_trip_from_dictconfig(self) -> None:
        """Round-trip DataSpace values through DictConfig."""
        config = DictConfig({"channels": 4, "name": "weather", "shape": [32, 48]})

        space = DataSpace.from_dict(config)
        result = space.to_dict()

        assert isinstance(result, DictConfig)
        assert result.channels == 4
        assert result.name == "weather"
        assert tuple(result.shape) == (32, 48)


class TestPlotSpec:
    """Tests for PlotSpec."""

    def test_accepts_dictconfig_override(self) -> None:
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

    def test_add_none_returns_same_spec(self) -> None:
        """Return the same PlotSpec when merging with None."""
        spec = PlotSpec(variable="sic")

        assert spec + None is spec

    def test_add_plot_spec_override(self) -> None:
        """Apply overrides supplied as another PlotSpec instance."""
        spec = PlotSpec(variable="sic", colourmap="viridis")
        override = PlotSpec(colourmap="magma", video_fps=5)

        result = spec + override

        assert result.colourmap == "magma"
        assert result.video_fps == 5

    def test_default_styles_are_not_shared(self) -> None:
        """Keep default per-variable style dictionaries independent."""
        first = PlotSpec()
        second = PlotSpec()

        first.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] = "magma"

        assert second.per_variable_styles["sic-ssmis:ice_conc"]["cmap"] == "Blues_r"

    def test_dict_override_preserves_other_values(self) -> None:
        """Apply dict overrides without changing unspecified PlotSpec values."""
        spec = PlotSpec(variable="sic", colourmap="viridis", video_fps=2)

        result = spec + {"colourmap": "magma", "video_fps": 5}

        assert result.variable == "sic"
        assert result.colourmap == "magma"
        assert result.video_fps == 5
        assert result.include_difference is True


class TestModelStepOutput:
    """Tests for ModelStepOutput."""

    def _make_output(self) -> ModelStepOutput:
        return ModelStepOutput(
            prediction=torch.zeros(1, 1, 1, 4, 4),
            target=torch.ones(1, 1, 1, 4, 4),
            loss=torch.tensor(0.5),
        )

    def test_copy_contains_all_keys(self) -> None:
        """copy() dict has exactly the three expected keys."""
        result = self._make_output().copy()
        assert set(result.keys()) == {"prediction", "target", "loss"}

    def test_copy_returns_plain_dict(self) -> None:
        """copy() returns a plain dict, not a ModelStepOutput or other Mapping."""
        result = self._make_output().copy()
        assert type(result) is dict

    def test_copy_values_are_original_tensors(self) -> None:
        """copy() values are the same tensor objects as the original fields."""
        output = self._make_output()
        result = output.copy()
        assert result["prediction"] is output.prediction
        assert result["target"] is output.target
        assert result["loss"] is output.loss

    def test_getitem_raises_key_error_for_unknown_key(self) -> None:
        """__getitem__ raises KeyError for a key that is not one of the three fields."""
        output = self._make_output()
        with pytest.raises(KeyError, match="unknown"):
            output["unknown"]

    def test_len_returns_three(self) -> None:
        """ModelStepOutput always reports a length of three."""
        assert len(self._make_output()) == 3

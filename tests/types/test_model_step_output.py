import torch

from icenet_mp.types.complex_datatypes import ModelStepOutput


class TestModelStepOutput:
    """Test suite for ModelStepOutput."""

    def _make_output(self) -> ModelStepOutput:
        return ModelStepOutput(
            prediction=torch.zeros(1, 1, 1, 4, 4),
            target=torch.ones(1, 1, 1, 4, 4),
            loss=torch.tensor(0.5),
        )

    def test_copy_returns_plain_dict(self) -> None:
        """copy() returns a plain dict, not a ModelStepOutput or other Mapping."""
        result = self._make_output().copy()
        assert type(result) is dict

    def test_copy_contains_all_keys(self) -> None:
        """copy() dict has exactly the three expected keys."""
        result = self._make_output().copy()
        assert set(result.keys()) == {"prediction", "target", "loss"}

    def test_copy_values_are_original_tensors(self) -> None:
        """copy() values are the same tensor objects as the original fields."""
        output = self._make_output()
        result = output.copy()
        assert result["prediction"] is output.prediction
        assert result["target"] is output.target
        assert result["loss"] is output.loss

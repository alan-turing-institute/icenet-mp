import pytest
import torch
from torch import nn

from icenet_mp.losses import TimeWeightedLoss


class TestTimeWeightedLoss:
    """Tests for lead-time weighting behaviour."""

    def test_later_forecast_errors_receive_more_weight(self) -> None:
        """Identical errors at later lead times should contribute more."""
        loss_fn = TimeWeightedLoss(nn.MSELoss(), initial_weight=1.0, final_weight=2.0)
        target = torch.zeros((1, 2, 1, 1, 1))
        early_error = target.clone()
        early_error[:, 0] = 1.0
        late_error = target.clone()
        late_error[:, 1] = 1.0

        early_loss = loss_fn(early_error, target)
        late_loss = loss_fn(late_error, target)

        assert late_loss == pytest.approx(2 * early_loss)

    def test_weights_are_normalised_to_preserve_average_scale(self) -> None:
        """Uniform per-step error should retain the base loss scale."""
        prediction = torch.ones((2, 5, 3, 4, 4))
        target = torch.zeros_like(prediction)
        base_loss = nn.MSELoss()
        loss_fn = TimeWeightedLoss(base_loss, initial_weight=1.0, final_weight=3.0)

        assert loss_fn(prediction, target) == pytest.approx(
            base_loss(prediction, target)
        )

    @pytest.mark.parametrize("n_forecast_steps", [1, 2, 7, 21])
    def test_supports_arbitrary_forecast_horizons(self, n_forecast_steps: int) -> None:
        """Generate the weight schedule dynamically from tensor shape."""
        prediction = torch.randn((3, n_forecast_steps, 2, 4, 4))
        target = torch.randn_like(prediction)
        loss = TimeWeightedLoss(nn.L1Loss())(prediction, target)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_single_step_is_exactly_the_base_loss(self) -> None:
        """A one-step forecast has no relative lead-time weighting."""
        prediction = torch.randn((2, 1, 2, 3, 3))
        target = torch.randn_like(prediction)
        base_loss = nn.HuberLoss(delta=0.5)
        loss_fn = TimeWeightedLoss(base_loss)

        assert torch.equal(loss_fn(prediction, target), base_loss(prediction, target))

    def test_equal_weights_are_exactly_the_base_loss(self) -> None:
        """Equal endpoint weights disable lead-time reweighting."""
        prediction = torch.randn((2, 4, 2, 3, 3))
        target = torch.randn_like(prediction)
        base_loss = nn.MSELoss()
        loss_fn = TimeWeightedLoss(base_loss, initial_weight=1.0, final_weight=1.0)

        assert torch.equal(loss_fn(prediction, target), base_loss(prediction, target))

    @pytest.mark.parametrize(
        ("initial_weight", "final_weight", "message"),
        [
            (0.0, 1.0, "initial_weight"),
            (-1.0, 1.0, "initial_weight"),
            (2.0, 1.0, "final_weight"),
        ],
    )
    def test_rejects_invalid_weight_ranges(
        self, initial_weight: float, final_weight: float, message: str
    ) -> None:
        """Invalid schedules should fail clearly."""
        with pytest.raises(ValueError, match=message):
            TimeWeightedLoss(
                nn.MSELoss(),
                initial_weight=initial_weight,
                final_weight=final_weight,
            )

    def test_rejects_mismatched_shapes(self) -> None:
        """Prediction and target must describe the same forecast tensor."""
        loss_fn = TimeWeightedLoss(nn.MSELoss())

        with pytest.raises(ValueError, match="does not match"):
            loss_fn(torch.zeros((1, 2, 1)), torch.zeros((1, 3, 1)))

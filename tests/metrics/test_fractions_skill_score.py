import pytest
import torch

from icenet_mp.metrics import FractionsSkillScorePerForecastDay


def _single_pixel_field(column: int) -> torch.Tensor:
    field = torch.zeros(1, 1, 1, 5, 5)
    field[..., 2, column] = 1.0
    return field


def test_fss_is_one_for_perfect_forecast() -> None:
    """Return perfect skill for identical sea-ice fields."""
    metric = FractionsSkillScorePerForecastDay(window_size=3)
    target = torch.rand(2, 3, 1, 8, 8)

    metric.update(target, target)

    torch.testing.assert_close(metric.compute(), torch.ones(3))


def test_fss_is_one_when_both_fields_have_no_event() -> None:
    """Treat correctly predicted no-ice fields as perfect agreement."""
    metric = FractionsSkillScorePerForecastDay(window_size=3)
    zeros = torch.zeros(1, 2, 1, 6, 6)

    metric.update(zeros, zeros)

    torch.testing.assert_close(metric.compute(), torch.ones(2))


def test_larger_neighbourhood_rewards_nearby_displacement() -> None:
    """Give partial spatial credit when forecast and target ice are nearby."""
    prediction = _single_pixel_field(2)
    target = _single_pixel_field(3)

    exact = FractionsSkillScorePerForecastDay(window_size=1)
    neighbourhood = FractionsSkillScorePerForecastDay(window_size=3)
    exact.update(prediction, target)
    neighbourhood.update(prediction, target)

    assert exact.compute().item() == pytest.approx(0.0)
    assert 0.0 < neighbourhood.compute().item() < 1.0


def test_fss_accumulates_multiple_batches_per_forecast_day() -> None:
    """Accumulate numerator and denominator across batches before scoring."""
    metric = FractionsSkillScorePerForecastDay(window_size=1)
    metric.update(_single_pixel_field(2), _single_pixel_field(2))
    metric.update(_single_pixel_field(2), _single_pixel_field(3))

    assert metric.compute().shape == (1,)
    assert metric.compute().item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.1}, "threshold"),
        ({"window_size": 0}, "positive odd"),
        ({"window_size": 2}, "positive odd"),
    ],
)
def test_fss_rejects_invalid_configuration(
    kwargs: dict[str, float | int], message: str
) -> None:
    """Validate threshold and neighbourhood width at construction time."""
    with pytest.raises(ValueError, match=message):
        FractionsSkillScorePerForecastDay(**kwargs)


def test_fss_rejects_incompatible_tensor_shapes() -> None:
    """Reject inputs that do not follow the shared NTCHW metric contract."""
    metric = FractionsSkillScorePerForecastDay()

    with pytest.raises(ValueError, match="matching prediction/target"):
        metric.update(torch.zeros(1, 1, 1, 4, 4), torch.zeros(1, 2, 1, 4, 4))

    with pytest.raises(ValueError, match="5 dimensions"):
        metric.update(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))

import pytest
import torch

from icenet_mp.schedulers import WarmupCosineAnnealingLR


def _make_scheduler() -> tuple[torch.optim.SGD, WarmupCosineAnnealingLR]:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        total_epochs=10,
        warmup_epochs=2,
        start_factor=0.1,
        eta_min=0.01,
    )
    return optimizer, scheduler


def _step_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineAnnealingLR,
) -> None:
    optimizer.step()
    scheduler.step()


def test_warmup_cosine_starts_low_and_warms_to_base_lr() -> None:
    optimizer, scheduler = _make_scheduler()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    _step_scheduler(optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55)

    _step_scheduler(optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_warmup_cosine_decays_after_warmup() -> None:
    optimizer, scheduler = _make_scheduler()

    for _ in range(6):
        _step_scheduler(optimizer, scheduler)

    assert 0.01 < optimizer.param_groups[0]["lr"] < 1.0


def test_warmup_cosine_never_restarts_after_training_horizon() -> None:
    optimizer, scheduler = _make_scheduler()
    learning_rates = []

    for _ in range(20):
        _step_scheduler(optimizer, scheduler)
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[9] == pytest.approx(0.01)
    assert learning_rates[10:] == pytest.approx([0.01] * 10)


@pytest.mark.parametrize(
    ("total_epochs", "warmup_epochs", "start_factor", "eta_min"),
    [
        (0, 2, 0.1, 0.01),
        (10, -1, 0.1, 0.01),
        (10, 10, 0.1, 0.01),
        (10, 2, 0.0, 0.01),
        (10, 2, 1.1, 0.01),
        (10, 2, 0.1, -1e-6),
    ],
)
def test_warmup_cosine_rejects_invalid_configuration(
    total_epochs: int,
    warmup_epochs: int,
    start_factor: float,
    eta_min: float,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    with pytest.raises(ValueError):
        WarmupCosineAnnealingLR(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            start_factor=start_factor,
            eta_min=eta_min,
        )

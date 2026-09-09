"""Shared input validation for SIC-only metrics."""

import torch


class SicOnlyMetricMixin:
    """Mixin guarding metric inputs that are only defined for a single SIC channel."""

    def ensure_single_channel(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape[2] != 1 or target.shape[2] != 1:
            msg = (
                f"{type(self).__name__} is only defined for a single "
                f"sea-ice-concentration channel, but got preds with "
                f"{preds.shape[2]} channel(s) and target with {target.shape[2]} "
                f"channel(s). Multi-channel targets (e.g. an auxiliary "
                f"ice-thickness variable) are not supported by this metric."
            )
            raise ValueError(msg)

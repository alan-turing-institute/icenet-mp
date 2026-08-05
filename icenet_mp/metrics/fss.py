"""Fractions Skill Score (FSS) metric.

Computes the FSS of the sea-ice edge at a fixed neighbourhood size, following
Roberts and Lean (2008) and its application to sea-ice edge position by
Melsom et al. (2019, https://doi.org/10.5194/os-15-615-2019). Adapted from the
`effectiveres_icenetv2_FSS` notebook's step-by-step computation.
"""

import torch
import torch.nn.functional as F
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining ice/no-ice, and hence the ice edge


class FractionsSkillScorePerForecastDay(Metric):
    """Fractions Skill Score (FSS) of the sea-ice edge, for use at multiple lead times.

    Each field is first reduced to a binary ice-edge map (cells that are ice but
    border a non-ice cell). The local fraction of edge cells is then computed within
    a fixed `neighborhood_size` x `neighborhood_size` window around every cell. FSS
    compares the mean squared error (MSE) between the predicted and true fraction
    fields to a reference (worst-case) MSE:

        FSS = 1 - MSE / MSE_ref

    FSS is 1 for a perfect match and 0 (or below) for no better than the worst-case
    reference. To assess effective resolution, instantiate this metric once per
    neighbourhood size of interest and compare where the resulting curve crosses 0.5.
    """

    def __init__(self, neighborhood_size: int = 1) -> None:
        """Initialize the FSS metric.

        Parameters
        ----------
        neighborhood_size: int, optional
            Size (in pixels) of the square neighbourhood window used to compute local
            edge-cell fractions. Must be a positive odd integer (default is 1).

        """
        super().__init__()
        if neighborhood_size < 1 or neighborhood_size % 2 == 0:
            msg = "neighborhood_size must be a positive odd integer."
            raise ValueError(msg)
        self.neighborhood_size = neighborhood_size

        self.sum_mse: torch.Tensor
        self.sum_mse_ref: torch.Tensor
        self.count: torch.Tensor

        # States initialized lazily on first update
        self.add_state(
            "sum_mse",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mse_ref",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count", default=torch.tensor([], dtype=torch.long), dist_reduce_fx="sum"
        )

    @staticmethod
    def _binary_edge(ice_mask: torch.Tensor) -> torch.Tensor:
        """Binary ice-edge map: True for ice cells that border a non-ice cell.

        Parameters
        ----------
        ice_mask : torch.Tensor
            Boolean tensor of shape (N, H, W).

        """
        padded = F.pad(ice_mask.float(), (1, 1, 1, 1), value=0.0).bool()
        up = padded[:, :-2, 1:-1]
        down = padded[:, 2:, 1:-1]
        left = padded[:, 1:-1, :-2]
        right = padded[:, 1:-1, 2:]
        return ice_mask & (
            (ice_mask != up)
            | (ice_mask != down)
            | (ice_mask != left)
            | (ice_mask != right)
        )

    def _neighborhood_fraction(self, edge: torch.Tensor) -> torch.Tensor:
        """Local fraction of edge cells within an n x n window of each cell.

        Parameters
        ----------
        edge : torch.Tensor
            Boolean tensor of shape (N, H, W).

        """
        n = self.neighborhood_size
        kernel = torch.ones((1, 1, n, n), dtype=torch.float32, device=edge.device)
        summed = F.conv2d(edge.float().unsqueeze(1), kernel, padding=n // 2)
        return summed.squeeze(1) / (n * n)

    def _config_stats(
        self, lambda_truth: torch.Tensor, lambda_model: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average MSE and reference MSE across all n x n non-overlapping sub-grids.

        Sampling every n-th cell at each of the n x n offsets removes the dependency
        of the MSE on a single, arbitrary alignment of the neighbourhood window.

        Parameters
        ----------
        lambda_truth : torch.Tensor
            Local edge-cell fraction field for the target, shape (N, H, W).
        lambda_model : torch.Tensor
            Local edge-cell fraction field for the prediction, shape (N, H, W).

        """
        n = self.neighborhood_size
        mse_terms = []
        mse_ref_terms = []
        for dx in range(n):
            for dy in range(n):
                truth_slice = lambda_truth[:, dx::n, dy::n]
                model_slice = lambda_model[:, dx::n, dy::n]
                area = truth_slice.shape[-2] * truth_slice.shape[-1]
                if area == 0:
                    continue
                diff = model_slice - truth_slice
                mse_terms.append(diff.pow(2).mean(dim=(-2, -1)))
                term1 = truth_slice.pow(2).sum(dim=(-2, -1)) + model_slice.pow(2).sum(
                    dim=(-2, -1)
                )
                term2 = (1 - truth_slice).pow(2).sum(dim=(-2, -1)) + (
                    1 - model_slice
                ).pow(2).sum(dim=(-2, -1))
                mse_ref_terms.append(torch.minimum(term1, term2) / area)
        mse = torch.stack(mse_terms, dim=0).mean(dim=0)
        mse_ref = torch.stack(mse_ref_terms, dim=0).mean(dim=0)
        return mse, mse_ref

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update the FSS accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        batch_size, n_steps, n_channels, height, width = preds.shape

        preds_mask = (preds > SEA_ICE_THRESHOLD).reshape(-1, height, width)
        target_mask = (target > SEA_ICE_THRESHOLD).reshape(-1, height, width)

        lambda_model = self._neighborhood_fraction(self._binary_edge(preds_mask))
        lambda_truth = self._neighborhood_fraction(self._binary_edge(target_mask))

        mse, mse_ref = self._config_stats(lambda_truth, lambda_model)

        mse = mse.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
        mse_ref = mse_ref.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
        batch_count = torch.full(
            (n_steps,), batch_size * n_channels, dtype=torch.long, device=preds.device
        )

        if self.sum_mse.numel() == 0:
            self.sum_mse = mse
            self.sum_mse_ref = mse_ref
            self.count = batch_count
        else:
            self.sum_mse += mse
            self.sum_mse_ref += mse_ref
            self.count += batch_count

    def compute(self) -> torch.Tensor:
        """Compute the final FSS per lead time."""
        if self.count.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.sum_mse.device)

        mean_mse = self.sum_mse / self.count
        mean_mse_ref = self.sum_mse_ref / self.count
        return 1 - mean_mse / mean_mse_ref

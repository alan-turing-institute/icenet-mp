"""Adjusted-MSE (AMSE) spectral anti-blur loss on a flat grid.

Adaptation of the spherical-harmonic AMSE loss of Subich et al. (2025),
"Fixing the Double Penalty in Data-Driven Weather Forecasting Through a
Modified Spherical Harmonic Loss Function" (arXiv:2501.19374), to the flat
equal-area EASE2 grids used by icenet-mp. Independent reimplementation of the
published formula; reference implementation (spherical, JAX) at
https://github.com/csubich/graphcast (branch ``amse``).

Pointwise squared-error losses reward damping the amplitude of scales the
model cannot predict exactly (the "double penalty"): per scale group, the
MSE-optimal amplitude ratio equals the coherence with the target, so partially
predictable fine scales are systematically blurred. AMSE removes this
incentive. Writing per scale group k the powers P_k(x), P_k(y) and coherence
Coh_k(x, y) of prediction x and target y,

    MSE(x, y)  = sum_k [ (sqrt(P_k(x)) - sqrt(P_k(y)))^2
                         + 2 * sqrt(P_k(x) * P_k(y)) * (1 - Coh_k(x, y)) ]
    AMSE(x, y) = sum_k [ (sqrt(P_k(x)) - sqrt(P_k(y)))^2
                         + 2 * max(P_k(x), P_k(y)) * (1 - Coh_k(x, y)) ]

so under AMSE the decorrelation penalty can no longer be reduced by shrinking
the predicted amplitude, and the amplitude optimum at any fixed coherence is
P_k(x) = P_k(y): the target's power spectrum is preserved scale by scale.

On the flat grid the orthonormal decomposition is the 2D DFT and the scale
groups are annular bins of the isotropic wavenumber |k| = sqrt(kx^2 + ky^2);
Parseval's identity makes the group decomposition of MSE exact, which is the
only property the construction needs (Subich et al., Sec. 2.2 and 4.2).

Scope: designed and tested for spatial-field predictions (single-stage
encode-process-decode). Using this loss with models whose training loss lives
in a different space (e.g. DDPM v-space) is untested.
"""

import torch
from torch import nn
from torch.nn import functional

# Type alias for the cached per-(H, W, device) binning tensors:
# (flat mode->bin index, flat mode weights, indices of non-empty bins)
BinCache = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Fields must carry at least (batch-like, H, W) dimensions.
MIN_FIELD_NDIM = 3


class AMSELoss(nn.Module):
    """Adjusted-MSE (anti-double-penalty) spectral loss on a flat grid.

    Two modes:

    - ``"hybrid"`` (default): the repo's standard Huber loss plus
      ``spectral_weight`` times the AMSE *excess*
      ``sum_k 2 * (max(P_k(x), P_k(y)) - sqrt(P_k(x) * P_k(y))) * (1 - Coh_k)``,
      which is >= 0, zero iff per-bin spectra match or coherence is perfect,
      and reduces to the unmodified Huber control at ``spectral_weight=0``.
    - ``"pure"``: the AMSE formula itself (per-bin amplitude + decorrelation
      terms plus a direct DC term), averaged over fields.

    Numerical guards follow the reference implementation: ``eps`` inside the
    geometric-mean square root (the coherence denominator is singular for a
    zero field), coherence clamped at 1 from above (Cauchy-Schwarz can be
    violated by the eps), the mean/DC component handled as a direct squared
    difference (catastrophic cancellation), and all spectral arithmetic in
    float32.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        spectral_weight: float = 0.1,
        delta: float = 0.5,
        merge_bins_below: int = 4,
        eps: float = 1e-12,
    ) -> None:
        """Initialise the AMSE loss.

        Args:
            mode: "hybrid" (Huber + spectral_weight * AMSE excess) or "pure"
                (the AMSE formula alone).
            spectral_weight: weight of the spectral excess term in hybrid mode.
            delta: Huber transition point for the hybrid base term (matches
                the repo's default Huber loss).
            merge_bins_below: annuli with rounded |k| <= this value share one
                bin (the lowest annuli contain too few modes for a stable
                per-sample coherence estimate).
            eps: additive guard inside sqrt(P_k(x) * P_k(y)); keeps gradients
                finite when either field has an empty scale bin.

        """
        super().__init__()
        if mode not in ("hybrid", "pure"):
            msg = f"Unknown AMSELoss mode: {mode!r} (expected 'hybrid' or 'pure')"
            raise ValueError(msg)
        if merge_bins_below < 1:
            msg = f"merge_bins_below must be >= 1, got {merge_bins_below}"
            raise ValueError(msg)
        self.mode = mode
        self.spectral_weight = spectral_weight
        self.delta = delta
        self.merge_bins_below = merge_bins_below
        self.eps = eps
        # Lazily-built cache of binning tensors, keyed by (H, W, device).
        # A plain dict (not buffers) so it never enters the state dict.
        self._bin_cache: dict[tuple[int, int, str], BinCache] = {}

    def _binning(self, height: int, width: int, device: torch.device) -> BinCache:
        """Return (bin index, mode weight, non-empty bin ids) for an H x W grid."""
        key = (height, width, str(device))
        cached = self._bin_cache.get(key)
        if cached is not None:
            return cached
        ky = torch.fft.fftfreq(height, device=device) * height
        kx = torch.fft.rfftfreq(width, device=device) * width
        kmag = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        bins = torch.round(kmag).long()
        # Merge the sparse low-|k| annuli (excluding DC) into one bin.
        bins = torch.where(
            (bins >= 1) & (bins < self.merge_bins_below), self.merge_bins_below, bins
        )
        # rfft halves the x axis: interior columns stand for +-kx pairs and
        # count twice; column 0 and (for even W) the Nyquist column appear once.
        weights = torch.full(kmag.shape, 2.0, device=device)
        weights[:, 0] = 1.0
        if width % 2 == 0:
            weights[:, -1] = 1.0
        # The DC mode is excluded from the annuli (handled as a direct term).
        weights[0, 0] = 0.0
        bins_flat = bins.reshape(-1)
        weights_flat = weights.reshape(-1)
        counts = torch.zeros(int(bins_flat.max()) + 1, device=device)
        counts.scatter_add_(0, bins_flat, (weights_flat > 0).float())
        present = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
        result = (bins_flat, weights_flat, present)
        self._bin_cache[key] = result
        return result

    def binned_spectra(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-field, per-bin spectral statistics of the mean-removed fields.

        Returns:
            Tuple ``(P, T, C, dc)`` where ``P[i, b]`` / ``T[i, b]`` are the
            prediction/target powers of field ``i`` in |k|-bin ``b``,
            ``C[i, b]`` is the real cross-spectrum, and ``dc[i]`` is the
            squared difference of the field means. Powers are normalised so
            that ``sum_b (P + T - 2C) + dc`` equals the per-field MSE
            (Parseval's identity).

        """
        if prediction.shape != target.shape:
            msg = (
                f"prediction shape {tuple(prediction.shape)} "
                f"!= target shape {tuple(target.shape)}"
            )
            raise ValueError(msg)
        if prediction.ndim < MIN_FIELD_NDIM:
            msg = (
                f"expected at least {MIN_FIELD_NDIM} dimensions, got {prediction.ndim}"
            )
            raise ValueError(msg)
        height, width = prediction.shape[-2], prediction.shape[-1]
        fields_p = prediction.reshape(-1, height, width).float()
        fields_t = target.reshape(-1, height, width).float()
        mean_p = fields_p.mean(dim=(-2, -1))
        mean_t = fields_t.mean(dim=(-2, -1))
        dc = (mean_p - mean_t) ** 2
        spec_p = torch.fft.rfft2(fields_p - mean_p[:, None, None], norm="ortho")
        spec_t = torch.fft.rfft2(fields_t - mean_t[:, None, None], norm="ortho")
        bins_flat, weights_flat, present = self._binning(height, width, fields_p.device)
        n_fields = spec_p.shape[0]
        n_bins_total = int(bins_flat.max()) + 1
        scale = 1.0 / (height * width)
        index = bins_flat.unsqueeze(0).expand(n_fields, -1)

        def accumulate(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(n_fields, n_bins_total, device=values.device)
            out.scatter_add_(
                1, index, (values * weights_flat).reshape(n_fields, -1) * scale
            )
            return out.index_select(1, present)

        power_p = accumulate(spec_p.abs().reshape(n_fields, -1) ** 2)
        power_t = accumulate(spec_t.abs().reshape(n_fields, -1) ** 2)
        cross = accumulate((spec_p * spec_t.conj()).real.reshape(n_fields, -1))
        return power_p, power_t, cross, dc

    def spectral_excess(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-field AMSE excess ``AMSE - MSE`` (the pure anti-blur surcharge).

        Non-negative; zero iff, in every |k|-bin, the two power spectra match
        or the coherence is perfect. Shrinking predicted amplitude below the
        target's cannot reduce this term.
        """
        power_p, power_t, cross, _ = self.binned_spectra(prediction, target)
        geo_mean = torch.sqrt(self.eps + power_p * power_t)
        coherence = torch.clamp(cross / geo_mean, max=1.0)
        excess_weight = (torch.maximum(power_p, power_t) - geo_mean).clamp_min(0.0)
        return (2.0 * excess_weight * (1.0 - coherence)).sum(dim=1)

    def pure_amse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-field AMSE (amplitude + adjusted decorrelation terms + DC term)."""
        power_p, power_t, cross, dc = self.binned_spectra(prediction, target)
        geo_mean = torch.sqrt(self.eps + power_p * power_t)
        coherence = torch.clamp(cross / geo_mean, max=1.0)
        amplitude = (power_p + power_t - 2.0 * geo_mean).clamp_min(0.0)
        decorrelation = 2.0 * torch.maximum(power_p, power_t) * (1.0 - coherence)
        return (amplitude + decorrelation).sum(dim=1) + dc

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss for prediction/target of shape [..., H, W]."""
        if self.mode == "pure":
            return self.pure_amse(prediction, target).mean()
        base = functional.huber_loss(
            prediction.float(), target.float(), delta=self.delta
        )
        return (
            base
            + self.spectral_weight * self.spectral_excess(prediction, target).mean()
        )

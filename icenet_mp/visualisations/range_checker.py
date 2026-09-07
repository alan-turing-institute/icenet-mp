import numpy as np

from icenet_mp.types import RangeCheckReport

# Constants for range check thresholds
_ZERO_THRESHOLD = 1e-12
_MAGNITUDE_LOW_THRESHOLD = 0.1
_MAGNITUDE_HIGH_THRESHOLD = 10.0
_SIGNIFICANT_VALUE_THRESHOLD = 1e-6


class RangeChecker:
    """Checks ground-truth/prediction arrays for magnitude and display-clipping issues."""

    def check(  # noqa: PLR0913
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        outside_warn: float = 0.05,
        severe_outside: float = 0.20,
        include_shared_range_mismatch_check: bool = True,
    ) -> RangeCheckReport:
        """Compute warnings about display/clipping and magnitude mismatches between ground truth and prediction arrays.

        This always runs:
          - a *magnitude mismatch* check (possible units/scaling error),
          - a *shared-scale clipping* check (many prediction values outside [vmin,vmax]),
        and annotates each warning to clarify whether the issue is about *display* (i.e. values
        will be clipped under a shared colourbar) or about *absolute magnitude* (possible scaling).
        """
        warnings: list[str] = []

        # --- 1) Magnitude / units mismatch check (absolute magnitude) ---
        warnings.extend(self._check_magnitude_mismatch(ground_truth, prediction))

        # --- 2) Shared-scale clipping / display mismatch check ---
        if include_shared_range_mismatch_check:
            # Compute the display bounds to use for the clipping check.
            # If callers passed None, fall back to the conventional 0..1 sea-ice range.
            vmin_for_check = 0.0 if vmin is None else float(vmin)
            vmax_for_check = 1.0 if vmax is None else float(vmax)
            if vmax_for_check <= vmin_for_check:
                # Invalid interval: we cannot run the clipping check, but inform the user.
                warnings.append("COLOUR ISSUE: Shared display range invalid.")
            else:
                warnings.extend(
                    self._check_display_clipping(
                        prediction,
                        vmin_for_check,
                        vmax_for_check,
                        outside_warn,
                        severe_outside,
                    )
                )

        return RangeCheckReport(warnings=self._dedupe_category_prefixes(warnings))

    def _check_magnitude_mismatch(
        self, ground_truth: np.ndarray, prediction: np.ndarray
    ) -> list[str]:
        """Check for magnitude/units mismatch between ground truth and prediction."""
        warnings: list[str] = []

        def _robust_median(arr: np.ndarray) -> float:
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                return 0.0
            return float(np.nanmedian(vals))

        def _robust_percentile(arr: np.ndarray, p: float) -> float:
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                return 0.0
            return float(np.nanpercentile(vals, p))

        gt_med = _robust_median(ground_truth)
        pred_med = _robust_median(prediction)

        # Try to avoid false positives when GT median is zero by falling back to 90th percentiles
        if abs(gt_med) < _ZERO_THRESHOLD:
            gt_ref = _robust_percentile(ground_truth, 90.0)
            pred_ref = _robust_percentile(prediction, 90.0)
        else:
            gt_ref = gt_med
            pred_ref = pred_med

        # Only examine when we have meaningful reference values
        if gt_ref > 0:
            ratio = pred_ref / gt_ref
            if ratio < _MAGNITUDE_LOW_THRESHOLD:
                warnings.append(f"MAGNITUDE: Prediction values too low (~{ratio:.2g}x)")
            elif ratio > _MAGNITUDE_HIGH_THRESHOLD:
                warnings.append(
                    f"MAGNITUDE: Prediction values too high (~{ratio:.2g}x)"
                )
        else:
            # If GT reference is zero, still warn if predictions have significant non-zero mass
            pred90 = _robust_percentile(prediction, 90.0)
            if pred90 > _SIGNIFICANT_VALUE_THRESHOLD:
                warnings.append(
                    "MAGNITUDE: Ground truth median is zero but predictions have substantial values "
                    f"(90th percentile ≈ {pred90:.3g})."
                )

        return warnings

    def _check_display_clipping(
        self,
        prediction: np.ndarray,
        vmin: float,
        vmax: float,
        outside_warn: float,
        severe_outside: float,
    ) -> list[str]:
        """Check for display clipping issues with shared colorbar."""
        warnings: list[str] = []

        pred_finite = np.isfinite(prediction)
        total_pred = float(np.sum(pred_finite))

        if total_pred > 0:
            pred_flat = prediction[np.isfinite(prediction)]
            frac_below = float(np.sum(pred_flat < vmin)) / pred_flat.size
            frac_above = float(np.sum(pred_flat > vmax)) / pred_flat.size

            # Above-vmax checks
            if frac_above >= severe_outside:
                warnings.append(
                    f"COLOUR ISSUE: !!! {frac_above:.0%} of prediction values above colour limit={vmax}"
                )
            elif frac_above >= outside_warn:
                warnings.append(
                    f"COLOUR ISSUE: Clipping likely: {frac_above:.0%} of prediction values above colour limit={vmax}"
                )

            # Below-vmin checks
            if frac_below >= severe_outside:
                warnings.append(
                    f"COLOUR ISSUE: !!! {frac_below:.0%} of prediction values below colour limit={vmin}"
                )
            elif frac_below >= outside_warn:
                warnings.append(
                    f"COLOUR ISSUE: Clipping likely: {frac_below:.0%} of prediction values below colour limit={vmin}"
                )
        else:
            # No finite predictions -> warn the user
            warnings.append("COLOUR ISSUE: No finite prediction values found.")

        return warnings

    @staticmethod
    def _dedupe_category_prefixes(warnings: list[str]) -> list[str]:
        """Keep the category prefix only on the first occurrence per category."""
        processed: list[str] = []
        prefixes: tuple[str, ...] = ("MAGNITUDE:", "COLOUR ISSUE:")
        seen: set[str] = set()

        for msg in warnings:
            for prefix in prefixes:
                if msg.startswith(prefix):
                    if prefix in seen:
                        processed.append(msg[len(prefix) :].lstrip())
                    else:
                        processed.append(msg)
                        seen.add(prefix)
                    break
            else:
                processed.append(msg)

        return processed

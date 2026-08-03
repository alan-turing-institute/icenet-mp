"""Shared CLI helpers for opt-in plotting of spatial RF screening results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

    from icenet_mp.input_explainability.spatial_rf import SpatialRFResult

logger = logging.getLogger(__name__)


def maybe_plot_spatial_rf_results(
    config: DictConfig,
    result: SpatialRFResult,
    output_dir: Path,
) -> None:
    """Render spatial RF visualisations when ``rf.spatial.plot_results`` is true.

    This helper is intentionally permissive: any plotting failure is logged and
    swallowed so the surrounding pipeline (and the existing JSON / text report
    written by :func:`save_spatial_rf_results`) is never broken.
    """
    spatial_cfg = (config.get("rf", {}) or {}).get("spatial", {}) or {}
    if not bool(spatial_cfg.get("plot_results", False)):
        return
    try:  # pragma: no cover - defensive guard around optional matplotlib
        from icenet_mp.visualisations.spatial_rf_plots import (  # noqa: PLC0415
            plot_spatial_rf_results,
        )

        paths = plot_spatial_rf_results(result, output_dir)
    except Exception as exc:  # noqa: BLE001 - deliberate broad catch
        logger.warning(
            "Spatial RF plotting failed (%s); JSON and text report remain valid.",
            exc,
        )
        return
    written = [name for name, path in paths.items() if path is not None]
    if written:
        logger.info(
            "Spatial RF plots written to %s: %s",
            output_dir,
            ", ".join(written),
        )
    skipped = [name for name, path in paths.items() if path is None]
    logger.info(
        "Spatial RF plots skipped: %s",
        ", ".join(skipped) if skipped else "none",
    )

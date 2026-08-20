"""Opt-in visualisation plots for the sampled-map Random Forest screening.

This module provides small, focused PNG visualisations that complement the JSON
output written by :func:`icenet_mp.input_explainability.spatial_rf.save_spatial_rf_results`.
Each plot function takes a :class:`SpatialRFResult` (or two, for the overlap
plot) and writes one PNG into the supplied directory. All functions are
independent, importable in isolation, and silently return ``None`` when the
data they need is absent so the surrounding pipeline is never broken.

The functions are deliberately decoupled from any CLI or Hydra entry point —
the orchestrator :func:`plot_spatial_rf_results` simply calls every plotter
and returns the mapping of plot names to saved paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from icenet_mp.input_explainability.spatial_rf import SpatialRFResult

# Ensure a non-interactive backend before any figure is created.
mpl.use("Agg")

logger = logging.getLogger(__name__)


# Maximum rasterisation DPI (kept small to keep PNG sizes modest).
_MAX_DPI = 120

# Canonical reliability categories produced by the screening pipeline.
_RELIABILITY_LEVELS: tuple[str, ...] = ("low_evidence", "candidate", "stable")
_RELIABILITY_COLOURS: dict[str, str] = {
    "low_evidence": "#d6604d",
    "candidate": "#fdae61",
    "stable": "#1a9850",
}

# Strata sampled by the spatial RF screening pipeline.
_STRATA: tuple[str, ...] = ("open_water", "pack_ice", "marginal_ice")


def _has_any_importance(result: SpatialRFResult) -> bool:
    """Return ``True`` when at least one lead retains importance information."""
    return any(lead.importance is not None for lead in result.leads)


def _importance_groups(result: SpatialRFResult) -> list[str]:
    """Return the sorted union of group keys across leads with importance."""
    groups: set[str] = set()
    for lead in result.leads:
        if lead.importance is not None:
            groups.update(lead.importance)
    return sorted(groups)


def _save_figure(fig: Figure, path: Path) -> Path:
    """Save ``fig`` to ``path`` at the configured DPI and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_MAX_DPI)
    plt.close(fig)
    return path


def plot_lead_importance_heatmap(
    result: SpatialRFResult,
    output_dir: Path,
) -> Path | None:
    """Plot a heatmap of mean MSE increase for every (lead x group) pair.

    Reads ``lead.importance[group].mean_mse_increase`` for all leads and groups.
    Silently returns ``None`` when ``importance`` is missing for every lead.
    """
    if not _has_any_importance(result):
        return None

    groups = _importance_groups(result)
    if not groups:
        return None
    leads = [lead.lead for lead in result.leads]
    matrix = np.full((len(leads), len(groups)), np.nan)
    for row, lead in enumerate(result.leads):
        if lead.importance is None:
            continue
        for col, group in enumerate(groups):
            entry = lead.importance.get(group)
            if entry is not None:
                matrix[row, col] = float(entry.get("mean_mse_increase", np.nan))
    if not np.isfinite(matrix).any():
        return None

    fig, ax = plt.subplots(
        figsize=(max(6, len(groups) * 0.9), max(4, len(leads) * 0.5))
    )
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(leads)))
    ax.set_yticklabels([f"Lead {lead}" for lead in leads], fontsize=8)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Lead (days)")
    ax.set_title("Spatial RF — mean MSE increase per (lead x group)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean MSE increase")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "importance_heatmap.png")


def _build_reliability_grid(
    result: SpatialRFResult,
    groups: list[str],
    leads: list[int],
) -> np.ndarray:
    """Build the integer grid that backs the reliability plot."""
    grid = np.full((len(leads), len(groups)), -1, dtype=int)
    for row, lead in enumerate(result.leads):
        if lead.importance is None:
            continue
        for col, group in enumerate(groups):
            entry = lead.importance.get(group)
            if entry is None:
                continue
            reliability = entry.get("reliability")
            if reliability in _RELIABILITY_LEVELS:
                grid[row, col] = _RELIABILITY_LEVELS.index(reliability)
    return grid


def plot_reliability_grid(
    result: SpatialRFResult,
    output_dir: Path,
) -> Path | None:
    """Plot a categorical grid of reliability classifications per (lead x group).

    Reads ``lead.importance[group].reliability`` (``"stable"``, ``"candidate"``,
    or ``"low_evidence"``). Silently returns ``None`` when no lead retains
    importance information.
    """
    if not _has_any_importance(result):
        return None

    groups = _importance_groups(result)
    if not groups:
        return None
    leads = [lead.lead for lead in result.leads]
    n_levels = len(_RELIABILITY_LEVELS)
    grid = _build_reliability_grid(result, groups, leads)
    if (grid < 0).all():
        return None

    cmap = ListedColormap(
        [_RELIABILITY_COLOURS[level] for level in _RELIABILITY_LEVELS]
    )
    bounds = np.arange(-0.5, n_levels, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(max(6, len(groups) * 0.9), max(4, len(leads) * 0.5))
    )
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(leads)))
    ax.set_yticklabels([f"Lead {lead}" for lead in leads], fontsize=8)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Lead (days)")
    ax.set_title("Spatial RF — reliability classification per (lead x group)")

    for row in range(len(leads)):
        for col in range(len(groups)):
            value = grid[row, col]
            if value < 0:
                continue
            ax.text(
                col,
                row,
                _RELIABILITY_LEVELS[value],
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.8, ticks=range(n_levels))
    cbar.ax.set_yticklabels(list(_RELIABILITY_LEVELS))
    cbar.set_label("Reliability")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "reliability_grid.png")


def _collect_confirmation_metrics(
    result: SpatialRFResult,
) -> list[tuple[int, dict[str, float]]]:
    """Aggregate per-lead confirmation metrics across all groups."""
    metrics_by_lead: list[tuple[int, dict[str, float]]] = []
    for lead in result.leads:
        if lead.importance is None:
            continue
        lead_metrics: dict[str, float] = {}
        for entry in lead.importance.values():
            for key in ("add_to_sic_gain", "drop_from_full_loss"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    lead_metrics[key] = float(value)
        if lead_metrics:
            metrics_by_lead.append((lead.lead, lead_metrics))
    return metrics_by_lead


def plot_confirmation_refits(
    result: SpatialRFResult,
    output_dir: Path,
) -> Path | None:
    """Plot confirmation-group ``add_to_sic_gain`` and ``drop_from_full_loss``.

    Reads the per-group confirmation metrics emitted by the spatial RF
    pipeline. Silently returns ``None`` when no importance entry carries the
    ``add_to_sic_gain`` or ``drop_from_full_loss`` keys.
    """
    metrics_by_lead = _collect_confirmation_metrics(result)
    if not metrics_by_lead:
        return None

    leads = [item[0] for item in metrics_by_lead]
    add_gains = [item[1].get("add_to_sic_gain", 0.0) for item in metrics_by_lead]
    drop_losses = [item[1].get("drop_from_full_loss", 0.0) for item in metrics_by_lead]

    fig, ax = plt.subplots(figsize=(max(6, len(leads) * 0.8), 5))
    positions = np.arange(len(leads))
    width = 0.35
    ax.bar(
        positions - width / 2,
        add_gains,
        width,
        label="add_to_sic_gain",
        color="#1a9850",
    )
    ax.bar(
        positions + width / 2,
        drop_losses,
        width,
        label="drop_from_full_loss",
        color="#d6604d",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Lead {lead}" for lead in leads])
    ax.set_xlabel("Lead (days)")
    ax.set_ylabel("MSE delta (vs reference)")
    ax.set_title("Spatial RF — confirmation-group refit deltas per lead")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir / "confirmation_refits.png")


def _stratum_availability(result: SpatialRFResult) -> set[str]:
    """Return the set of strata observed across all leads with importance."""
    available: set[str] = set()
    for lead in result.leads:
        if lead.importance is None:
            continue
        for entry in lead.importance.values():
            available.update(entry.get("by_stratum", {}).keys())
    return available


def _stratum_values_for_group(
    result: SpatialRFResult,
    group: str,
    stratum: str,
) -> list[float]:
    """Collect per-lead ``mean_mse_increase`` for one (group, stratum) pair."""
    values: list[float] = []
    for lead in result.leads:
        if lead.importance is None:
            continue
        entry = lead.importance.get(group)
        if entry is None:
            continue
        stratum_entry = entry.get("by_stratum", {}).get(stratum)
        if stratum_entry is None:
            continue
        value = stratum_entry.get("mean_mse_increase")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def plot_stratum_importance(
    result: SpatialRFResult,
    output_dir: Path,
) -> Path | None:
    """Plot a heatmap of ``mean_mse_increase`` per (group x stratum).

    Reads ``lead.importance[group].by_stratum[stratum].mean_mse_increase`` for
    every stratum in :data:`_STRATA`. Silently returns ``None`` when any of
    those strata is absent from the result.
    """
    if not _has_any_importance(result):
        return None

    available = _stratum_availability(result)
    if any(stratum not in available for stratum in _STRATA):
        return None

    groups = _importance_groups(result)
    if not groups:
        return None

    matrix = np.zeros((len(groups), len(_STRATA)))
    for row, group in enumerate(groups):
        for col, stratum in enumerate(_STRATA):
            values = _stratum_values_for_group(result, group, stratum)
            if values:
                matrix[row, col] = float(np.mean(values))

    fig, ax = plt.subplots(
        figsize=(max(6, len(_STRATA) * 1.6), max(4, len(groups) * 0.5))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(_STRATA)))
    ax.set_xticklabels(list(_STRATA), rotation=20, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Feature group")
    ax.set_title("Spatial RF — mean MSE increase per (group x stratum)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean MSE increase")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "stratum_importance.png")


def _mean_per_group(result: SpatialRFResult) -> dict[str, float]:
    """Return each group's mean ``mean_mse_increase`` across leads."""
    totals: dict[str, list[float]] = {}
    for lead in result.leads:
        if lead.importance is None:
            continue
        for group, entry in lead.importance.items():
            value = entry.get("mean_mse_increase")
            if isinstance(value, (int, float)) and np.isfinite(value):
                totals.setdefault(group, []).append(float(value))
    return {group: float(np.mean(values)) for group, values in totals.items()}


def plot_target_mode_overlap(
    result_a: SpatialRFResult,
    result_b: SpatialRFResult,
    output_dir: Path,
    *,
    top_k: int = 10,
) -> Path | None:
    """Scatter plot of mean importance overlap between two screening runs.

    Reads ``lead.importance[group].mean_mse_increase`` from each result,
    averages across leads, and highlights the top-``top_k`` groups selected by
    each run (overlap vs. one-sided-only). Silently returns ``None`` when
    either result lacks importance or when the two runs share no groups.
    """
    if not _has_any_importance(result_a) or not _has_any_importance(result_b):
        return None

    means_a = _mean_per_group(result_a)
    means_b = _mean_per_group(result_b)
    common = sorted(set(means_a) & set(means_b))
    if not common:
        return None

    top_a = set(sorted(common, key=lambda g: -means_a[g])[:top_k])
    top_b = set(sorted(common, key=lambda g: -means_b[g])[:top_k])
    overlap = sorted(top_a & top_b)
    only_a = sorted(top_a - top_b)
    only_b = sorted(top_b - top_a)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        [means_a[g] for g in common],
        [means_b[g] for g in common],
        c="lightgrey",
        alpha=0.6,
        label=f"all {len(common)}",
    )
    if overlap:
        ax.scatter(
            [means_a[g] for g in overlap],
            [means_b[g] for g in overlap],
            c="#1a9850",
            label=f"top-{top_k} overlap ({len(overlap)})",
        )
    if only_a:
        ax.scatter(
            [means_a[g] for g in only_a],
            [means_b[g] for g in only_a],
            c="#fdae61",
            label="only in result A",
        )
    if only_b:
        ax.scatter(
            [means_a[g] for g in only_b],
            [means_b[g] for g in only_b],
            c="#d6604d",
            label="only in result B",
        )
    ax.set_xlabel("Mean MSE increase — result A")
    ax.set_ylabel("Mean MSE increase — result B")
    ax.set_title(f"Spatial RF — top-{top_k} importance overlap between two runs")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir / "target_mode_overlap.png")


def plot_model_quality(
    result: SpatialRFResult,
    output_dir: Path,
) -> Path | None:
    """Grouped bar chart comparing RF ``mse`` to ``baseline_mse`` per lead.

    Reads ``lead.mse`` and ``lead.baseline_mse`` for every lead. Silently
    returns ``None`` when either list is empty.
    """
    leads: list[int] = []
    mses: list[float] = []
    baselines: list[float] = []
    for lead in result.leads:
        leads.append(lead.lead)
        mses.append(float(lead.mse))
        baselines.append(float(lead.baseline_mse))
    if not leads or not mses or not baselines:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(leads) * 0.8), 5))
    positions = np.arange(len(leads))
    width = 0.35
    ax.bar(
        positions - width / 2,
        mses,
        width,
        label="RF MSE",
        color="#1a9850",
    )
    ax.bar(
        positions + width / 2,
        baselines,
        width,
        label="Baseline MSE",
        color="#d6604d",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Lead {lead}" for lead in leads])
    ax.set_xlabel("Lead (days)")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Spatial RF — model quality vs persistence per lead")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir / "model_quality.png")


def plot_spatial_rf_results(
    result: SpatialRFResult,
    output_dir: Path,
    *,
    other: SpatialRFResult | None = None,
    top_k: int = 10,
) -> dict[str, Path | None]:
    """Run every spatial RF plot and return their saved paths.

    Each entry is the path to the saved PNG when the data was available, or
    ``None`` when the corresponding plot function chose to skip. The overlap
    plot is only produced when ``other`` is supplied.
    """
    return {
        "importance_heatmap": plot_lead_importance_heatmap(result, output_dir),
        "reliability_grid": plot_reliability_grid(result, output_dir),
        "confirmation_refits": plot_confirmation_refits(result, output_dir),
        "stratum_importance": plot_stratum_importance(result, output_dir),
        "model_quality": plot_model_quality(result, output_dir),
        "target_mode_overlap": (
            plot_target_mode_overlap(result, other, output_dir, top_k=top_k)
            if other is not None
            else None
        ),
    }

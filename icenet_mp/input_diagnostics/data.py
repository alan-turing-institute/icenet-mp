"""Shared data extraction for input diagnostics.

Resolves dataset paths from Hydra config, builds SingleDataset instances, and
constructs the sample matrix (one row per timestep, one column per variable).

This module is the single source of truth for data loading across VIF, PCA, and EOF.
It bypasses ``CommonDataModule`` because diagnostics do not require a prediction target.

"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig, ListConfig

if TYPE_CHECKING:
    from icenet_mp.data_loaders.single_dataset import SingleDataset

logger = logging.getLogger(__name__)


def resolve_datasets(
    config: DictConfig,
) -> tuple[dict[str, list[Path]], dict[str, list[str]]]:
    """Resolve dataset group names to zarr paths and variable selections from config.

    Mirrors the path-resolution logic in ``CommonDataModule`` but does not require a
    prediction target group.

    Args:
        config: Hydra-composed config.

    Returns:
        Tuple of (group_paths, group_variables) where group_paths maps group_as names to
        lists of zarr paths and group_variables maps group_as names to variable name lists.

    """
    base_path = Path(config["base_path"])
    datasets_cfg = config.get("data", {}).get("datasets", {})

    if not datasets_cfg:
        msg = "No datasets configured."
        raise ValueError(msg)

    # Resolve paths grouped by group_as.
    group_paths: dict[str, list[Path]] = defaultdict(list)
    group_variables: dict[str, list[str]] = {}

    for dataset_key in datasets_cfg:
        dataset = datasets_cfg[dataset_key]
        name = str(dataset["name"])
        group_as = str(dataset.get("group_as", name))
        zarr_path = (base_path / "data" / "anemoi" / f"{name}.zarr").resolve()

        # Handle ListConfig from Hydra overrides.
        if isinstance(zarr_path, list):
            for p in zarr_path:
                group_paths[group_as].append(Path(str(p)))
        else:
            group_paths[group_as].append(zarr_path)

        variables = dataset.get("variables")
        if variables is not None:
            # Handle ListConfig from Hydra overrides.
            var_list = (
                list(variables)
                if isinstance(variables, (list, ListConfig))
                else [str(variables)]
            )
            group_variables[group_as] = sorted(var_list)

    logger.info(
        "Resolved %d dataset groups for analysis: %s.",
        len(group_paths),
        list(group_paths.keys()),
    )
    for idx, (group_name, paths) in enumerate(group_paths.items(), start=1):
        vars_str = (
            f" (variables: {', '.join(group_variables.get(group_name, []))})"
            if group_variables.get(group_name)
            else ""
        )
        logger.info("%d) %s%s:", idx, group_name, vars_str)
        for path in paths:
            logger.info("   - %s", path)

    return dict(group_paths), group_variables


def build_datasets(
    group_paths: dict[str, list[Path]],
    group_variables: dict[str, list[str]],
    *,
    normalise: bool = True,
) -> dict[str, SingleDataset]:
    """Build ``SingleDataset`` instances for each dataset group.

    Args:
        group_paths: Mapping of group name to zarr paths (from :func:`resolve_datasets`).
        group_variables: Mapping of group name to variable lists (from :func:`resolve_datasets`).
        normalise: Whether to apply the dataset's stored normalisation statistics.

    Returns:
        Mapping of group name to ``SingleDataset`` instance.

    """
    # Import here to avoid pulling SingleDataset into module-level imports.
    from icenet_mp.data_loaders.single_dataset import SingleDataset  # noqa: PLC0415

    datasets: dict[str, SingleDataset] = {}
    for group_name, paths in group_paths.items():
        vars_list = group_variables.get(group_name)
        if vars_list is not None and len(vars_list) > 0:
            logger.info(
                "Loading dataset group %r (%d paths, variables: %s).",
                group_name,
                len(paths),
                vars_list,
            )
            datasets[group_name] = SingleDataset(
                group_name,
                paths,
                normalise=normalise,
                variables=vars_list,
            )
        else:
            logger.warning(
                "Dataset group %r has no variable filter — loading all variables (%d paths). "
                "Consider specifying 'variables' to limit data loaded.",
                group_name,
                len(paths),
            )
            datasets[group_name] = SingleDataset(group_name, paths, normalise=normalise)

    return datasets


def _get_max_samples(config: DictConfig, module_name: str) -> int | None:
    """Resolve ``max_samples`` for a given analysis module.

    Looks up ``module.max_samples`` first (e.g. ``pca.max_samples``), falling back to
    the shared ``vif.max_samples`` key so a single value can cap every strand at once.

    Args:
        config: Hydra-composed config.
        module_name: Analysis module name (e.g. ``"vif"``, ``"pca"``, ``"eof"``).

    Returns:
        The configured max sample count, or ``None`` for no limit.

    """
    mod_cfg = config.get(module_name, {})
    if mod_cfg is None:
        mod_cfg = {}
    return mod_cfg.get("max_samples", config.get("vif", {}).get("max_samples"))


def build_sample_matrix(
    datasets: dict[str, SingleDataset],
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load raw data from all datasets and stack into a sample matrix.

    Each timestep becomes one row; each column is the **spatial mean** of one variable.
    Diagnostics measure multicollinearity / variance *between* variables — aggregating
    spatially avoids creating hundreds of thousands of near-duplicate columns (one per
    grid cell) which would make the computation intractable and meaningless.

    Dates are **intersected** across all datasets (same logic as ``CombinedDataset.dates``)
    so that only timesteps present in every dataset are used — matching how training drops
    incomplete dates.

    Args:
        datasets: Mapping of group name to SingleDataset.
        max_samples: If set, sample evenly across the available common dates.

    Returns:
        Tuple of (sample_matrix, variable_names) where sample_matrix has shape
        (n_samples, n_variables) and variable_names is a flat list of column labels.

    """
    # Intersect dates across all datasets — same logic as CombinedDataset.dates.
    common_dates = sorted(
        set.intersection(*(set(ds.dates) for ds in datasets.values()))
    )

    if not common_dates:
        msg = (
            "No common dates found across all datasets. Analysis requires that "
            "every dataset has data at the same timesteps."
        )
        raise ValueError(msg)

    # Apply max_samples limit by evenly sampling from common dates.
    if max_samples is not None and max_samples < 1:
        msg = f"max_samples must be >= 1 (got {max_samples})."
        raise ValueError(msg)

    if max_samples is not None and max_samples < len(common_dates):
        indices = np.linspace(0, len(common_dates) - 1, max_samples, dtype=int)
        sampled_dates: list[np.datetime64] = [common_dates[i] for i in indices]
        logger.info(
            "Found %d common dates across all datasets; sampling %d evenly.",
            len(common_dates),
            max_samples,
        )
    else:
        sampled_dates = common_dates

    # Build variable name list and collect spatially-aggregated data.
    var_names: list[str] = []
    arrays: list[np.ndarray] = []  # one array per variable, shape (n_timesteps,)

    for group_name, ds in datasets.items():
        # Use get_tchw which handles arbitrary (non-consecutive) date sequences.
        tchw = ds.get_tchw(sampled_dates)  # shape: (T, C, H, W)

        for var_idx, var_name in enumerate(ds.variable_names):
            full_label = f"{group_name}/{var_name}"
            var_names.append(full_label)

            c = tchw.shape[1]
            if var_idx < c:
                # Spatial mean: average across HxW to get one value per timestep.
                spatial_mean = tchw[:, var_idx, :, :].mean(axis=(-2, -1))
                arrays.append(spatial_mean)

    if not arrays:
        msg = "No variables found in any dataset."
        raise ValueError(msg)

    sample_matrix = np.column_stack(arrays)  # shape: (n_timesteps, n_variables)
    return sample_matrix, var_names

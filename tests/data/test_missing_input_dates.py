from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from icenet_mp.data.combined_dataset import CombinedDataset
from icenet_mp.data.common_data_module import CommonDataModule
from icenet_mp.data.single_dataset import SingleDataset


def _datasets_with_missing_input_date(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
) -> tuple[SingleDataset, SingleDataset]:
    """Build a complete target dataset and an input dataset missing one date."""
    target = SingleDataset(name="target", input_files=[mock_dataset])
    sparse = SingleDataset(name="sparse", input_files=[mock_dataset])
    sparse.dates = [dates_as_np[0], *dates_as_np[2:]]
    return target, sparse


def test_missing_input_sentinel_keeps_otherwise_valid_window(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
) -> None:
    """Sentinel filling should retain a window lost by strict intersection."""
    target, sparse = _datasets_with_missing_input_date(mock_dataset, dates_as_np)

    strict = CombinedDataset(
        datasets=[target, sparse],
        target_group_name="target",
        target_variables=["ice_conc"],
        n_history_steps=2,
        n_forecast_steps=1,
    )
    filled = CombinedDataset(
        datasets=[target, sparse],
        target_group_name="target",
        target_variables=["ice_conc"],
        missing_input_value=-1.0,
        n_history_steps=2,
        n_forecast_steps=1,
    )

    assert dates_as_np[0] not in strict.dates
    assert dates_as_np[0] in filled.dates

    batch = filled[filled.dates.index(dates_as_np[0])]
    np.testing.assert_array_equal(
        batch["sparse"][1],
        np.full((3, 2, 2), -1.0, dtype=np.float32),
    )
    assert not np.all(batch["sparse"][0] == -1.0)


def test_missing_target_forecast_date_is_never_filled(
    mock_dataset: Path,
    dates_as_np: tuple[np.datetime64, ...],
) -> None:
    """Forecast targets should remain mandatory even when inputs may be filled."""
    target, sparse = _datasets_with_missing_input_date(mock_dataset, dates_as_np)
    combined = CombinedDataset(
        datasets=[target, sparse],
        target_group_name="target",
        target_variables=["ice_conc"],
        missing_input_value=-1.0,
        n_history_steps=2,
        n_forecast_steps=1,
    )
    combined.target.dates = [date for date in dates_as_np if date != dates_as_np[2]]

    # d0 would forecast d2, so it must be rejected when d2 is missing from the target.
    assert dates_as_np[0] not in combined.dates


def test_missing_input_value_is_read_from_data_config(
    cfg_common_data_module: DictConfig,
) -> None:
    """CommonDataModule should expose the configured input sentinel value."""
    cfg_common_data_module["data"]["missing_input_value"] = -1.0

    data_module = CommonDataModule(cfg_common_data_module)

    assert data_module.missing_input_value == -1.0

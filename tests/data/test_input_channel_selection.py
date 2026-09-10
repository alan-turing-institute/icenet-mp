from pathlib import Path

import pytest
from omegaconf import DictConfig

from icenet_mp.data import CommonDataModule, SingleDataset


def _with_dataset(
    config: DictConfig,
    mock_dataset: Path,
    *,
    variables: list[str] | None = None,
) -> CommonDataModule:
    if variables is not None:
        config["inputs"] = {"group1": {"variables": variables}}
    config["predict"]["target"]["variables"] = ["ice_conc"]
    module = CommonDataModule(config)
    module.__dict__["datasets"] = {
        "group1": SingleDataset(name="group1", input_files=[mock_dataset])
    }
    return module


def test_input_selection_reduces_data_space_channels(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Build encoder-facing DataSpace objects from selected input variables only."""
    module = _with_dataset(
        cfg_common_data_module,
        mock_dataset,
        variables=["ice_conc", "temperature"],
    )

    assert module.input_spaces[0].channels == 2
    assert set(module.input_variable_names["group1"]) == {
        "ice_conc",
        "temperature",
    }


def test_omitted_input_selection_preserves_all_channels(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Keep the existing all-variable behaviour when no input selection is configured."""
    module = _with_dataset(cfg_common_data_module, mock_dataset)

    assert module.input_spaces[0].channels == 3
    assert set(module.input_variable_names["group1"]) == {
        "ice_conc",
        "ice_thickness",
        "temperature",
    }


def test_target_selection_remains_independent_from_input_selection(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Keep predict.target.variables as the output contract after input subsetting."""
    module = _with_dataset(
        cfg_common_data_module,
        mock_dataset,
        variables=["temperature", "ice_conc"],
    )

    assert module.output_space.channels == 1
    assert module.target_variables == ["ice_conc"]
    target_index = module.target_variable_indices[0]
    assert module.input_variable_names["group1"][target_index] == "ice_conc"


def test_dataloader_returns_only_selected_input_channels(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Apply input selection to arrays returned by CombinedDataset and DataLoader."""
    none_period = [{"start": None, "end": None}]
    for split_name in ("predict", "test", "train", "validate"):
        cfg_common_data_module["data"]["split"][split_name] = none_period
    module = _with_dataset(
        cfg_common_data_module,
        mock_dataset,
        variables=["ice_conc", "temperature"],
    )

    batch = next(iter(module.train_dataloader()))

    assert batch["group1"].shape[2] == 2
    assert batch["target"].shape[2] == 1


def test_unknown_input_group_is_rejected(
    cfg_common_data_module: DictConfig,
) -> None:
    """Reject selections for group names that are absent from configured datasets."""
    cfg_common_data_module["inputs"] = {"missing-group": {"variables": ["ice_conc"]}}

    with pytest.raises(ValueError, match="missing-group"):
        CommonDataModule(cfg_common_data_module)


def test_unknown_input_variable_is_rejected(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Reject requested channels that are not present in the selected dataset group."""
    module = _with_dataset(
        cfg_common_data_module,
        mock_dataset,
        variables=["not-a-variable"],
    )

    with pytest.raises(ValueError, match="not-a-variable"):
        _ = module.input_datasets


def test_target_variable_must_remain_available_in_target_input(
    cfg_common_data_module: DictConfig, mock_dataset: Path
) -> None:
    """Fail clearly if input selection removes the variable needed for persistence."""
    module = _with_dataset(
        cfg_common_data_module,
        mock_dataset,
        variables=["temperature"],
    )

    with pytest.raises(ValueError, match="ice_conc"):
        _ = module.target_variable_indices

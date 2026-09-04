import logging
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, SequentialSampler

from icenet_mp.data.common_data_module import CommonDataModule
from icenet_mp.utils import mask_dir

NONE_PERIOD = [{"start": None, "end": None}]


def _build_config(
    base_path: str,
    datasets: dict,
    *,
    input_variables: dict[str, list[str]],
    target_variables: dict[str, list[str]],
    split: dict[str, list[dict[str, str | None]]] | None = None,
) -> DictConfig:
    """Build a minimal CommonDataModule config for tests."""
    return DictConfig(
        {
            "base_path": base_path,
            "data": {
                "datasets": datasets,
                "split": split
                or {
                    "predict": NONE_PERIOD,
                    "test": NONE_PERIOD,
                    "train": NONE_PERIOD,
                    "validate": NONE_PERIOD,
                },
            },
            "variables": {"input": input_variables, "target": target_variables},
            "window": {
                "batch_size": 2,
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


def _single_group_config(
    mock_dataset: Path,
    *,
    input_variables: list[str],
    target_variables: list[str],
    group_as: str = "group1",
) -> DictConfig:
    """Build a config with one dataset group backed by the real `mock_dataset` fixture."""
    return _build_config(
        str(mock_dataset.parent.parent.parent),
        {"ds1": {"name": mock_dataset.stem, "group_as": group_as}},
        input_variables={group_as: input_variables},
        target_variables={group_as: target_variables},
    )


class TestPeriods:
    """Period bounds are stringified while preserving None (YAML null)."""

    def test_null_preserved_as_none(self, cfg_common_data_module: DictConfig) -> None:
        """Python None (YAML null) must not be stringified to 'None'."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.predict_periods == [{"start": None, "end": None}]

    def test_string_values_unchanged(self, cfg_common_data_module: DictConfig) -> None:
        """Date strings must pass through without modification."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.test_periods == [{"start": "2020-01-01", "end": "2020-12-31"}]
        assert dm.val_periods == [{"start": "2020-01-01", "end": "2020-03-31"}]

    def test_mixed_none_and_string_in_same_period(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A period with one None bound and one date string normalises both correctly."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.train_periods == [
            {"start": None, "end": "2019-12-31"},
            {"start": "2018-01-01", "end": None},
        ]


class TestTargetGroupValidation:
    """The target group must exist, be unique, and its variables must be selectable."""

    def test_missing_target_group_raises(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A target group absent from the configured datasets should raise."""
        available_group = next(
            iter(cfg_common_data_module["data"]["datasets"].values())
        )["group_as"]
        cfg_common_data_module["variables"]["target"] = {"missing-target": ["mock_var"]}
        dm = CommonDataModule(cfg_common_data_module)

        with pytest.raises(ValueError, match="missing-target") as exc_info:
            _ = dm.target_group_name

        message = str(exc_info.value)
        assert str(available_group) in message

    def test_multiple_target_groups_raises(self, mock_dataset: Path) -> None:
        """Only one target dataset group is supported; more than one must raise."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={
                "group1": ["ice_conc"],
                "group2": ["ice_conc"],
            },
            target_variables={
                "group1": ["ice_conc"],
                "group2": ["ice_conc"],
            },
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="exactly one target variable group"):
            _ = dm.target_group_name

    def test_target_variable_must_be_input_variable(self, mock_dataset: Path) -> None:
        """A target variable not requested as an input variable for its group raises."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc"],
            target_variables=["ice_conc", "ice_thickness"],
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="ice_thickness") as exc_info:
            _ = dm.target_variable_indices

        message = str(exc_info.value)
        assert "ice_conc" in message

    def test_omitted_input_selection_causes_a_crash(self, mock_dataset: Path) -> None:
        """Omitting the target's own group from `variables.input` crashes."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group2": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(StopIteration):
            _ = dm.target_variables


class TestInputVariableSelection:
    """Requested input variables change what is loaded and validate against the data."""

    def test_unknown_input_group_is_rejected(self, mock_dataset: Path) -> None:
        """An input group not present in the configured datasets should raise."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {"ds1": {"name": mock_dataset.stem, "group_as": "group1"}},
            input_variables={"not-a-group": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="not-a-group") as exc_info:
            _ = dm.variable_names

        assert "group1" in str(exc_info.value)

    def test_unknown_input_variable_is_rejected(self, mock_dataset: Path) -> None:
        """A variable not present in its dataset group should raise."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["not-a-variable"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="not-a-variable") as exc_info:
            _ = dm.variable_names

        message = str(exc_info.value)
        assert "ice_conc" in message

    def test_input_selection_changes_data_space_channels(
        self, mock_dataset: Path
    ) -> None:
        """Requesting a subset of input variables changes the DataSpace.

        The underlying `mock_dataset` fixture has three variables (`ice_conc`,
        `ice_thickness`, `temperature`); requesting only two of them as input should
        produce a DataSpace with 2 channels, not 3.
        """
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.datasets["group1"].variable_names == ["ice_conc", "temperature"]
        [space] = dm.input_spaces
        assert space.channels == 2

    def test_datasets_unfiltered_ignores_variable_selection(
        self, mock_dataset: Path
    ) -> None:
        """`datasets_unfiltered` always exposes every variable, regardless of `variables.input`."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.datasets["group1"].variable_names == ["ice_conc"]
        assert set(dm.datasets_unfiltered["group1"].variable_names) == {
            "ice_conc",
            "ice_thickness",
            "temperature",
        }


class TestDerivedProperties:
    """Properties derived from the resolved datasets: hemisphere, spaces, coordinates."""

    def test_hemisphere_returns_consistent_value(self, mock_dataset: Path) -> None:
        """A single dataset group's hemisphere is returned directly."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)
        assert dm.hemisphere == "south"

    def test_hemisphere_raises_when_groups_disagree(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed hemispheres across dataset groups must raise, not silently pick one."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group1": ["ice_conc"], "group2": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)
        monkeypatch.setattr(dm.datasets["group2"], "hemisphere", "north")

        with pytest.raises(ValueError, match="different hemisphere"):
            _ = dm.hemisphere

    def test_output_space_reflects_target_variables(self, mock_dataset: Path) -> None:
        """The output space is named after, and sized by, the target group/variables."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.output_space.name == "group1"
        assert dm.output_space.channels == 1

    def test_latitudes_and_longitudes_keyed_by_group_name(
        self, mock_dataset: Path
    ) -> None:
        """Coordinates are returned per dataset group, matching the grid size."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        assert set(dm.latitudes) == {"group1"}
        assert set(dm.longitudes) == {"group1"}
        assert len(dm.latitudes["group1"]) == len(dm.longitudes["group1"]) > 0


class TestDataLoaders:
    """Dataloader construction: worker assignment, shuffling, and channel content."""

    def test_assign_workers_updates_dataloaders(self, mock_dataset: Path) -> None:
        """assign_workers propagates to num_workers/persistent_workers/prefetch_factor."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        dm.assign_workers(4)
        loader = dm.train_dataloader()
        assert loader.num_workers == 4
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 1

        dm.assign_workers(0)
        loader = dm.train_dataloader()
        assert loader.num_workers == 0
        assert loader.persistent_workers is False
        assert loader.prefetch_factor is None

    def test_only_train_dataloader_shuffles(self, mock_dataset: Path) -> None:
        """Only train_dataloader should shuffle; predict/test/val must stay sequential."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        assert isinstance(dm.train_dataloader().sampler, RandomSampler)
        assert isinstance(dm.test_dataloader().sampler, SequentialSampler)
        assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
        assert isinstance(dm.predict_dataloader().sampler, SequentialSampler)

    def test_dataloader_returns_only_selected_input_channels(
        self, mock_dataset: Path
    ) -> None:
        """A batch's input tensor has one channel per requested (not available) variable."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        batch = next(iter(dm.train_dataloader()))
        assert batch["group1"].shape[2] == 2  # NTCHW: [batch, time, channels, H, W]
        assert batch["target"].shape[2] == 1


class TestTargetMaskDir:
    """Mask directory resolution for the target group, including multi-dataset groups."""

    def test_derived_from_base_path_and_target_dataset_name(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """The mask dir is built from base_path and the target dataset's name."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.mask_directory == Path("/mock/base/path/data/masks/mock")

    def test_picks_first_dataset_with_an_existing_mask(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With several datasets in the group, use the first that has a mask on disk."""
        cfg = _build_config(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},  # no mask on disk
                "ds2": {"name": "sic_b", "group_as": "sic"},  # has a mask
            },
            input_variables={"sic": ["mock_var"]},
            target_variables={"sic": ["mock_var"]},
        )
        mdir = mask_dir(tmp_path, "sic_b")
        mdir.mkdir(parents=True)
        np.save(mdir / "active_mask.npy", np.ones((4, 4), dtype=np.uint8))

        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        # Picked sic_b (the available one), not sic_a (the first listed).
        assert chosen == mask_dir(tmp_path, "sic_b")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

    def test_falls_back_to_first_when_no_masks_exist(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No dataset has a mask: fall back to the first (old behaviour) and warn."""
        cfg = _build_config(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},
                "ds2": {"name": "sic_b", "group_as": "sic"},
            },
            input_variables={"sic": ["mock_var"]},
            target_variables={"sic": ["mock_var"]},
        )
        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        assert chosen == mask_dir(tmp_path, "sic_a")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

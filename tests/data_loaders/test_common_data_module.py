import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from icenet_mp.data_loaders.common_data_module import CommonDataModule
from icenet_mp.utils import mask_dir


class TestCommonDataModule:
    def test_active_mask_path_derived_from_target_dataset(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """The mask path is built from the root and the SIC target dataset's name."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.active_mask_path == Path(
            "/mock/base/path/data/preprocessing/masks/mock/active_mask.npy"
        )

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

    def test_all_four_split_types_normalised(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """None propagates correctly through every split type."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.predict_periods[0]["start"] is None
        assert dm.predict_periods[0]["end"] is None
        assert dm.test_periods[0]["start"] == "2020-01-01"
        assert dm.train_periods[0]["start"] is None
        assert dm.train_periods[1]["end"] is None
        assert dm.val_periods[0]["end"] == "2020-03-31"


class TestTargetMaskDir:
    """B2: choosing the mask when the target group holds multiple datasets."""

    @staticmethod
    def _cfg(base_path: str, datasets: dict) -> DictConfig:
        none_period = [{"start": None, "end": None}]
        return DictConfig(
            {
                "base_path": base_path,
                "data": {
                    "datasets": datasets,
                    "split": {
                        "batch_size": 2,
                        "predict": none_period,
                        "test": none_period,
                        "train": none_period,
                        "validate": none_period,
                    },
                },
                "predict": {
                    "target": {"group_name": "sic"},
                    "n_forecast_steps": 1,
                    "n_history_steps": 1,
                },
            }
        )

    def test_picks_first_dataset_with_an_existing_mask(self, tmp_path, caplog) -> None:  # noqa: ANN001
        """With several datasets in the group, use the first that has a mask on disk."""
        cfg = self._cfg(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},  # no mask on disk
                "ds2": {"name": "sic_b", "group_as": "sic"},  # has a mask
            },
        )
        mdir = mask_dir(tmp_path, "sic_b")
        mdir.mkdir(parents=True)
        np.save(mdir / "active_mask.npy", np.ones((4, 4), dtype=np.uint8))

        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            active = dm.active_mask_path
        # Picked sic_b (the available one), not sic_a (the first listed).
        assert active == mask_dir(tmp_path, "sic_b") / "active_mask.npy"
        assert dm.land_mask_path == mask_dir(tmp_path, "sic_b") / "land_mask.npy"
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

    def test_falls_back_to_first_when_no_masks_exist(self, tmp_path, caplog) -> None:  # noqa: ANN001
        """No dataset has a mask: fall back to the first (old behaviour) and warn."""
        cfg = self._cfg(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},
                "ds2": {"name": "sic_b", "group_as": "sic"},
            },
        )
        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            active = dm.active_mask_path
        assert active == mask_dir(tmp_path, "sic_a") / "active_mask.npy"
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

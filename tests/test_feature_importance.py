from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from icenet_mp.feature_importance import compute_feature_importance


class TestComputeFeatureImportance:
    def test_returns_one_importance_per_variable(self, mock_dataset: Path) -> None:
        base_path = mock_dataset.parents[2]
        config = DictConfig(
            {
                "base_path": str(base_path),
                "data": {
                    "datasets": {
                        "ds1": {"name": "mock_dataset", "group_as": "group1"},
                    },
                    "split": {
                        "batch_size": 2,
                        "predict": [{"start": None, "end": None}],
                        "test": [{"start": None, "end": None}],
                        "train": [{"start": None, "end": None}],
                        "validate": [{"start": None, "end": None}],
                    },
                },
                "predict": {
                    "target": {"group_name": "group1", "variables": ["ice_conc"]},
                    "n_forecast_steps": 1,
                    "n_history_steps": 1,
                },
            }
        )

        ranked = compute_feature_importance(config, n_estimators=10)

        assert {name for name, _ in ranked} == {
            "group1/ice_conc",
            "group1/ice_thickness",
            "group1/temperature",
        }
        importances = np.array([score for _, score in ranked])
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0)
        # Sorted most important first.
        assert list(importances) == sorted(importances, reverse=True)

"""Integration tests for icenet_mp.cli.pre_feature_analysis._run_all_strands."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from icenet_mp.cli.pre_feature_analysis import _run_all_strands


def _make_mock_cfg() -> tuple[MagicMock, dict[str, Any]]:
    """Create a mock config with standard strand settings."""
    cfg_map: dict[str, Any] = {
        "vif": {"threshold": 5.0},
        "pca": None,
        "eof": None,
        "rf": None,
        "vif.max_samples": None,
        "pca.max_samples": None,
        "eof.max_samples": None,
        "rf.max_samples": None,
    }

    def _get(k: str, d: Any) -> Any:  # noqa: ANN401
        return cfg_map.get(k, d)

    mock = MagicMock()
    mock.get.side_effect = _get
    return mock, cfg_map


class TestRunAllStrands:
    """Integration tests for the full pre-feature-analysis pipeline."""

    def test_all_strands_run_with_multi_variable(self, tmp_path: Path) -> None:
        """With 5+ variables, all strands should produce output."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        mock_config, cfg_map = _make_mock_cfg()
        cfg_map["vif"] = {"threshold": 5.0, "max_samples": 80}
        cfg_map["pca"] = {"max_samples": 60}
        cfg_map["eof"] = {"max_samples": 40}
        cfg_map["rf"] = {"max_samples": 20}

        with (
            patch(
                "icenet_mp.cli.pre_feature_analysis.resolve_datasets"
            ) as mock_resolve,
            patch("icenet_mp.cli.pre_feature_analysis.build_datasets") as mock_build_ds,
            patch(
                "icenet_mp.cli.pre_feature_analysis.build_sample_matrix"
            ) as mock_sample,
        ):
            mock_resolve.return_value = ({"group_a": [Path("/tmp/test.zarr")]}, {})  # noqa: S108
            mock_build_ds.return_value = {"group_a": MagicMock()}
            mock_sample.return_value = matrix, names

            _run_all_strands(mock_config, tmp_path)

        assert mock_sample.call_count == 3
        assert [call.kwargs["max_samples"] for call in mock_sample.call_args_list] == [
            80,
            60,
            40,
        ]

        assert (tmp_path / "vif" / "vif_results.json").exists()
        assert (tmp_path / "pca" / "pca_results.json").exists()
        assert (tmp_path / "eof" / "eof_results.json").exists()
        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()

    def test_single_variable_skips_vif_pca(self, tmp_path: Path) -> None:
        """With only 1 variable, VIF/PCA should skip; correlation still runs."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))
        names = ["only_var"]

        mock_config, _cfg_map = _make_mock_cfg()

        with (
            patch(
                "icenet_mp.cli.pre_feature_analysis.resolve_datasets"
            ) as mock_resolve,
            patch("icenet_mp.cli.pre_feature_analysis.build_datasets") as mock_build_ds,
            patch(
                "icenet_mp.cli.pre_feature_analysis.build_sample_matrix"
            ) as mock_sample,
        ):
            mock_resolve.return_value = ({"group_a": [Path("/tmp/test.zarr")]}, {})  # noqa: S108
            mock_build_ds.return_value = {"group_a": MagicMock()}
            mock_sample.return_value = matrix, names

            _run_all_strands(mock_config, tmp_path)

        assert not (tmp_path / "vif" / "vif_results.json").exists()
        assert not (tmp_path / "pca" / "pca_results.json").exists()
        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()

    def test_two_variables_minimum_for_vif_pca_eof(self, tmp_path: Path) -> None:
        """With exactly 2 variables, VIF/PCA/EOF should succeed."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 2))
        names = ["var_a", "var_b"]

        mock_config, _cfg_map = _make_mock_cfg()

        with (
            patch(
                "icenet_mp.cli.pre_feature_analysis.resolve_datasets"
            ) as mock_resolve,
            patch("icenet_mp.cli.pre_feature_analysis.build_datasets") as mock_build_ds,
            patch(
                "icenet_mp.cli.pre_feature_analysis.build_sample_matrix"
            ) as mock_sample,
        ):
            mock_resolve.return_value = ({"group_a": [Path("/tmp/test.zarr")]}, {})  # noqa: S108
            mock_build_ds.return_value = {"group_a": MagicMock()}
            mock_sample.return_value = matrix, names

            _run_all_strands(mock_config, tmp_path)

        assert (tmp_path / "vif" / "vif_results.json").exists()
        assert (tmp_path / "pca" / "pca_results.json").exists()
        assert (tmp_path / "eof" / "eof_results.json").exists()
        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()

    def test_strand_failure_does_not_stop_others(self, tmp_path: Path) -> None:
        """If one strand fails, others should still run."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        mock_config, _cfg_map = _make_mock_cfg()

        fail_msg = "VIF intentionally failed"

        def failing_vif(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(fail_msg)

        with (
            patch(
                "icenet_mp.cli.pre_feature_analysis.resolve_datasets"
            ) as mock_resolve,
            patch("icenet_mp.cli.pre_feature_analysis.build_datasets") as mock_build_ds,
            patch(
                "icenet_mp.cli.pre_feature_analysis.build_sample_matrix"
            ) as mock_sample,
            patch("icenet_mp.cli.pre_feature_analysis._run_vif_strand", failing_vif),
        ):
            mock_resolve.return_value = ({"group_a": [Path("/tmp/test.zarr")]}, {})  # noqa: S108
            mock_build_ds.return_value = {"group_a": MagicMock()}
            mock_sample.return_value = matrix, names

            _run_all_strands(mock_config, tmp_path)

        assert not (tmp_path / "vif" / "vif_results.json").exists()
        assert (tmp_path / "pca" / "pca_results.json").exists()
        assert (tmp_path / "eof" / "eof_results.json").exists()
        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()

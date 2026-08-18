from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from icenet_mp.ingestion.downloaders import build_downloaders
from icenet_mp.ingestion.postprocessors import (
    StatusFlagMaskGenerator,
    SyntheticMaskGenerator,
)
from icenet_mp.ingestion.preprocessors import IceNetSICPreprocessor

ICENET_SIC_TARGET = "icenet_mp.ingestion.preprocessors.IceNetSICPreprocessor"
STATUS_FLAG_TARGET = "icenet_mp.ingestion.postprocessors.StatusFlagMaskGenerator"
SYNTHETIC_TARGET = "icenet_mp.ingestion.postprocessors.SyntheticMaskGenerator"


def _config(tmp_path: Path, dataset_overrides: dict[str, Any]) -> DictConfig:
    """Build a minimal Hydra-style config for a single dataset named "test"."""
    return OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "data": {
                "datasets": {
                    "test": {
                        "name": "test",
                        "dates": {
                            "start": "2020-01-01",
                            "end": "2020-01-31",
                            "frequency": "24h",
                        },
                        **dataset_overrides,
                    }
                },
            },
        }
    )


def test_missing_pre_and_postprocessors_build_empty_composites(tmp_path: Path) -> None:
    """A dataset with no preprocessors/postprocessors keys builds empty composites."""
    downloader = build_downloaders(_config(tmp_path, {}))[0]
    assert downloader.preprocessor.children == []
    assert downloader.postprocessor.children == []


def test_single_preprocessor_spec_builds_composite_and_interpolates_dates(
    tmp_path: Path,
) -> None:
    """A single preprocessor spec builds a composite of one, with dates interpolated."""
    config = _config(
        tmp_path,
        {
            "preprocessors": {
                "icenet_sic": {
                    "_target_": ICENET_SIC_TARGET,
                    "hemisphere": "north",
                    "dates": "${...dates}",
                }
            }
        },
    )
    (child,) = build_downloaders(config)[0].preprocessor.children
    assert isinstance(child, IceNetSICPreprocessor)
    assert str(child.date_range[0].date()) == "2020-01-01"


def test_single_postprocessor_spec_builds_composite_of_one(tmp_path: Path) -> None:
    """A single postprocessor spec builds a composite of one."""
    config = _config(
        tmp_path,
        {"postprocessors": {"status_flag_masks": {"_target_": STATUS_FLAG_TARGET}}},
    )
    children = build_downloaders(config)[0].postprocessor.children
    assert [type(child) for child in children] == [StatusFlagMaskGenerator]


def test_multiple_preprocessor_specs_build_composite_in_order(tmp_path: Path) -> None:
    """Multiple preprocessor specs build a composite preserving config order."""
    config = _config(
        tmp_path,
        {
            "preprocessors": {
                "north": {
                    "_target_": ICENET_SIC_TARGET,
                    "hemisphere": "north",
                    "dates": "${...dates}",
                },
                "south": {
                    "_target_": ICENET_SIC_TARGET,
                    "hemisphere": "south",
                    "dates": "${...dates}",
                },
            }
        },
    )
    north, south = build_downloaders(config)[0].preprocessor.children

    assert isinstance(north, IceNetSICPreprocessor)
    assert isinstance(south, IceNetSICPreprocessor)
    assert north.is_north is True
    assert south.is_north is False


def test_multiple_postprocessor_specs_build_composite_in_order(tmp_path: Path) -> None:
    """Multiple postprocessor specs build a composite preserving config order."""
    config = _config(
        tmp_path,
        {
            "postprocessors": {
                "status_flag_masks": {"_target_": STATUS_FLAG_TARGET},
                "synthetic_masks": {"_target_": SYNTHETIC_TARGET},
            }
        },
    )
    children = build_downloaders(config)[0].postprocessor.children

    assert [type(child) for child in children] == [
        StatusFlagMaskGenerator,
        SyntheticMaskGenerator,
    ]

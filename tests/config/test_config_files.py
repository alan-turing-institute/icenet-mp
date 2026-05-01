from pathlib import Path

import pytest
import yaml

DATASETS_DIR = Path(__file__).parents[2] / "icenet_mp" / "config" / "data" / "datasets"
YAML_FILES = sorted(DATASETS_DIR.glob("*.yaml"))


class TestConfigFiles:
    """Tests for dataset config files."""

    @pytest.mark.parametrize(
        "config_file", YAML_FILES, ids=[f.name for f in YAML_FILES]
    )
    def test_filename_matches_key_and_name(self, config_file: Path) -> None:
        expected_key = config_file.stem.replace("_", "-")
        with config_file.open() as f:
            data = yaml.safe_load(f)

        assert list(data.keys()) == [expected_key], (
            f"Top-level key {list(data.keys())!r} does not match filename-derived key {expected_key!r}"
        )
        assert data[expected_key]["name"] == expected_key, (
            f"name attribute {data[expected_key]['name']!r} does not match top-level key {expected_key!r}"
        )

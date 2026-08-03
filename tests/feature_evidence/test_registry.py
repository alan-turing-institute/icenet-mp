"""Tests for feature-evidence registry validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from icenet_mp.feature_evidence.registry import load_feature_registry


def _config(entries: list[dict[str, object]]) -> object:
    """Create a minimal registry configuration."""
    return OmegaConf.create({"feature_evidence": {"registry": {"entries": entries}}})


def test_loads_canonical_registry_entry() -> None:
    """A canonical source/variable identifier is retained verbatim."""
    registry = load_feature_registry(
        _config([{"source": "era5", "variable": "2t", "family": "surface_temperature"}])  # type: ignore[arg-type]
    )

    assert registry.groups["era5/2t"].family == "surface_temperature"


def test_rejects_duplicate_identifier() -> None:
    """Registry groups must have one unambiguous canonical identifier."""
    entries = [
        {"source": "era5", "variable": "2t", "family": "surface_temperature"},
        {"source": "era5", "variable": "2t", "family": "duplicate"},
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        load_feature_registry(_config(entries))  # type: ignore[arg-type]


def test_rejects_unavailable_variable() -> None:
    """Selected groups must be present in the resolved source variables."""
    registry = load_feature_registry(
        _config([{"source": "era5", "variable": "2t", "family": "surface_temperature"}])  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="unavailable"):
        registry.validate_available({"era5": ["msl"]})

"""Canonical feature-group registry for the opt-in evidence suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omegaconf import DictConfig


@dataclass(frozen=True)
class FeatureGroup:
    """One analysable physical input parameter."""

    identifier: str
    source: str
    variable: str
    family: str
    included: bool = True
    correlated_alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureRegistry:
    """Validated collection of named feature groups."""

    groups: dict[str, FeatureGroup]

    def select(
        self,
        identifiers: Sequence[str] | None = None,
        *,
        included_only: bool = True,
    ) -> list[FeatureGroup]:
        """Return an explicit subset or the registry's included feature groups."""
        if identifiers is None:
            return [
                group
                for group in self.groups.values()
                if group.included or not included_only
            ]
        missing = sorted(set(identifiers) - self.groups.keys())
        if missing:
            msg = f"Feature registry does not contain: {', '.join(missing)}."
            raise ValueError(msg)
        return [self.groups[identifier] for identifier in identifiers]

    def analysis_groups(self, feature_names: Sequence[str]) -> dict[str, list[int]]:
        """Map all lagged or summarised columns to canonical physical parameters.

        Feature columns must begin with the registry's canonical ``source/variable``
        identifier.  Suffixes such as ``_t-0`` and optional spatial-summary labels
        therefore remain part of one physical-parameter group.
        """
        result: dict[str, list[int]] = {group.identifier: [] for group in self.select()}
        for index, feature_name in enumerate(feature_names):
            matches = [
                identifier
                for identifier in result
                if feature_name == identifier
                or feature_name.startswith(f"{identifier}_")
            ]
            if len(matches) > 1:
                msg = (
                    f"Feature column {feature_name!r} matches multiple registry groups."
                )
                raise ValueError(msg)
            if matches:
                result[matches[0]].append(index)
        return {
            identifier: columns for identifier, columns in result.items() if columns
        }

    def validate_available(
        self, available_variables: Mapping[str, Sequence[str]]
    ) -> None:
        """Require every selected registry variable to be available from its source."""
        missing = [
            group.identifier
            for group in self.select()
            if group.variable not in available_variables.get(group.source, ())
        ]
        if missing:
            msg = f"Registry variables unavailable from resolved datasets: {', '.join(sorted(missing))}."
            raise ValueError(msg)

    def validate_correlated_alternatives(self) -> None:
        """Require correlated alternatives to reference distinct known parameters."""
        invalid = sorted(
            {
                alternative
                for group in self.groups.values()
                for alternative in group.correlated_alternatives
                if alternative not in self.groups or alternative == group.identifier
            }
        )
        if invalid:
            msg = (
                "Registry correlated alternatives must reference distinct registry "
                f"identifiers: {', '.join(invalid)}."
            )
            raise ValueError(msg)


def load_feature_registry(config: DictConfig) -> FeatureRegistry:
    """Load and validate ``feature_evidence.registry.entries`` from Hydra config."""
    entries = config.get("feature_evidence", {}).get("registry", {}).get("entries", [])
    if not entries:
        msg = "Feature evidence requires feature_evidence.registry.entries."
        raise ValueError(msg)

    groups: dict[str, FeatureGroup] = {}
    for entry in entries:
        source = str(entry["source"])
        variable = str(entry["variable"])
        identifier = str(entry.get("identifier", f"{source}/{variable}"))
        if identifier != f"{source}/{variable}":
            msg = f"Feature identifier {identifier!r} must equal {source}/{variable}."
            raise ValueError(msg)
        if identifier in groups:
            msg = f"Duplicate feature registry identifier {identifier!r}."
            raise ValueError(msg)
        groups[identifier] = FeatureGroup(
            identifier=identifier,
            source=source,
            variable=variable,
            family=str(entry["family"]),
            included=bool(entry.get("included", True)),
            correlated_alternatives=tuple(
                str(value) for value in entry.get("correlated_alternatives", [])
            ),
        )
    registry = FeatureRegistry(groups)
    registry.validate_correlated_alternatives()
    return registry

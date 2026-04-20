from collections.abc import Iterable
from datetime import datetime as Datetime
from typing import Any
from typing import override as method_override

from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import Metadata

from .geographic_grid import GeographicGrid


class GeographicMetadata(Metadata):
    def __init__(self, metadata: Metadata, geography: GeographicGrid) -> None:
        self.metadata_ = metadata
        self.geography_ = geography

    @property
    @method_override
    def geography(self) -> Geography:
        return self.geography_

    @method_override
    def __contains__(self, key) -> bool:
        return key in self.metadata_

    @method_override
    def __iter__(self) -> Iterable:
        """Return an iterator over the metadata keys."""
        return iter(self.metadata_)

    @method_override
    def __len__(self) -> int:
        return len(self.metadata_)

    @method_override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.metadata_!r},{self.geography_!r})"

    @method_override
    def _hide_internal_keys(self) -> Metadata:
        return self.metadata_._hide_internal_keys()

    @method_override
    def as_namespace(self, namespace=None) -> dict[str, Any]:
        return self.metadata_.as_namespace(namespace)

    @method_override
    def base_datetime(self) -> Datetime:
        return self.metadata_.base_datetime()

    @method_override
    def data_format(self) -> str:
        return self.metadata_.data_format()

    @method_override
    def datetime(self) -> dict[str, Datetime]:
        return self.metadata_.datetime()

    @method_override
    def describe_keys(self) -> list[str]:
        return self.metadata_.describe_keys()

    @method_override
    def dump(self, **kwargs):
        return self.metadata_.dump(**kwargs)

    @method_override
    def get(
        self,
        key,
        default=None,
        *,
        astype: type | None = None,
        raise_on_missing: bool = False,
    ):
        if raise_on_missing and key not in self.keys():
            raise KeyError(f"Invalid key '{key}'")
        result = self.metadata_.get(key, default)
        if astype is None:
            return result
        try:
            return astype(result)
        except Exception as exc:
            msg = f"Failed to convert metadata key '{key}' to type {astype}: {exc}"
            raise ValueError(msg) from exc

    @method_override
    def index_keys(self) -> list[str]:
        return self.metadata_.index_keys()

    @method_override
    def items(self):
        return self.metadata_.items()

    @method_override
    def keys(self):
        return self.metadata_.keys()

    @method_override
    def ls_keys(self) -> list[str]:
        return self.metadata_.ls_keys()

    @method_override
    def namespaces(self) -> list[str]:
        return self.metadata_.namespaces()

    @method_override
    def override(self, *args, **kwargs):
        return self.metadata_.override(*args, **kwargs)

    @method_override
    def valid_datetime(self) -> Datetime:
        return self.metadata_.valid_datetime()

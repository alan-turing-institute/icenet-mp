from typing import Any, cast

from earthkit.data import Field
from earthkit.data.core.metadata import Metadata

from .geographic_grid import GeographicGrid
from .geographic_metadata import GeographicMetadata


class GeographicField(Field):
    def __init__(self, field: Field, geography: GeographicGrid) -> None:
        self._field = field
        self.geo_metadata = GeographicMetadata(
            cast(Metadata, field.metadata()), geography
        )

    def __repr__(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, repr(self._field))

    @property
    def _metadata(self) -> Metadata:
        return self.geo_metadata

    def _values(self, dtype=None) -> Any:
        return self._field._values(dtype)

    def message(self) -> bytes:
        return self._field.message()

    def clone(self, *, values=None, metadata=None, **kwargs):
        return GeographicField(
            self._field.clone(values=values, metadata=self.metadata, **kwargs),
            self.geo_metadata.geography,
        )

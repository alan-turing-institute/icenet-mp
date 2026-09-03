from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from earthkit.data.core.metadata import Metadata

from icenet_mp.geotools.geographic_metadata import GeographicMetadata
from icenet_mp.geotools.grid_factory import epsg_4326n_builder

NO_ARG_DELEGATING_METHODS = [
    ("base_datetime", datetime(2020, 1, 1)),
    ("data_format", "grib"),
    ("datetime", {"base_time": datetime(2020, 1, 1)}),
    ("describe_keys", ["a", "b"]),
    ("index_keys", ["a", "b"]),
    ("items", [("a", 1)]),
    ("keys", ["a", "b"]),
    ("ls_keys", ["a", "b"]),
    ("namespaces", ["default", "mars"]),
    ("valid_datetime", datetime(2020, 1, 2)),
]


class _MetadataWithoutAsNamespace:
    """A minimal metadata double that deliberately omits as_namespace.

    Every real earthkit Metadata subclass implements as_namespace (it is
    abstract on the base class), so this exercises the local fallback branches
    in GeographicMetadata.as_namespace that a spec'd mock can never reach.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def get(self, key: str, default: object = None, **_kwargs: Any) -> object:
        return self._data.get(key, default)


class TestGeographicMetadata:
    """Unit tests for the GeographicMetadata class."""

    def test_geography_property_returns_constructor_argument(self) -> None:
        """Geography returns the GeographicGrid passed at construction."""
        geography = epsg_4326n_builder("1p0", (2, 2))
        geo_metadata = GeographicMetadata(MagicMock(spec=Metadata), geography)

        assert geo_metadata.geography is geography

    def test_contains_delegates_to_wrapped_metadata(self) -> None:
        """__contains__ delegates the membership check to the wrapped metadata."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.__contains__.return_value = True
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert "some_key" in geo_metadata
        mock_metadata.__contains__.assert_called_once_with("some_key")

    def test_iter_delegates_to_wrapped_metadata(self) -> None:
        """__iter__ delegates iteration to the wrapped metadata."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.__iter__.return_value = iter(["a", "b"])
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert list(geo_metadata) == ["a", "b"]

    def test_len_delegates_to_wrapped_metadata(self) -> None:
        """__len__ delegates to the wrapped metadata."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.__len__.return_value = 3
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert len(geo_metadata) == 3

    def test_repr_includes_wrapped_metadata_and_geography(self) -> None:
        """__repr__ reports both the wrapped metadata and the geography."""
        mock_metadata = MagicMock(spec=Metadata)
        geography = epsg_4326n_builder("1p0", (2, 2))
        geo_metadata = GeographicMetadata(mock_metadata, geography)

        assert (
            repr(geo_metadata) == f"GeographicMetadata({mock_metadata!r},{geography!r})"
        )

    def test_hide_internal_keys_delegates_to_wrapped_metadata(self) -> None:
        """_hide_internal_keys() returns whatever the wrapped metadata produces."""
        mock_metadata = MagicMock(spec=Metadata)
        hidden = MagicMock(spec=Metadata)
        mock_metadata._hide_internal_keys.return_value = hidden
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata._hide_internal_keys() is hidden

    @pytest.mark.parametrize(
        ("method_name", "return_value"),
        NO_ARG_DELEGATING_METHODS,
        ids=[name for name, _ in NO_ARG_DELEGATING_METHODS],
    )
    def test_no_arg_methods_delegate_to_wrapped_metadata(
        self, method_name: str, return_value: object
    ) -> None:
        """Simple no-argument accessors delegate straight to the wrapped metadata."""
        mock_metadata = MagicMock(spec=Metadata)
        setattr(mock_metadata, method_name, MagicMock(return_value=return_value))
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        result = getattr(geo_metadata, method_name)()

        assert result == return_value
        getattr(mock_metadata, method_name).assert_called_once_with()

    def test_dump_forwards_kwargs_to_wrapped_metadata(self) -> None:
        """dump() forwards its keyword arguments and returns the delegate's result."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.dump.return_value = "dump-output"
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.dump(depth=2) == "dump-output"
        mock_metadata.dump.assert_called_once_with(depth=2)

    def test_get_forwards_default_when_not_raising_on_missing(self) -> None:
        """get() passes the default through to the wrapped metadata by default."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.get.return_value = "value"
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.get("key", default="fallback") == "value"
        mock_metadata.get.assert_called_once_with("key", default="fallback")

    def test_get_with_raise_on_missing_does_not_forward_the_flag(self) -> None:
        """Current behaviour: raise_on_missing is consumed locally, not forwarded.

        GeographicMetadata.get() only ever omits `default` from the delegate
        call when raise_on_missing is True; it never passes raise_on_missing
        itself down to the wrapped metadata's get().
        """
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.get.return_value = "value"
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.get("key", raise_on_missing=True) == "value"
        mock_metadata.get.assert_called_once_with("key")

    def test_get_converts_result_with_astype(self) -> None:
        """get(astype=...) converts the delegate's result to the requested type."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.get.return_value = "42"
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.get("key", astype=int) == 42

    def test_get_raises_value_error_when_astype_conversion_fails(self) -> None:
        """get(astype=...) wraps a failed conversion in a ValueError."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.get.return_value = "not-an-int"
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        with pytest.raises(ValueError, match="Failed to convert metadata key 'key'"):
            geo_metadata.get("key", astype=int)

    def test_override_wraps_result_with_same_geography(self) -> None:
        """override() delegates to the wrapped metadata and keeps the geography."""
        mock_metadata = MagicMock(spec=Metadata)
        overridden_metadata = MagicMock(spec=Metadata)
        mock_metadata.override.return_value = overridden_metadata
        geography = epsg_4326n_builder("1p0", (2, 2))
        geo_metadata = GeographicMetadata(mock_metadata, geography)

        result = geo_metadata.override({"key1": 1}, key2=2)

        mock_metadata.override.assert_called_once_with({"key1": 1}, key2=2)
        assert isinstance(result, GeographicMetadata)
        assert result.metadata_ is overridden_metadata
        assert result.geography is geography

    def test_as_namespace_delegates_when_wrapped_metadata_supports_it(self) -> None:
        """as_namespace() delegates directly when the wrapped metadata implements it."""
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.as_namespace.return_value = {"foo": "bar"}
        geo_metadata = GeographicMetadata(
            mock_metadata, epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.as_namespace("mars") == {"foo": "bar"}
        mock_metadata.as_namespace.assert_called_once_with("mars")

    @pytest.mark.parametrize("namespace", [None, "", "default"])
    def test_as_namespace_falls_back_to_all_keys_for_default_namespace(
        self, namespace: str | None
    ) -> None:
        """Without a delegate, the default/empty namespace returns every key/value."""
        fake_metadata = _MetadataWithoutAsNamespace(
            {"key1": "value1", "key2": "value2"}
        )
        geo_metadata = GeographicMetadata(
            cast("Metadata", fake_metadata), epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.as_namespace(namespace) == {
            "key1": "value1",
            "key2": "value2",
        }

    def test_as_namespace_returns_empty_dict_for_mars_without_delegate(self) -> None:
        """Without a delegate, the "mars" namespace falls back to an empty dict."""
        fake_metadata = _MetadataWithoutAsNamespace({"key1": "value1"})
        geo_metadata = GeographicMetadata(
            cast("Metadata", fake_metadata), epsg_4326n_builder("1p0", (2, 2))
        )

        assert geo_metadata.as_namespace("mars") == {}

    def test_as_namespace_raises_for_unsupported_namespace_without_delegate(
        self,
    ) -> None:
        """Without a delegate, an unrecognised namespace raises ValueError."""
        fake_metadata = _MetadataWithoutAsNamespace({"key1": "value1"})
        geo_metadata = GeographicMetadata(
            cast("Metadata", fake_metadata), epsg_4326n_builder("1p0", (2, 2))
        )

        with pytest.raises(ValueError, match="Unsupported namespace 'bogus'"):
            geo_metadata.as_namespace("bogus")

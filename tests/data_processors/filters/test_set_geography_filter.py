from unittest.mock import MagicMock

import pandas as pd
import pytest

from icenet_mp.data_processors.filters import SetGeographyFilter


@pytest.fixture
def mock_geo_filter(monkeypatch: pytest.MonkeyPatch) -> SetGeographyFilter:
    """Patch grid_factory and return a SetGeographyFilter instance."""
    factory = MagicMock()
    factory.create.side_effect = lambda *_args, **_kwargs: MagicMock()
    monkeypatch.setattr(
        "icenet_mp.data_processors.filters.set_geography_filter.grid_factory",
        factory,
    )
    return SetGeographyFilter(crs="epsg:6931", resolution="25km")


class TestSetGeographyFilter:
    def test_forward_raises_type_error_for_non_fieldlist(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """forward() raises TypeError when passed a non-FieldList."""
        with pytest.raises(TypeError, match="FieldList"):
            mock_geo_filter.forward(pd.DataFrame())

    def test_cached_geography_same_shape_returns_same_object(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """The same shape returns the cached geography object."""
        geo1 = mock_geo_filter.cached_geography((432, 432))
        geo2 = mock_geo_filter.cached_geography((432, 432))
        assert geo1 is geo2

    def test_cached_geography_different_shapes_computed_separately(
        self, mock_geo_filter: SetGeographyFilter
    ) -> None:
        """Different shapes yield distinct geography objects."""
        geo_a = mock_geo_filter.cached_geography((432, 432))
        geo_b = mock_geo_filter.cached_geography((100, 100))
        assert geo_a is not geo_b

    @pytest.mark.usefixtures("mock_geo_filter")
    def test_geography_cache_is_per_instance(self) -> None:
        """Two SetGeographyFilter instances have independent geography caches."""
        f1 = SetGeographyFilter(crs="epsg:6931", resolution="25km")
        f2 = SetGeographyFilter(crs="epsg:6931", resolution="25km")
        geo1 = f1.cached_geography((432, 432))
        geo2 = f2.cached_geography((432, 432))
        assert geo1 is not geo2

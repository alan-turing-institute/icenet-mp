import pytest

from icenet_mp.geotools import GeographicGrid, GridFactory
from icenet_mp.geotools.grid_factory import epsg_6931_builder


def test_grid_factory_registers_and_dispatches_builder() -> None:
    """Create a grid through the builder registered for a CRS."""
    factory = GridFactory()
    factory.register_crs("EPSG:6931", epsg_6931_builder)

    grid = factory.create("EPSG:6931", resolution="25p0km", shape=(4, 6))

    assert isinstance(grid, GeographicGrid)
    assert grid.native_crs == "EPSG:6931"
    assert grid.resolution() == "25p0km"
    assert grid.shape() == (6, 4)


def test_grid_factory_rejects_unknown_crs() -> None:
    """Reject requests for CRS builders that have not been registered."""
    with pytest.raises(ValueError, match="No builder registered for CRS: EPSG:9999"):
        GridFactory().create("EPSG:9999", resolution="25p0km", shape=(4, 4))

from unittest.mock import MagicMock

import numpy as np
import pytest
from earthkit.data.utils.bbox import BoundingBox
from pyproj import Transformer

from icenet_mp.geotools.geographic_grid import GeographicGrid
from icenet_mp.geotools.grid_factory import ease2_grid_helper


class TestGeographicGrid:
    """Unit tests for the GeographicGrid class."""

    def test_coordinate_accessors_support_dtype_conversion(self) -> None:
        """Allow callers to request coordinate arrays in a different dtype."""
        grid = GeographicGrid(
            "EPSG:4326", "1.0", np.array([-0.5, 0.5]), np.array([-0.5, 0.5])
        )

        assert grid.x(dtype=np.float32).dtype == np.float32
        assert grid.y(dtype=np.float32).dtype == np.float32
        assert grid.latitudes(dtype=np.float32).dtype == np.float32
        assert grid.longitudes(dtype=np.float32).dtype == np.float32

    def test_latitudes_and_longitudes_pass_through_for_epsg_4326(self) -> None:
        """A native EPSG:4326 grid returns its x/y arrays as lon/lat without transforming."""
        lon_points = np.array([-1.5, -0.5, 0.5, 1.5])
        lat_points = np.array([86.5, 87.5, 88.5, 89.5])
        grid = GeographicGrid("EPSG:4326", "1.0", lon_points, lat_points)

        np.testing.assert_allclose(grid.longitudes(), grid.x())
        np.testing.assert_allclose(grid.latitudes(), grid.y())

    def test_latitudes_and_longitudes_transform_and_cache_for_projected_crs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A projected grid transforms x/y to lat/lon once and caches the result."""
        from_crs_spy = MagicMock(wraps=Transformer.from_crs)
        monkeypatch.setattr(Transformer, "from_crs", from_crs_spy)

        _, h_points, w_points = ease2_grid_helper("500p0km", 2, 2)
        grid = GeographicGrid("EPSG:6931", "500p0km", h_points, w_points)

        expected_lat, expected_lon = Transformer.from_crs(
            "EPSG:6931", "EPSG:4326"
        ).transform(grid.x(), grid.y())

        np.testing.assert_allclose(grid.latitudes(), expected_lat)
        np.testing.assert_allclose(grid.longitudes(), expected_lon)
        grid.latitudes()
        grid.longitudes()

        # One call above to build `expected_*`, one for the grid under test.
        assert from_crs_spy.call_count == 2

    def test_mars_area(self) -> None:
        """mars_area() reports the (north, west, south, east) bounds of the grid."""
        grid = GeographicGrid(
            "EPSG:4326",
            "1.0",
            np.array([-1.5, -0.5, 0.5, 1.5]),
            np.array([86.5, 87.5, 88.5, 89.5]),
        )

        assert grid.mars_area() == (89.5, -1.5, 86.5, 1.5)

    def test_mars_grid(self) -> None:
        """mars_grid() reports the per-step lat/lon spacing implied by the grid extent."""
        grid = GeographicGrid(
            "EPSG:4326",
            "1.0",
            np.array([-1.5, -0.5, 0.5, 1.5]),
            np.array([86.5, 87.5, 88.5, 89.5]),
        )

        lat_step, lon_step = grid.mars_grid()

        assert lat_step == pytest.approx(3.0 / 4)
        assert lon_step == pytest.approx(3.0 / 4)

    def test_bounding_box_matches_mars_area(self) -> None:
        """bounding_box() wraps mars_area() in an earthkit BoundingBox."""
        grid = GeographicGrid(
            "EPSG:4326",
            "1.0",
            np.array([-1.5, -0.5, 0.5, 1.5]),
            np.array([86.5, 87.5, 88.5, 89.5]),
        )

        box = grid.bounding_box()

        assert box == BoundingBox(north=89.5, west=-1.5, south=86.5, east=1.5)

    def test_unimplemented_earthkit_grid_methods_raise(self) -> None:
        """Keep unsupported EarthKit geography methods explicit rather than silent."""
        grid = GeographicGrid(
            "EPSG:4326", "1.0", np.array([-0.5, 0.5]), np.array([-0.5, 0.5])
        )

        for method in (
            grid._unique_grid_id,
            grid.gridspec,
            grid.grid_spec,
            grid.projection,
        ):
            with pytest.raises(NotImplementedError):
                method()

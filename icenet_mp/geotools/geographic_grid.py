import logging
from typing import Any, override

import numpy as np
from earthkit.data.core.geography import Geography
from earthkit.data.utils.bbox import BoundingBox
from numpy.typing import NDArray
from pyproj import Transformer

logger = logging.getLogger(__name__)


class GeographicGrid(Geography):
    def __init__(
        self,
        native_crs: str,
        resolution: str,
        x: np.ndarray[tuple[int]],
        y: np.ndarray[tuple[int]],
    ) -> None:
        self.x_, self.y_ = np.meshgrid(x, y)
        self.resolution_ = resolution
        self.native_crs = native_crs
        self.latitudes_: NDArray[np.float32] | None = None
        self.longitudes_: NDArray[np.float32] | None = None

    # def _load_lat_lon(self) -> tuple[np.ndarray[tuple[int]], np.ndarray[tuple[int]]]:
    def _load_lat_lon(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        latlon_trf = Transformer.from_crs(self.native_crs, "EPSG:4326")
        return latlon_trf.transform(self.x(), self.y())

    @override
    def latitudes(self, dtype: type | None = None) -> NDArray[np.float32]:
        if self.latitudes_ is None:
            self.latitudes_, self.longitudes_ = self._load_lat_lon()
        if dtype is not None:
            return self.latitudes_.astype(dtype)
        return self.latitudes_

    @override
    def longitudes(self, dtype: type | None = None) -> NDArray[np.float32]:
        if self.longitudes_ is None:
            self.latitudes_, self.longitudes_ = self._load_lat_lon()
        if dtype is not None:
            return self.longitudes_.astype(dtype)
        return self.longitudes_

    @override
    def x(self, dtype: type | None = None) -> NDArray[np.float32]:
        if dtype is not None:
            return self.x_.astype(dtype)
        return self.x_

    @override
    def y(self, dtype: type | None = None) -> NDArray[np.float32]:
        if dtype is not None:
            return self.y_.astype(dtype)
        return self.y_

    @override
    def bounding_box(self):
        north, west, south, east = self.mars_area()
        return BoundingBox(north=north, west=west, south=south, east=east)

    @override
    def mars_area(self) -> tuple[float, float, float, float]:
        lats = self.latitudes()
        lons = self.latitudes()
        return (
            float(np.max(lats)),
            float(np.min(lons)),
            float(np.min(lats)),
            float(np.max(lons)),
        )

    @override
    def mars_grid(self) -> tuple[float, float]:
        north, west, south, east = self.mars_area()
        lat_steps, lon_steps = self.shape()
        return (
            (north - south) / lat_steps,
            (east - west) / lon_steps,
        )

    @override
    def resolution(self) -> str:
        return self.resolution_

    @override
    def shape(self) -> tuple[int, ...]:
        return self.x().shape

    @override
    def _unique_grid_id(self) -> None:
        raise NotImplementedError()

    @override
    def gridspec(self) -> None:
        raise NotImplementedError()

    @override
    def grid_spec(self) -> None:
        raise NotImplementedError()

    @override
    def projection(self) -> None:
        raise NotImplementedError()

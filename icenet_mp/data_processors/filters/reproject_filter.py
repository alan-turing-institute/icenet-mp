import logging
from collections.abc import Sequence
from typing import cast

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import (
    WrappedField,
    new_field_from_numpy,
    new_fieldlist_from_list,
)
from anemoi.transform.filter import Filter
from earthkit.data import Field

from icenet_mp.geotools import (
    GeographicField,
    GeographicGrid,
    grid_factory,
    nearest_neighbour_indices,
)

logger = logging.getLogger(__name__)


class ReprojectFilter(Filter):
    """A filter to reproject fields to a new grid."""

    def __init__(self, *, crs: str, resolution: str, shape: tuple[int, int]) -> None:
        self.output_geography: GeographicGrid = grid_factory.create(
            crs, resolution=resolution, shape=shape
        )
        self.mapped_indices_h: np.ndarray[tuple[int, int]] | None = None
        self.mapped_indices_w: np.ndarray[tuple[int, int]] | None = None

    def build_projection(
        self, data: ekd.FieldList
    ) -> tuple[np.ndarray[tuple[int, int]], np.ndarray[tuple[int, int]]]:
        """Build the reprojection mapping from the input data to the output geography."""

        # Get the input grid from the data
        if field := next(field for field in data if isinstance(field, Field)):
            lats, lons = field.grid_points()
            input_latlons = np.stack(
                (
                    np.clip(lats, -90, 90).reshape(field.shape),
                    np.clip(lons, -180, 180).reshape(field.shape),
                ),
                axis=-1,
            )
        else:
            raise ValueError("No latitudes/longitudes were found in the input data.")

        # Get the output grid from the output geography
        output_latlons = np.stack(
            (self.output_geography.latitudes(), self.output_geography.longitudes()),
            axis=-1,
        )

        return nearest_neighbour_indices(input_latlons, output_latlons)

    def forward(self, data: ekd.FieldList | pd.DataFrame) -> ekd.FieldList:
        """Apply the forward regridding transformation.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        if not isinstance(data, ekd.FieldList):
            raise TypeError(f"Expected data to be a FieldList, but got {type(data)}.")

        if self.mapped_indices_h is None or self.mapped_indices_w is None:
            self.mapped_indices_h, self.mapped_indices_w = self.build_projection(data)

        return new_fieldlist_from_list(
            [
                new_field_from_numpy(
                    field.to_numpy()[self.mapped_indices_h, self.mapped_indices_w],
                    template=WrappedField(
                        GeographicField(field, self.output_geography)
                    ),
                )
                for field in cast(Sequence[Field], data)
            ]
        )

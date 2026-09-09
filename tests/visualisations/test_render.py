"""Tests for the minimal panel-plot/panel-video rendering core."""

from io import BytesIO

import pytest
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW
from icenet_mp.visualisations.render import render_panels, render_panels_video


class TestRenderPanels:
    def test_single_panel(self, era5_temperature_2d: ArrayHW) -> None:
        result = render_panels([era5_temperature_2d])

        assert isinstance(result, ImageFile)
        assert result.width > 0
        assert result.height > 0

    def test_three_panels_wider_than_one(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        single = render_panels([era5_temperature_2d])
        triple = render_panels(
            [era5_temperature_2d, osisaf_ice_conc_2d, osisaf_ice_conc_2d]
        )

        assert triple.width > single.width

    def test_titles_applied(self, era5_temperature_2d: ArrayHW) -> None:
        result = render_panels([era5_temperature_2d], panel_titles=["Ground Truth"])

        assert isinstance(result, ImageFile)

    def test_shared_vmin_vmax(self, era5_temperature_2d: ArrayHW) -> None:
        result = render_panels([era5_temperature_2d], vmin=260.0, vmax=290.0)

        assert isinstance(result, ImageFile)

    def test_mismatched_titles_length_raises(
        self, era5_temperature_2d: ArrayHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            render_panels([era5_temperature_2d], panel_titles=["a", "b"])

    def test_per_panel_cmap_and_scale(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        result = render_panels(
            [era5_temperature_2d, osisaf_ice_conc_2d],
            cmap=["RdBu_r", "Blues_r"],
            vmin=[260.0, 0.0],
            vmax=[290.0, 1.0],
        )

        assert isinstance(result, ImageFile)

    def test_suptitle(self, era5_temperature_2d: ArrayHW) -> None:
        result = render_panels([era5_temperature_2d], figure_title="Shown: 2020-01-15")

        assert isinstance(result, ImageFile)


class TestRenderPanelsVideo:
    def test_single_panel(self, era5_temperature_thw: ArrayTHW) -> None:
        result = render_panels_video([era5_temperature_thw])

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_three_panels(self, era5_temperature_thw: ArrayTHW) -> None:
        result = render_panels_video(
            [era5_temperature_thw, era5_temperature_thw, era5_temperature_thw]
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_mismatched_titles_length_raises(
        self, era5_temperature_thw: ArrayTHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            render_panels_video([era5_temperature_thw], panel_titles=["a", "b"])

    def test_fps_is_configurable(self, era5_temperature_thw: ArrayTHW) -> None:
        result = render_panels_video([era5_temperature_thw], fps=4)

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

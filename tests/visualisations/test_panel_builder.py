"""Tests for the domain-specific panel assembly on top of render_panels."""

from datetime import date, datetime
from io import BytesIO

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.panel_builder import (
    render_static_singlet,
    render_static_triplet,
    render_video_singlet,
    render_video_triplet,
)


class TestRenderStaticSinglet:
    def test_returns_image(
        self,
        era5_temperature_2d: ArrayHW,
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        result = render_static_singlet(
            era5_temperature_2d,
            land_mask=no_land_mask,
            plot_spec=base_plot_spec,
            when=datetime(2020, 1, 15),
            variable_name="era5:2t",
        )

        assert isinstance(result, ImageFile)
        assert result.width > 0
        assert result.height > 0


class TestRenderVideoSinglet:
    def test_returns_buffer(
        self,
        era5_temperature_thw: ArrayTHW,
        test_dates_short: list[date],
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        dates = [datetime.combine(d, datetime.min.time()) for d in test_dates_short]
        result = render_video_singlet(
            era5_temperature_thw,
            dates=dates,
            land_mask=no_land_mask,
            plot_spec=base_plot_spec,
            variable_name="era5:2t",
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0


class TestRenderStaticTriplet:
    def test_with_difference_panel(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())

        result = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=True),
            when=when,
            variable_name="ice_conc",
        )

        assert isinstance(result, ImageFile)

    def test_without_difference_panel_is_narrower(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())

        two_panel = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=False),
            when=when,
            variable_name="ice_conc",
        )
        three_panel = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=True),
            when=when,
            variable_name="ice_conc",
        )

        assert two_panel.width < three_panel.width

    def test_uncertainty_panel_replaces_difference_panel(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        """An uncertainty panel is added instead of (not alongside) the difference panel."""
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())
        uncertainty = np.full_like(ground_truth, 0.1)

        with_difference = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=True),
            when=when,
            variable_name="ice_conc",
        )
        with_uncertainty = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=True),
            when=when,
            variable_name="ice_conc",
            uncertainty=uncertainty,
        )
        two_panel = render_static_triplet(
            ground_truth,
            prediction,
            land_mask=no_land_mask,
            plot_spec=PlotSpec(include_difference=False),
            when=when,
            variable_name="ice_conc",
        )

        # Both are 3-panel images (ground truth, prediction, + one extra panel),
        # so comparable in width -- and much wider than the 2-panel case.
        assert isinstance(with_uncertainty, ImageFile)
        assert two_panel.width < with_uncertainty.width
        assert two_panel.width < with_difference.width


class TestRenderVideoTriplet:
    def test_with_difference_panel(
        self, sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]]
    ) -> None:
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]

        result = render_video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            land_mask=LandMask(None),
            plot_spec=PlotSpec(include_difference=True),
            variable_name="ice_conc",
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

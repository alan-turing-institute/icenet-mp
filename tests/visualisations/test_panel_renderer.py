"""Tests for the domain-specific panel assembly on top of render_panels."""

from datetime import date, datetime
from io import BytesIO

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.panel_renderer import PanelRenderer


class TestRenderStaticSinglet:
    def test_returns_image(
        self,
        era5_temperature_2d: ArrayHW,
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        renderer = PanelRenderer(no_land_mask, base_plot_spec)

        result = renderer.static_singlet(
            era5_temperature_2d,
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
        renderer = PanelRenderer(no_land_mask, base_plot_spec)

        result = renderer.video_singlet(
            era5_temperature_thw,
            dates=dates,
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
        renderer = PanelRenderer(no_land_mask, PlotSpec(include_difference=True))

        result = renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
        )

        assert isinstance(result, ImageFile)

    def test_without_difference_panel_is_narrower(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())

        two_panel_renderer = PanelRenderer(
            no_land_mask, PlotSpec(include_difference=False)
        )
        two_panel = two_panel_renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
        )
        three_panel_renderer = PanelRenderer(
            no_land_mask, PlotSpec(include_difference=True)
        )
        three_panel = three_panel_renderer.static_triplet(
            ground_truth,
            prediction,
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

        renderer = PanelRenderer(no_land_mask, PlotSpec(include_difference=True))
        with_difference = renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
        )
        with_uncertainty = renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
            uncertainty=uncertainty,
        )
        two_panel_renderer = PanelRenderer(
            no_land_mask, PlotSpec(include_difference=False)
        )
        two_panel = two_panel_renderer.static_triplet(
            ground_truth,
            prediction,
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
        renderer = PanelRenderer(LandMask(None), PlotSpec(include_difference=True))

        result = renderer.video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            variable_name="ice_conc",
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

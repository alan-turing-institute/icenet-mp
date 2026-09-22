"""Tests for the domain-specific panel assembly on top of MatplotlibRenderer."""

from datetime import date, datetime
from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, ArrayTHW, Metadata, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer
from icenet_mp.visualisations.panel_renderer import PanelRenderer

if TYPE_CHECKING:
    from collections.abc import Callable


class TestMetadata:
    def test_metadata_bound_at_construction_appears_in_footers(
        self, no_land_mask: LandMask
    ) -> None:
        """Metadata passed at construction is used by the renderer's own annotator."""
        renderer = PanelRenderer(no_land_mask, Metadata(model="unet"), PlotSpec())

        assert renderer._annotator.footer_for_static() == "Model: unet"


class TestRenderStaticSinglet:
    def test_returns_image(
        self,
        era5_temperature_2d: ArrayHW,
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        renderer = PanelRenderer(no_land_mask, Metadata(), base_plot_spec)

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
        renderer = PanelRenderer(no_land_mask, Metadata(), base_plot_spec)

        result = renderer.video_singlet(
            era5_temperature_thw,
            dates=dates,
            variable_name="era5:2t",
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_figure_title_changes_per_frame(
        self,
        era5_temperature_thw: ArrayTHW,
        test_dates_short: list[date],
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The figure_title passed to panels_video reflects each frame's own date."""
        dates = [datetime.combine(d, datetime.min.time()) for d in test_dates_short]
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        renderer = PanelRenderer(no_land_mask, Metadata(), base_plot_spec)

        renderer.video_singlet(
            era5_temperature_thw,
            dates=dates,
            variable_name="era5:2t",
        )

        title_for_frame: Callable[[int], str] = fake_render.call_args.kwargs[
            "figure_title"
        ]
        assert callable(title_for_frame)
        titles = [title_for_frame(i) for i in range(len(dates))]
        assert len(set(titles)) == len(dates)
        assert all(dates[i].date().isoformat() in titles[i] for i in range(len(dates)))

    def test_rejects_2d_array(
        self,
        era5_temperature_2d: ArrayHW,
        test_dates_short: list[date],
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        """A 2D array should be rejected deterministically, not fail deep inside rendering."""
        dates = [datetime.combine(d, datetime.min.time()) for d in test_dates_short]
        renderer = PanelRenderer(no_land_mask, Metadata(), base_plot_spec)

        with pytest.raises(InvalidArrayError):
            renderer.video_singlet(
                era5_temperature_2d,  # type: ignore[arg-type]
                dates=dates,
                variable_name="era5:2t",
            )

    def test_rejects_dates_not_matching_frame_count(
        self,
        era5_temperature_thw: ArrayTHW,
        test_dates_short: list[date],
        no_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        dates = [
            datetime.combine(d, datetime.min.time()) for d in test_dates_short[:-1]
        ]
        renderer = PanelRenderer(no_land_mask, Metadata(), base_plot_spec)

        with pytest.raises(InvalidArrayError):
            renderer.video_singlet(
                era5_temperature_thw,
                dates=dates,
                variable_name="era5:2t",
            )


class TestRenderStaticTriplet:
    def test_with_difference_panel(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())
        renderer = PanelRenderer(
            no_land_mask, Metadata(), PlotSpec(include_difference=True)
        )

        result = renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
        )

        assert isinstance(result, ImageFile)

    def test_panel_titles_override_only_the_given_keys(
        self,
        sic_pair_2d: tuple[ArrayHW, ArrayHW, date],
        no_land_mask: LandMask,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller-supplied panel_titles entry replaces its default; other keys fall back.

        Used for the climatology/prediction/difference panel, where the first
        panel holds climatology data rather than the ground truth.
        """
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        renderer = PanelRenderer(no_land_mask, Metadata(), PlotSpec())

        renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            panel_titles={"ground_truth": "Climatology"},
            variable_name="ice_conc",
        )

        panel_titles = fake_render.call_args.kwargs["panel_titles"]
        assert panel_titles[0] == "Climatology"
        assert panel_titles[1] == "Prediction"

    def test_without_difference_panel_is_narrower(
        self, sic_pair_2d: tuple[ArrayHW, ArrayHW, date], no_land_mask: LandMask
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())

        two_panel_renderer = PanelRenderer(
            no_land_mask, Metadata(), PlotSpec(include_difference=False)
        )
        two_panel = two_panel_renderer.static_triplet(
            ground_truth,
            prediction,
            when=when,
            variable_name="ice_conc",
        )
        three_panel_renderer = PanelRenderer(
            no_land_mask, Metadata(), PlotSpec(include_difference=True)
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

        renderer = PanelRenderer(
            no_land_mask, Metadata(), PlotSpec(include_difference=True)
        )
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
            no_land_mask, Metadata(), PlotSpec(include_difference=False)
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

    def test_ice_edge_contours_ground_truth_and_prediction_only(
        self,
        sic_pair_2d: tuple[ArrayHW, ArrayHW, date],
        no_land_mask: LandMask,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """include_ice_edge contours the GT/prediction panels, not the difference panel."""
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            fake_render,
        )
        plot_spec = PlotSpec(include_difference=True, include_ice_edge=True)
        renderer = PanelRenderer(no_land_mask, Metadata(), plot_spec)

        renderer.static_triplet(
            ground_truth, prediction, when=when, variable_name="ice_conc"
        )

        contour_arrays = fake_render.call_args.kwargs["contour_arrays"]
        assert contour_arrays[0] is not None
        assert contour_arrays[1] is not None
        assert contour_arrays[2] is None
        assert (
            fake_render.call_args.kwargs["contour_level"]
            == plot_spec.ice_edge_threshold
        )

    def test_no_contour_when_ice_edge_disabled(
        self,
        sic_pair_2d: tuple[ArrayHW, ArrayHW, date],
        no_land_mask: LandMask,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ground_truth, prediction, raw_when = sic_pair_2d
        when = datetime.combine(raw_when, datetime.min.time())
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            fake_render,
        )
        renderer = PanelRenderer(
            no_land_mask, Metadata(), PlotSpec(include_ice_edge=False)
        )

        renderer.static_triplet(
            ground_truth, prediction, when=when, variable_name="ice_conc"
        )

        assert fake_render.call_args.kwargs["contour_arrays"] is None


class TestRenderVideoTriplet:
    def test_with_difference_panel(
        self, sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]]
    ) -> None:
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]
        renderer = PanelRenderer(
            LandMask(None), Metadata(), PlotSpec(include_difference=True)
        )

        result = renderer.video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            variable_name="ice_conc",
        )

        assert isinstance(result, BytesIO)

    def test_panel_titles_override_only_the_given_keys(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller-supplied panel_titles entry replaces its default; other keys fall back.

        Used for the climatology/prediction/difference panel, where the first
        panel holds climatology data rather than the ground truth.
        """
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        renderer = PanelRenderer(LandMask(None), Metadata(), PlotSpec())

        renderer.video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            panel_titles={"ground_truth": "Climatology"},
            variable_name="ice_conc",
        )

        panel_titles = fake_render.call_args.kwargs["panel_titles"]
        assert panel_titles[0] == "Climatology"
        assert panel_titles[1] == "Prediction"

    def test_figure_title_changes_per_frame(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The figure_title passed to panels_video reflects each frame's own date.

        Regression test: the animation used to be built with a single title
        string computed from the first frame's date, so every frame of the
        rendered video showed the first date even as the panels animated.
        """
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        renderer = PanelRenderer(LandMask(None), Metadata(), PlotSpec())

        renderer.video_triplet(
            ground_truth, prediction, dates=dates, variable_name="ice_conc"
        )

        title_for_frame: Callable[[int], str] = fake_render.call_args.kwargs[
            "figure_title"
        ]
        assert callable(title_for_frame)
        titles = [title_for_frame(i) for i in range(len(dates))]
        assert len(set(titles)) == len(dates)
        assert all(dates[i].date().isoformat() in titles[i] for i in range(len(dates)))

    def test_uncertainty_panel_replaces_difference_panel(
        self, sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]]
    ) -> None:
        """An uncertainty panel is added instead of (not alongside) the difference panel."""
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]
        uncertainty = np.full_like(ground_truth, 0.1)

        renderer = PanelRenderer(
            LandMask(None), Metadata(), PlotSpec(include_difference=True)
        )
        with_uncertainty = renderer.video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            variable_name="ice_conc",
            uncertainty=uncertainty,
        )
        two_panel_renderer = PanelRenderer(
            LandMask(None), Metadata(), PlotSpec(include_difference=False)
        )
        two_panel = two_panel_renderer.video_triplet(
            ground_truth,
            prediction,
            dates=dates,
            variable_name="ice_conc",
        )

        assert isinstance(with_uncertainty, BytesIO)
        assert with_uncertainty.getbuffer().nbytes > 0
        assert two_panel.getbuffer().nbytes != with_uncertainty.getbuffer().nbytes

    def test_ice_edge_contours_ground_truth_and_prediction_only(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """include_ice_edge contours the GT/prediction panels, not the difference panel."""
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates]
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            fake_render,
        )
        plot_spec = PlotSpec(include_difference=True, include_ice_edge=True)
        renderer = PanelRenderer(LandMask(None), Metadata(), plot_spec)

        renderer.video_triplet(
            ground_truth, prediction, dates=dates, variable_name="ice_conc"
        )

        contour_arrays = fake_render.call_args.kwargs["contour_arrays"]
        assert contour_arrays[0] is not None
        assert contour_arrays[1] is not None
        assert contour_arrays[2] is None
        assert (
            fake_render.call_args.kwargs["contour_level"]
            == plot_spec.ice_edge_threshold
        )

    def test_rejects_2d_array(
        self,
        sic_pair_2d: tuple[ArrayHW, ArrayHW, date],
    ) -> None:
        """A 2D array should be rejected deterministically, not fail deep inside rendering."""
        ground_truth, prediction, raw_when = sic_pair_2d
        dates = [datetime.combine(raw_when, datetime.min.time())]
        renderer = PanelRenderer(LandMask(None), Metadata(), PlotSpec())

        with pytest.raises(InvalidArrayError):
            renderer.video_triplet(
                ground_truth,  # type: ignore[arg-type]
                prediction,  # type: ignore[arg-type]
                dates=dates,
                variable_name="ice_conc",
            )

    def test_rejects_dates_not_matching_frame_count(
        self,
        sic_pair_3d_stream: tuple[ArrayTHW, ArrayTHW, list[date]],
    ) -> None:
        ground_truth, prediction, raw_dates = sic_pair_3d_stream
        dates = [datetime.combine(d, datetime.min.time()) for d in raw_dates[:-1]]
        renderer = PanelRenderer(LandMask(None), Metadata(), PlotSpec())

        with pytest.raises(InvalidArrayError):
            renderer.video_triplet(
                ground_truth, prediction, dates=dates, variable_name="ice_conc"
            )

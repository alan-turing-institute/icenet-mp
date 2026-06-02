"""Tests for raw input plotting functionality.

This module tests both static plots and animations of raw input variables,
covering ERA5 weather data, OSISAF sea ice concentration, and various
styling configurations.
"""

import io
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pytest

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import PlotSpec
from icenet_mp.visualisations import DEFAULT_SIC_SPEC, convert
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_video import (
    plot_video_inputs,
    plot_video_prediction,
    plot_video_single_input,
)

# Import test constants from conftest
from .conftest import TEST_DATE, TEST_HEIGHT, TEST_WIDTH


@pytest.fixture
def fake_video_from_animation(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., io.BytesIO]:
    """Monkeypatch video_from_animation so video_maps runs fast in tests."""

    def _fake_save(
        _anim: object,
        *,
        fps: int = 2,  # noqa: ARG001
        video_format: str = "gif",  # noqa: ARG001
    ) -> io.BytesIO:
        # Parameters match real video_from_animation signature but are unused in fake
        return io.BytesIO(b"fake-video-data")

    monkeypatch.setattr(convert, "video_from_animation", _fake_save)
    return _fake_save


# --- Tests for plot_video_prediction
class TestPlotVideoPrediction:
    @pytest.mark.parametrize("video_format", ["gif", "mp4"])
    def test_returns_buffer(
        self,
        sic_pair_3d_stream: tuple[np.ndarray, np.ndarray, Sequence[date]],
        fake_video_from_animation: Callable[..., io.BytesIO],
        video_format: Literal["gif", "mp4"],
    ) -> None:
        """video_maps should produce a dict with a BytesIO video buffer (fast path)."""
        ground_truth, prediction, dates = sic_pair_3d_stream
        spec = replace(
            DEFAULT_SIC_SPEC, include_difference=True, video_format=video_format
        )

        result = plot_video_prediction(
            ground_truth,
            prediction,
            dates=dates,
            land_mask=LandMask(None, "north"),
            plot_spec=spec,
        )

        # Reference the fixture to avoid unused-argument lint error
        assert fake_video_from_animation is not None

        expected_key = f"sea-ice-concentration-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert expected_key in result
        buffer = result[expected_key]
        assert isinstance(buffer, io.BytesIO)
        assert buffer.getvalue()  # not empty


class TestPlotVideoSingleInput:
    def test_single_variable_basic(
        self,
        era5_temperature_thw: np.ndarray,
        test_dates_short: list[date],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test basic video creation for a single variable."""
        video_buffer = plot_video_single_input(
            "era5:2t",
            era5_temperature_thw,
            dates=test_dates_short,
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
        )

        assert isinstance(video_buffer, io.BytesIO)
        video_buffer.seek(0)
        assert len(video_buffer.read()) > 1000  # Reasonable GIF size

    def test_video_with_land_mask(
        self,
        era5_temperature_thw: np.ndarray,
        test_dates_short: list[date],
        mock_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test video creation with land mask."""
        video_buffer = plot_video_single_input(
            "era5:2t",
            era5_temperature_thw,
            dates=test_dates_short,
            land_mask=mock_land_mask,
            plot_spec=base_plot_spec,
        )

        assert isinstance(video_buffer, io.BytesIO)
        video_buffer.seek(0)
        assert len(video_buffer.read()) > 1000

    @pytest.mark.parametrize("video_format", ["gif", "mp4"])
    def test_video_formats(
        self,
        era5_temperature_thw: np.ndarray,
        test_dates_short: list[date],
        base_plot_spec: PlotSpec,
        video_format: Literal["gif", "mp4"],
    ) -> None:
        """Test creating videos in different formats."""
        plot_spec = replace(base_plot_spec, video_format=video_format)
        video_buffer = plot_video_single_input(
            "era5:2t",
            era5_temperature_thw,
            dates=test_dates_short,
            land_mask=LandMask(None, "north"),
            plot_spec=plot_spec,
        )

        assert isinstance(video_buffer, io.BytesIO)
        video_buffer.seek(0)
        content = video_buffer.read()
        assert len(content) > 1000

        # Check file signature
        if video_format == "gif":
            assert content[:6] == b"GIF89a"  # GIF header
        elif video_format == "mp4":
            # MP4 typically has ftyp box early
            assert b"ftyp" in content[:100]

    def test_video_wrong_dimension(
        self,
        test_dates_short: list[date],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test error when video data is not 3D."""
        rng = np.random.default_rng(42)
        wrong_dim_array = rng.random((5, 5)).astype(np.float32)  # 2D instead of 3D

        with pytest.raises(InvalidArrayError, match="Expected 3D"):
            plot_video_single_input(
                "era5:2t",
                wrong_dim_array,
                dates=test_dates_short,
                land_mask=LandMask(None, "north"),
                plot_spec=base_plot_spec,
            )

    def test_video_mismatched_dates(
        self,
        era5_temperature_thw: np.ndarray,
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test error when number of dates doesn't match timesteps."""
        wrong_dates = [
            TEST_DATE,
            TEST_DATE + timedelta(days=1),
        ]  # Only 2 dates for 4 timesteps

        with pytest.raises(
            InvalidArrayError, match="Number of dates.*!= number of timesteps"
        ):
            plot_video_single_input(
                "era5:2t",
                era5_temperature_thw,
                dates=wrong_dates,
                land_mask=LandMask(None, "north"),
                plot_spec=base_plot_spec,
            )


class TestPlotVideoInputs:
    def test_video_multiple_variables(
        self,
        test_dates_short: list[date],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test batch video creation for multiple variables."""
        rng = np.random.default_rng(100)
        n_timesteps = len(test_dates_short)

        # Create data streams for multiple variables
        variables = {
            "era5:2t": rng.uniform(
                270, 280, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)
            ).astype(np.float32),
            "era5:10u": rng.normal(
                0, 5, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)
            ).astype(np.float32),
            "sic-icenet:ice_conc": rng.uniform(
                0, 1, size=(n_timesteps, TEST_HEIGHT, TEST_WIDTH)
            ).astype(np.float32),
        }

        results = plot_video_inputs(
            dates=test_dates_short,
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
            variables=variables,
        )

        assert len(results) == len(variables)
        for name in variables:
            key = f"{name}-{test_dates_short[0].strftime('%Y-%m-%d')}"
            assert key in results
            video_buffer = results[key]
            assert isinstance(video_buffer, io.BytesIO)
            video_buffer.seek(0)
            assert len(video_buffer.read()) > 1000

    def test_full_workflow_video(
        self,
        multi_channel_thw: dict[str, np.ndarray],
        test_dates_short: list[date],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test complete workflow: static plots + videos for multiple variables."""
        # Create videos
        video_results = plot_video_inputs(
            dates=test_dates_short,
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
            variables=multi_channel_thw,
        )

        assert len(video_results) == len(multi_channel_thw)
        assert all(isinstance(video, io.BytesIO) for video in video_results.values())

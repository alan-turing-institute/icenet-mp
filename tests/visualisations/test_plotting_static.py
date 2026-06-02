"""Tests for raw input plotting functionality.

This module tests both static plots and animations of raw input variables,
covering ERA5 weather data, OSISAF sea ice concentration, and various
styling configurations.
"""

import logging
from dataclasses import replace
from datetime import date
from typing import Any, Literal

import numpy as np
import pytest
from PIL.ImageFile import ImageFile

from icenet_mp.types import PlotSpec
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_core import style_for_variable
from icenet_mp.visualisations.plotting_static import (
    plot_static_inputs,
    plot_static_prediction,
)
from icenet_mp.visualisations.range_check import compute_range_check_report

from .conftest import TEST_DATE


class TestPlotStaticPrediction:
    def test_returns_image(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    ) -> None:
        """plot_static_prediction should produce a dict with a PIL image of nonzero size."""
        ground_truth, prediction, date = sic_pair_2d
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)

        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None, "north"),
            plot_spec=spec,
        )

        expected_key = f"sea-ice-concentration-{date.strftime('%Y-%m-%d')}"
        assert expected_key in result
        images = result[expected_key]
        assert len(images) == 1
        image = images[0]
        assert image.width > 0
        assert image.height > 0

    def test_emits_warning_badge(
        self,
        sic_pair_warning_2d: tuple[np.ndarray, np.ndarray, date],
    ) -> None:
        """plot_static_prediction should add a red warning text when range_check report warns.

        We assert by recomputing the range_check report for the same inputs and
        requiring that warnings are non-empty. The function draws the warning text
        directly onto the figure; we avoid brittle image text OCR here.
        """
        ground_truth, prediction, date = sic_pair_warning_2d
        spec = replace(
            DEFAULT_SIC_SPEC, include_difference=True, colourbar_strategy="shared"
        )

        # Range Check report should have warnings under shared [0,1]
        (gt_min, gt_max), _ = (
            (0.0, 1.0),
            (0.0, 1.0),
        )  # shared strategy uses spec range for both
        report = compute_range_check_report(
            ground_truth,
            prediction,
            vmin=gt_min,
            vmax=gt_max,
            outside_warn=spec.outside_warn,
            severe_outside=spec.severe_outside,
            include_shared_range_mismatch_check=spec.include_shared_range_mismatch_check,
        )
        assert report.warnings, (
            "Expected non-empty warnings for constructed out-of-range data"
        )

        # Ensure plot_static_prediction runs without error and returns an image
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None, "north"),
            plot_spec=spec,
        )
        images = result[f"sea-ice-concentration-{date.strftime('%Y-%m-%d')}"]
        assert len(images) == 1, "Expected 1 image"
        assert images[0].width > 0, "Image width should be greater than 0"
        assert images[0].height > 0, "Image height should be greater than 0"

    def test_with_land_mask(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    ) -> None:
        """plot_static_prediction should work with land mask overlay."""
        ground_truth, prediction, date = sic_pair_2d
        height, width = ground_truth.shape

        # Create a simple land mask with land in the centre
        land_mask = LandMask(None, "north")
        mask = np.zeros((height, width), dtype=bool)
        centre_h, centre_w = height // 2, width // 2
        mask[centre_h - 5 : centre_h + 5, centre_w - 5 : centre_w + 5] = True
        land_mask.add_mask(mask)

        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None, "north"),
            plot_spec=spec,
        )
        expected_key = f"sea-ice-concentration-{date.strftime('%Y-%m-%d')}"
        assert expected_key in result
        images = result[expected_key]
        assert len(images) == 1
        image = images[0]
        assert image.width > 0
        assert image.height > 0

    def test_with_invalid_land_mask_shape(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """plot_static_prediction should log a warning for missing land mask shape."""
        ground_truth, prediction, date = sic_pair_2d

        # Create land mask with wrong shape
        land_mask = LandMask(None, "north")
        wrong_shape_mask = np.zeros((10, 10), dtype=bool)
        land_mask.add_mask(wrong_shape_mask)

        with caplog.at_level(logging.WARNING):
            plot_static_prediction(
                ground_truth,
                prediction,
                date=date,
                land_mask=land_mask,
                plot_spec=DEFAULT_SIC_SPEC,
            )
            assert "No land mask available for shape (48, 48)." in caplog.text


# --- Tests for plot_static_inputs ---
class TestPlotStaticInputs:
    def test_basic(
        self,
        era5_temperature_2d: np.ndarray,
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test basic single channel plotting."""
        results = plot_static_inputs(
            {"era5:2t": era5_temperature_2d},
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"era5:2t-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert isinstance(pil_images[0], ImageFile)

    def test_land_mask(
        self,
        era5_temperature_2d: np.ndarray,
        mock_land_mask: LandMask,
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test plotting with land mask applied."""
        results = plot_static_inputs(
            {"era5:2t": era5_temperature_2d},
            land_mask=mock_land_mask,
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"era5:2t-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert isinstance(pil_images[0], ImageFile)

    def test_custom_styles(
        self,
        era5_temperature_2d: np.ndarray,
        base_plot_spec: PlotSpec,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test plotting with custom variable styling."""
        plot_spec = replace(base_plot_spec, per_variable_styles=variable_styles)
        results = plot_static_inputs(
            {"era5:2t": era5_temperature_2d},
            land_mask=LandMask(None, "north"),
            plot_spec=plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"era5:2t-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert isinstance(pil_images[0], ImageFile)

    def test_multiple_channels(
        self,
        multi_channel_hw: dict[str, np.ndarray],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test plotting multiple channels at once."""
        results = plot_static_inputs(
            multi_channel_hw,
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == len(multi_channel_hw)
        for expected_name in multi_channel_hw:
            expected_key = f"{expected_name}-{TEST_DATE.strftime('%Y-%m-%d')}"
            assert expected_key in results
            pil_images = results[expected_key]
            assert isinstance(pil_images[0], ImageFile)

    def test_scientific_notation(
        self,
        era5_humidity_2d: np.ndarray,
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test plotting with scientific notation enabled."""
        styles_with_scientific: dict[str, dict[str, str | float | bool]] = {
            "era5:q_10": {
                "cmap": "viridis",
                "decimals": 2,
                "units": "kg/kg",
                "use_scientific_notation": True,
            },
        }
        plot_spec = replace(base_plot_spec, per_variable_styles=styles_with_scientific)

        results = plot_static_inputs(
            {"era5:q_10": era5_humidity_2d},
            land_mask=LandMask(None, "north"),
            plot_spec=plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"era5:q_10-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert isinstance(pil_images[0], ImageFile)

    @pytest.mark.parametrize("colourbar_location", ["vertical", "horizontal"])
    def test_colourbar_locations(
        self,
        era5_temperature_2d: np.ndarray,
        colourbar_location: Literal["vertical", "horizontal"],
    ) -> None:
        """Test plotting with different colorbar orientations."""
        plot_spec = PlotSpec(
            variable="raw_inputs",
            colourmap="viridis",
            colourbar_location=colourbar_location,
        )

        results = plot_static_inputs(
            {"era5:2t": era5_temperature_2d},
            land_mask=LandMask(None, "north"),
            plot_spec=plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1

    @pytest.mark.parametrize(
        ("var_name", "fixture_name"),
        [
            ("era5:2t", "era5_temperature_2d"),
            ("era5:q_10", "era5_humidity_2d"),
            ("era5:10u", "era5_wind_u_2d"),
            ("osisaf-south:ice_conc", "osisaf_ice_conc_2d"),
        ],
    )
    def test_different_variables(
        self,
        var_name: str,
        fixture_name: str,
        base_plot_spec: PlotSpec,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test plotting different types of variables with appropriate styling."""
        data = request.getfixturevalue(fixture_name)

        results = plot_static_inputs(
            {var_name: data},
            land_mask=LandMask(None, "north"),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"{var_name}-{TEST_DATE.strftime('%Y-%m-%d')}"
        assert isinstance(pil_images[0], ImageFile)

    def test_wrong_dimension(
        self,
        base_plot_spec: PlotSpec,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test error when input array is not 2D."""
        rng = np.random.default_rng(42)
        wrong_dim_array = rng.random((5, 5, 5)).astype(np.float32)  # 3D instead of 2D

        with caplog.at_level(logging.WARNING):
            plot_static_inputs(
                {"era5:2t": wrong_dim_array},
                land_mask=LandMask(None, "north"),
                plot_spec=base_plot_spec,
                when=TEST_DATE,
            )

            assert "Expected 2D" in caplog.text


class TestStyleForVariable:
    def test_exact_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test exact variable name matching in styling."""
        style = style_for_variable("era5:2t", variable_styles)

        assert style.cmap == "RdBu_r"
        assert style.two_slope_centre == 273.15
        assert style.units == "K"
        assert style.decimals == 1

    def test_wildcard_match(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test wildcard pattern matching in styling."""
        # Add wildcard pattern
        styles_with_wildcard = {
            **variable_styles,
            "era5:q_*": {"cmap": "viridis", "decimals": 4},
        }

        style = style_for_variable("era5:q_500", styles_with_wildcard)

        assert style.cmap == "viridis"
        assert style.decimals == 4

    def test_scientific_notation(
        self,
        variable_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Test scientific notation option in styling."""
        # Add style with scientific notation
        styles_with_scientific = {
            **variable_styles,
            "era5:q_10": {
                "cmap": "viridis",
                "decimals": 2,
                "units": "kg/kg",
                "use_scientific_notation": True,
            },
        }

        style = style_for_variable("era5:q_10", styles_with_scientific)

        assert style.cmap == "viridis"
        assert style.decimals == 2
        assert style.units == "kg/kg"
        assert style.use_scientific_notation is True

    def test_no_match(self) -> None:
        """Test default styling when no match found."""
        style = style_for_variable("unknown:variable", {})

        # Should use defaults
        assert style.cmap is None
        assert style.decimals is None

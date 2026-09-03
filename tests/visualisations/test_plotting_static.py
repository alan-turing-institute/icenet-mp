"""Tests for raw input plotting functionality.

This module tests both static plots and animations of raw input variables,
covering ERA5 weather data, OSISAF sea ice concentration, and various
styling configurations.
"""

import logging
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL.ImageFile import ImageFile

from icenet_mp.types import PlotSpec
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_core import style_for_variable
from icenet_mp.visualisations.plotting_static import (
    plot_static_climatology,
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

        variable_name = "test-variable"
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
        )

        expected_key = f"{date.strftime('%Y-%m-%d')}-{variable_name}"
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
        variable_name = "test-variable"
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
        )
        images = result[f"{date.strftime('%Y-%m-%d')}-{variable_name}"]
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
        land_mask = LandMask(None)
        mask = np.zeros((height, width), dtype=bool)
        centre_h, centre_w = height // 2, width // 2
        mask[centre_h - 5 : centre_h + 5, centre_w - 5 : centre_w + 5] = True
        land_mask.add_mask(mask)

        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        variable_name = "test-variable"
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=land_mask,
            plot_spec=spec,
            variable_name=variable_name,
        )
        expected_key = f"{date.strftime('%Y-%m-%d')}-{variable_name}"
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
        """plot_static_prediction should log a debug message for missing land mask shape."""
        ground_truth, prediction, date = sic_pair_2d

        # Create land mask with wrong shape
        land_mask = LandMask(None)
        wrong_shape_mask = np.zeros((10, 10), dtype=bool)
        land_mask.add_mask(wrong_shape_mask)

        with caplog.at_level(logging.DEBUG):
            plot_static_prediction(
                ground_truth,
                prediction,
                date=date,
                land_mask=land_mask,
                plot_spec=DEFAULT_SIC_SPEC,
                variable_name="dummy",
            )
            assert (
                "No land mask associated with this dataset has shape (48, 48)."
                in caplog.text
            )


class TestPlotStaticPredictionClimatology:
    """Tests for the optional climatology map on plot_static_prediction."""

    @pytest.fixture
    def climatology_field(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> np.ndarray:
        """A fixed [H, W] climatology field for the same date as sic_pair_2d."""
        ground_truth = sic_pair_2d[0]
        return np.clip(
            np.random.default_rng(7).random(ground_truth.shape), 0.0, 1.0
        ).astype(np.float32)

    def test_with_climatology_adds_climatology_key(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        climatology_field: np.ndarray,
    ) -> None:
        """A climatology field adds a standalone figure under the climatology key."""
        ground_truth, prediction, date = sic_pair_2d
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        variable_name = "test-variable"
        result = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
            climatology=climatology_field,
        )

        expected_main = f"{date.strftime('%Y-%m-%d')}-{variable_name}"
        expected_climatology = f"{expected_main}-climatology"
        assert set(result) == {expected_main, expected_climatology}
        images = result[expected_climatology]
        assert len(images) == 1
        assert images[0].width > 0
        assert images[0].height > 0

    def test_without_climatology_is_unchanged(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        climatology_field: np.ndarray,
    ) -> None:
        """Without a climatology field the output is exactly as before.

        Omitting the kwarg and passing ``None`` both return only the main key, and
        providing a climatology field leaves the main image byte-identical.
        """
        ground_truth, prediction, date = sic_pair_2d
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        variable_name = "test-variable"

        baseline = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
        )
        explicit_none = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
            climatology=None,
        )
        with_climatology = plot_static_prediction(
            ground_truth,
            prediction,
            date=date,
            land_mask=LandMask(None),
            plot_spec=spec,
            variable_name=variable_name,
            climatology=climatology_field,
        )

        expected_main = f"{date.strftime('%Y-%m-%d')}-{variable_name}"
        assert set(baseline) == {expected_main}
        assert set(explicit_none) == {expected_main}
        # The main plot is unaffected by the climatology argument.
        assert (
            baseline[expected_main][0].tobytes()
            == explicit_none[expected_main][0].tobytes()
        )
        assert (
            with_climatology[expected_main][0].tobytes()
            == baseline[expected_main][0].tobytes()
        )

    def test_plot_static_climatology_non_2d_returns_empty(
        self,
        base_plot_spec: PlotSpec,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """plot_static_climatology skips (and logs) non-2D arrays."""
        rng = np.random.default_rng(42)
        wrong_dim = rng.random((5, 5, 5)).astype(np.float32)

        with caplog.at_level(logging.WARNING):
            result = plot_static_climatology(
                wrong_dim,
                date=TEST_DATE,
                land_mask=LandMask(None),
                plot_spec=base_plot_spec,
                variable_name="dummy",
            )

        assert result == {}
        assert "Expected 2D" in caplog.text

    def test_climatology_title_uses_variable_name(
        self,
        base_plot_spec: PlotSpec,
        climatology_field: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The climatology figure title carries the variable name, not the dict key."""
        captured_titles: list[str] = []

        def _capture_suptitle(fig: Figure, text: str) -> None:  # noqa: ARG001
            captured_titles.append(text)

        monkeypatch.setattr(
            "icenet_mp.visualisations.plotting_static.set_suptitle_with_box",
            _capture_suptitle,
        )
        result = plot_static_climatology(
            climatology_field,
            date=TEST_DATE,
            land_mask=LandMask(None),
            plot_spec=base_plot_spec,
            variable_name="test-variable",
        )

        expected_key = f"{TEST_DATE.strftime('%Y-%m-%d')}-test-variable-climatology"
        assert set(result) == {expected_key}
        assert len(captured_titles) == 1
        assert captured_titles[0].startswith("test-variable")
        assert "climatology" not in captured_titles[0]

    def test_climatology_render_failure_keeps_main_figure(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        climatology_field: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A climatology render failure drops only the climatology panel."""

        def _raise(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            msg = "simulated climatology render failure"
            raise ValueError(msg)

        monkeypatch.setattr(
            "icenet_mp.visualisations.plotting_static.plot_static_climatology",
            _raise,
        )
        ground_truth, prediction, date = sic_pair_2d
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        variable_name = "test-variable"

        with caplog.at_level(logging.WARNING):
            result = plot_static_prediction(
                ground_truth,
                prediction,
                date=date,
                land_mask=LandMask(None),
                plot_spec=spec,
                variable_name=variable_name,
                climatology=climatology_field,
            )

        expected_main = f"{date.strftime('%Y-%m-%d')}-{variable_name}"
        assert set(result) == {expected_main}
        assert result[expected_main][0].width > 0
        assert "climatology" in caplog.text.lower()


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
            land_mask=LandMask(None),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"{TEST_DATE.strftime('%Y-%m-%d')}-era5:2t"
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
        assert name == f"{TEST_DATE.strftime('%Y-%m-%d')}-era5:2t"
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
            land_mask=LandMask(None),
            plot_spec=plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"{TEST_DATE.strftime('%Y-%m-%d')}-era5:2t"
        assert isinstance(pil_images[0], ImageFile)

    def test_multiple_channels(
        self,
        multi_channel_hw: dict[str, np.ndarray],
        base_plot_spec: PlotSpec,
    ) -> None:
        """Test plotting multiple channels at once."""
        results = plot_static_inputs(
            multi_channel_hw,
            land_mask=LandMask(None),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == len(multi_channel_hw)
        for expected_name in multi_channel_hw:
            expected_key = f"{TEST_DATE.strftime('%Y-%m-%d')}-{expected_name}"
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
            land_mask=LandMask(None),
            plot_spec=plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"{TEST_DATE.strftime('%Y-%m-%d')}-era5:q_10"
        assert isinstance(pil_images[0], ImageFile)

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
            land_mask=LandMask(None),
            plot_spec=base_plot_spec,
            when=TEST_DATE,
        )

        assert len(results) == 1
        name, pil_images = next(iter(results.items()))
        assert name == f"{TEST_DATE.strftime('%Y-%m-%d')}-{var_name}"
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
                land_mask=LandMask(None),
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

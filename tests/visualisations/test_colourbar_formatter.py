from dataclasses import replace
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.types import DiffColourmapSpec
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.colourbar_formatter import ColourbarFormatter
from icenet_mp.visualisations.layout_builder import LayoutBuilder


class TestGetCbarLimitsFromMappable:
    """get_cbar_limits_from_mappable falls back through mappable.norm, then defaults."""

    def test_falls_back_to_norm_attributes_without_get_clim(
        self, monkeypatch: pytest.MonkeyPatch, era5_temperature_2d: np.ndarray
    ) -> None:
        fig, ax, cax = LayoutBuilder().build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(era5_temperature_2d, levels=10)
        cbar = plt.colorbar(image, cax=cax)

        class _FakeNorm:
            vmin = 260.0
            vmax = 290.0

        class _FakeMappable:
            norm = _FakeNorm()

        # _FakeMappable deliberately has no get_clim, forcing the AttributeError fallback.
        monkeypatch.setattr(cbar, "mappable", _FakeMappable())

        vmin, vmax = ColourbarFormatter().get_cbar_limits_from_mappable(cbar)
        assert vmin == pytest.approx(260.0)
        assert vmax == pytest.approx(290.0)

        plt.close(fig)

    def test_falls_back_to_default_when_norm_has_no_limits(
        self, monkeypatch: pytest.MonkeyPatch, era5_temperature_2d: np.ndarray
    ) -> None:
        fig, ax, cax = LayoutBuilder().build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(era5_temperature_2d, levels=10)
        cbar = plt.colorbar(image, cax=cax)

        class _FakeMappable:
            norm = None

        monkeypatch.setattr(cbar, "mappable", _FakeMappable())

        vmin, vmax = ColourbarFormatter().get_cbar_limits_from_mappable(cbar)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

        plt.close(fig)


class TestAddColourbars:
    """add_colourbars: dedicated cbar_axes vs. automatic-placement fallback."""

    def test_separate_strategy_uses_dedicated_axes_for_each_panel(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="separate",
            colourbar_location="vertical",
            include_difference=False,
        )
        fig, axs, cbar_axes = LayoutBuilder().build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        ColourbarFormatter().add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            plot_spec=spec,
            cbar_axes=cbar_axes,
        )

        assert cbar_axes["groundtruth"] is not None
        assert cbar_axes["prediction"] is not None
        assert len(cbar_axes["groundtruth"].get_yticks()) == 5
        assert len(cbar_axes["prediction"].get_yticks()) == 5

        plt.close(fig)

    def test_shared_strategy_falls_back_to_automatic_placement_without_cbar_axes(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=False,
        )
        fig, axs, _ = LayoutBuilder().build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        n_axes_before = len(fig.axes)
        ColourbarFormatter().add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            plot_spec=spec,
            cbar_axes=None,
        )
        # The fallback path creates one new automatically-placed colourbar axis.
        assert len(fig.axes) == n_axes_before + 1

        plt.close(fig)


class TestAddColourbarsDifferencePanel:
    """add_colourbars difference-panel branches: signed (TwoSlopeNorm) vs absolute."""

    def test_signed_difference_uses_symmetric_tick_formatting(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=True,
            diff_mode="signed",
        )
        fig, axs, cbar_axes = LayoutBuilder().build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        image_difference = axs[2].contourf(
            prediction - ground_truth, levels=10, cmap="RdBu_r", norm=norm
        )
        diff_colour_scale = DiffColourmapSpec(
            norm=norm, vmin=None, vmax=None, cmap="RdBu_r"
        )

        ColourbarFormatter().add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            image_difference=image_difference,
            plot_spec=spec,
            diff_colour_scale=diff_colour_scale,
            cbar_axes=cbar_axes,
        )

        diff_cax = cbar_axes["difference"]
        assert diff_cax is not None
        ticks = diff_cax.get_yticks()
        assert len(ticks) == 5
        # Symmetric ticks: [vmin, mid, centre, mid, vmax], centre defaults to 0.0.
        assert ticks[2] == pytest.approx(0.0, abs=1e-6)
        assert ticks[0] == pytest.approx(-ticks[-1])

        plt.close(fig)

    def test_absolute_difference_without_cbar_axes_uses_automatic_placement(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=True,
            diff_mode="absolute",
        )
        fig, axs, _ = LayoutBuilder().build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)
        image_difference = axs[2].contourf(
            np.abs(prediction - ground_truth), levels=10, cmap="magma"
        )
        # norm=None routes through the plain Normalize(vmin, vmax) construction branch.
        diff_colour_scale = DiffColourmapSpec(
            norm=None, vmin=0.0, vmax=1.0, cmap="magma"
        )

        n_axes_before = len(fig.axes)
        ColourbarFormatter().add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            image_difference=image_difference,
            plot_spec=spec,
            diff_colour_scale=diff_colour_scale,
            cbar_axes=None,
        )
        # Both the shared GT/prediction fallback and the difference fallback fire,
        # each creating one new automatically-placed colourbar axis.
        assert len(fig.axes) == n_axes_before + 2

        plt.close(fig)


class TestFormatSymmetricTicksScientificNotation:
    """format_symmetric_ticks supports scientific-notation tick labels."""

    def test_use_scientific_notation_sets_exponential_formatter(self) -> None:
        fig, ax, cax = LayoutBuilder().build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(np.random.default_rng(1).random((16, 16)), levels=10)
        cbar = plt.colorbar(image, cax=cax)

        ColourbarFormatter().format_symmetric_ticks(
            cbar, vmin=-1.0, vmax=1.0, is_vertical=True, use_scientific_notation=True
        )

        formatter = cbar.ax.yaxis.get_major_formatter()
        formatted = formatter(0.123456, 0)
        assert "e" in formatted.lower()

        plt.close(fig)

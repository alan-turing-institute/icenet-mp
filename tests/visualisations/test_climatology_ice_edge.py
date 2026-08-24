from datetime import date
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_static import (
    CLIMATOLOGY_ICE_EDGE_COLOUR,
    ICE_EDGE_THRESHOLD,
    PREDICTION_ICE_EDGE_COLOUR,
    _draw_climatology_ice_edge,
    plot_static_prediction,
)


def test_draw_climatology_ice_edge_uses_distinct_contours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw prediction and climatology ice edges with distinct contour styles."""
    fig, ax = plt.subplots()
    contour = MagicMock()
    monkeypatch.setattr(ax, "contour", contour)
    prediction = np.zeros((8, 8), dtype=np.float32)
    climatology = np.ones((8, 8), dtype=np.float32)

    try:
        _draw_climatology_ice_edge(
            ax,
            prediction,
            climatology,
            LandMask(None),
        )
    finally:
        plt.close(fig)

    assert contour.call_count == 2
    prediction_call, climatology_call = contour.call_args_list
    assert prediction_call.kwargs["levels"] == [ICE_EDGE_THRESHOLD]
    assert prediction_call.kwargs["colors"] == [PREDICTION_ICE_EDGE_COLOUR]
    assert climatology_call.kwargs["levels"] == [ICE_EDGE_THRESHOLD]
    assert climatology_call.kwargs["colors"] == [CLIMATOLOGY_ICE_EDGE_COLOUR]


def test_draw_climatology_ice_edge_rejects_shape_mismatch() -> None:
    """Reject climatology arrays that do not match the prediction shape."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(InvalidArrayError, match="different shape"):
            _draw_climatology_ice_edge(
                ax,
                np.zeros((8, 8), dtype=np.float32),
                np.zeros((7, 8), dtype=np.float32),
                LandMask(None),
            )
    finally:
        plt.close(fig)


def test_static_prediction_accepts_climatology_overlay() -> None:
    """Render a static prediction plot when a climatology overlay is supplied."""
    x = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    prediction = np.tile(x, (48, 1))
    ground_truth = np.roll(prediction, 1, axis=1)
    climatology = np.roll(prediction, 4, axis=1)
    when = date(2026, 1, 1)

    result = plot_static_prediction(
        ground_truth,
        prediction,
        date=when,
        land_mask=LandMask(None),
        plot_spec=DEFAULT_SIC_SPEC,
        variable_name="sic",
        climatology=climatology,
    )

    image = result["2026-01-01-sic"][0]
    assert image.width > 0
    assert image.height > 0

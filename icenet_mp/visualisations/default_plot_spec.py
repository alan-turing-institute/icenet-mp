from icenet_mp.types import PlotSpec

#: Default plotting specification for sea ice concentration visualisation
DEFAULT_SIC_SPEC = PlotSpec(
    variable="sea_ice_concentration",
    title_groundtruth="Ground Truth",
    title_prediction="Prediction",
    title_difference="Difference",
    colourmap="viridis",
    diff_mode="signed",
    vmin=0.0,
    vmax=1.0,
    colourbar_location="horizontal",
    colourbar_strategy="shared",
    outside_warn=0.05,
    severe_outside=0.20,
    include_shared_range_mismatch_check=True,
)

# CARRA2 evaluation strategy

CARRA2 should be evaluated as a **high-resolution target for Arctic downscaling**, rather than as a drop-in replacement for ERA5 in the standard IceNet-MP weather inputs.

## Decision

Keep ERA5 as the general-purpose weather input for current forecasting configurations.
CARRA2 is pan-Arctic rather than global, so it cannot replace ERA5 for southern-hemisphere runs and would reduce the weather domain available to northern-hemisphere models. Its principal advantage is its finer spatial resolution.

For that reason, the preferred first use of CARRA2 is to build paired ERA5/CARRA2 samples over their common Arctic domain and assess whether a model can recover useful high-resolution structure from the coarser ERA5 fields.

## Proposed downscaling experiment

A CARRA2 downscaling experiment should:

1. restrict training and evaluation to the common ERA5/CARRA2 spatial domain;
2. select variables that have compatible physical definitions in both datasets;
3. align timestamps explicitly, accounting for their different native temporal resolutions;
4. reproject or resample the ERA5 fields onto the CARRA2 target grid without using future information;
5. train the downscaling model on disjoint temporal training, validation and test periods; and
6. compare the learned model against a non-learned interpolation baseline on the same held-out samples.

The primary comparison should report error at CARRA2 resolution. Spatial-detail metrics or spectral diagnostics should also be considered so that a model is not judged successful merely for reproducing a smooth interpolation.

## Acceptance criteria

CARRA2 should be adopted for the downscaling role only if the experiment shows a consistent improvement over the interpolation baseline on held-out data and the required variable/time/grid alignment can be reproduced by the data pipeline.

This decision does not make CARRA2 a dependency of existing ERA5-based forecasting configurations, and it does not imply support for southern-hemisphere CARRA2 inputs.

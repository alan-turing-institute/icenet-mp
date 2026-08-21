# IceNet-like baseline scope

Issue #296 tracks an IceNet-like baseline for comparing IceNet-MP against the original IceNet modelling choices. The existing `baseline/01_unet.yaml` is a useful UNet baseline, but it should not be described as a reproduction of IceNet because the input features and preprocessing are not yet equivalent.

## What the current pipeline already provides

The full ERA5 descriptors already expose several raw fields corresponding to IceNet inputs:

| IceNet input | Current raw source |
| --- | --- |
| 2 m air temperature (`tas`) | ERA5 `2t` |
| 500 hPa air temperature (`ta500`) | ERA5 `t` at 500 hPa |
| mean sea-level pressure (`psl`) | ERA5 `msl` |
| 250/500 hPa geopotential (`zg250`, `zg500`) | ERA5 `z` at 250/500 hPa |
| 10 hPa zonal wind (`ua10`) | ERA5 `u` at 10 hPa |
| 10 m winds (`uas`, `vas`) | ERA5 `10u`, `10v` |
| sea-ice concentration | OSI SAF `ice_conc` |
| land/active masks | generated from the SIC source status flags |

The ERA5 descriptor also provides `cos_julian_day`, `sin_julian_day`, and `insolation` forcings. These are useful seasonal features, but they are not identical to the original IceNet sine/cosine encoding of the initialisation month.

## Gaps before calling a configuration IceNet-like

A closer reproduction still needs explicit decisions or implementation for:

- anomaly generation for the meteorological variables used as anomalies by IceNet
- sea-surface temperature (`tos`)
- downward and upward surface solar-radiation inputs (`rsds`, `rsus`)
- the linear-trend sea-ice forecast feature
- the original initialisation-month encoding
- IceNet's history/forecast cadence and any architecture-specific differences

These should be treated as explicit compatibility gaps rather than silently substituting raw ERA5 fields for derived IceNet features.

## Recommended baseline progression

1. Keep `baseline/01_unet.yaml` as the current simple UNet control.
2. Add missing raw variables only when they are required for the comparison.
3. Implement derived anomaly/trend features as named preprocessing steps so they can be switched on and off independently.
4. Add a dedicated baseline config only once its inputs are sufficiently close to IceNet to make the name meaningful.
5. Compare the dedicated baseline with persistence and the existing IceNet-MP baselines using the same evaluation periods and metrics.

This staged approach keeps the existing baselines unchanged while making the remaining work for #296 explicit and reviewable.

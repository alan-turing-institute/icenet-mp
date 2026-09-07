# Submit forecasts to SIPN

This guide records the information needed for a first, manual submission of an IceNet-MP forecast to the Sea Ice Prediction Network (SIPN) or SIPN South.

Submission protocols change between seasons. Before generating a forecast, check the current call and treat the requirements below as a reproducible checklist rather than a replacement for the organisers' instructions.

## Choose the forecast run

Use the best current model agreed by the project and record enough information to reproduce the submitted forecast:

- checkpoint path or identifier;
- IceNet-MP git commit;
- configuration used for inference;
- forecast initialisation date and input-data cutoff;
- ensemble method and number of members, if applicable;
- native grid, coordinate reference information, and land/ocean mask;
- units and missing-value convention for each submitted field.

Keep the unmodified model output as well as the final submission files.

## Arctic Sea Ice Outlook

The current [2026 Sea Ice Outlook call](https://www.seaiceprediction.org/sea-ice-outlook/august-2026-call-for-contributions) requests September monthly mean sea-ice extent and optionally full spatial forecast fields. The remaining September 2026 contribution deadline is **14 September 2026**.

### Sea-ice extent

For consistency with the NSIDC Sea Ice Index definition used by the call:

1. For each day, sum the areas of grid cells whose sea-ice concentration exceeds 15%.
2. Average those daily extent values over September to obtain the September monthly mean extent.

The call accepts pan-Arctic and pan-Antarctic extent forecasts and an Alaska regional forecast covering the Bering, Chukchi, and Beaufort seas. The Alaska request references the NSIDC 25 km polar-stereographic regional mask; if a different regional definition is used, document it in the submission.

### Full spatial fields

The call requests the following full-field diagnostics:

- Sea Ice Probability (SIP), the fraction of ensemble members with September sea-ice concentration above 15%;
- Ice-Free Date (IFD), the first date concentration falls below 15%, with an optional 80% threshold;
- Ice Advance Date (IAD) for August and September, the first freeze-up date concentration rises above 15%, with an optional 80% threshold;
- spatial initial-condition fields, particularly sea-ice concentration and sea-ice thickness, with the initialisation date recorded.

Daily sea-ice concentration fields can be submitted instead of pre-computed SIP, IFD, and IAD fields; the organisers state that they can derive those metrics from daily concentration.

The public 2026 call does not prescribe a common grid or a complete NetCDF schema for full-field uploads. Preserve the model's geospatial information and confirm the current file-format and forecast-horizon expectations with the SIPN team before uploading. Existing contributors upload full fields to the SIPN data server; new contributors should request access using the contact in the current call.

## SIPN South

The latest published [SIPN South call](https://fmassonn.github.io/sipn-south.github.io/call-contributions.html) is for the 2025-2026 season. Use it as a planning baseline until the 2026-2027 call is published, then update dates and filenames to match the new call.

The 2025-2026 protocol requests forecasts for **90 daily timesteps from 1 December 2025 through 28 February 2026**. The diagnostics most directly relevant to IceNet-MP are listed below in priority order.

### Circumpolar sea-ice area

Submit one text file containing one row of 90 comma-separated daily Antarctic sea-ice area values.

- Units are 10^6 km².
- Values use four decimal places, including trailing zeroes.
- The 2025-2026 filename pattern is `<group-name>_<forecast-id>_20251201-20260228_total-area.txt`.
- Ensemble forecasts use one file per member and increment the three-digit forecast ID.

### Regional sea-ice area

Submit one text file containing 36 rows, each with 90 comma-separated daily values using the same units and precision as the circumpolar file.

Each row represents a 10° longitude bin, from `0° <= longitude < 10°` through `350° <= longitude < 360°`.

The 2025-2026 filename pattern is `<group-name>_<forecast-id>_20251201-20260228_regional-area.txt`.

### Sea-ice concentration

Submit one NetCDF file with 90 daily timesteps. The protocol requires CMIP6-style conventions and the following variables:

| Variable | Meaning | Units |
| --- | --- | --- |
| `siconc` | Fraction of each grid cell covered by sea ice | % |
| `longitude` | Grid-cell longitude | degrees east |
| `latitude` | Grid-cell latitude | degrees north |
| `sftof` | Fraction of each grid cell covered by ocean | % |
| `areacello` | Grid-cell area | m² |

The 2025-2026 filename pattern is `<group-name>_<forecast-id>_20251201-20260228_concentration.nc`.

The protocol does not require a single common spatial grid for concentration fields, but it does require longitude, latitude, ocean fraction, and cell area to accompany the forecast. Do not discard those fields during conversion from IceNet-MP output.

For ensembles, keep one submission file per member and increment the forecast ID.

## Validate the files

Before submission, check all of the following:

- the forecast uses the intended checkpoint and configuration;
- the first and last forecast dates match the current call;
- the number of daily timesteps is correct;
- sea-ice concentration units match the protocol, especially `%` for SIPN South `siconc`;
- longitude, latitude, ocean fraction, and cell area align with the forecast grid;
- land and missing values are handled consistently;
- scalar area or extent diagnostics can be reproduced from the submitted spatial fields;
- every ensemble member has a unique forecast identifier;
- filenames use the current season's required dates and naming convention.

Keep a small submission record containing the checkpoint identifier, git commit, configuration, initialisation date, input-data cutoff, generated filenames, and submission date.

## Submit manually

For the first implementation, keep submission manual:

1. Generate the required forecast from the selected current model.
2. Convert the output into the current call's required scalar and/or spatial formats.
3. Run the validation checklist above.
4. Archive the exact files being submitted in shared project storage.
5. Submit through the route specified by the current call.
6. Record the submission date and the forecast identifiers used.

For Arctic SIO full fields, use the SIPN data-server process described in the current call. For SIPN South, the latest protocol asks contributors to place the files in a URL-accessible archive, provide that URL through the submission form, and notify the organiser after submission.

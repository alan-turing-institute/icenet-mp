# Run VIF analysis for multicollinearity

Variance Inflation Factor (VIF) analysis identifies linear relationships between input variables in your model configuration. High VIF scores indicate that a variable is redundant with others, which can degrade training stability and interpretability.

This guide walks through running VIF on your dataset to check for multicollinearity before training.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md) before continuing.

You will need real data available at `base_path` (typically `/Volumes/Storage/ClimateData`). The sample datasets used by default in the config are small and may not be available on all machines.

## How it works

1. **Data loading**: Variables are loaded from zarr-backed Anemoi datasets via `SingleDataset`. Dates are intersected across all datasets (matching how training drops incomplete dates).
2. **Spatial aggregation**: Each variable's spatial grid is averaged to a single value per timestep. VIF measures multicollinearity *between* variables, not within the spatial structure of one variable.
3. **VIF computation**: For each variable, an OLS regression against all other variables is fitted and the variance inflation factor computed using `statsmodels`.

## Running VIF analysis

### Using a single dataset (quick check)

```bash
uv run imp vif --config-name baseline/01_unet.yaml \
  'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
  ++vif.max_samples=1000
```

This loads all variables from the SIC dataset and computes VIF for each one. Use `++vif.max_samples` to limit the number of timesteps loaded (useful for large datasets).

### Using multiple datasets (full analysis)

The baseline configs reference three dataset groups: SIC, ERA5 weather, and ARGO float data. To analyse multicollinearity across all input variables:

```bash
uv run imp vif --config-name baseline/01_unet.yaml \
  ++base_path=/Volumes/Storage/ClimateData \
  'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2, full_weathernorth_era5_25p0km_1999_2025_24h_v3, full_floatnorth_argo_25p0km_1999_2025_24h_v2]' \
  ++vif.max_samples=1000
```

**Important**: You must override `data/datasets` to list all three groups. Passing a single dataset name replaces the entire list rather than adding to it.

### Using specific variables only

To limit analysis to particular variables (faster, less output):

```bash
uv run imp vif --config-name baseline/01_unet.yaml \
  ++base_path=/Volumes/Storage/ClimateData \
  'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
  '+data.datasets.full-sicnorth-ssmis-25p0km-1979-2024-24h-v2.variables=[ice_conc, TEMP, U10]' \
  ++vif.max_samples=1000
```

### Custom threshold and output directory

```bash
uv run imp vif --config-name baseline/01_unet.yaml \
  ++base_path=/Volumes/Storage/ClimateData \
  'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
  ++vif.threshold=10.0 \
  --output-dir outputs/my_vif_run
```

## Interpreting results

| VIF score | Interpretation |
|-----------|---------------|
| ~1.0 | No multicollinearity — variable is independent of others |
| 1–5 | Low to moderate correlation; generally acceptable |
| >5 | High multicollinearity (default threshold); consider removing or combining variables |
| >10 | Very high multicollinearity; strong candidate for removal |

### Example output

```
VIF Analysis Results (threshold=5.0, samples=1000)
----------------------------------------------------------------------
Variable                                           VIF
----------------------------------------------------------------------
sic-ssmis/total_standard_uncertainty          984877.85 ***
sic-ssmis/algorithm_standard_uncertainty      984448.73 ***
sic-ssmis/smearing_standard_uncertainty          41.73 ***
sic-ssmis/status_flag                            16.66 ***
sic-ssmis/ice_conc                               14.40 ***
sic-ssmis/raw_ice_conc_values                     1.01
----------------------------------------------------------------------
Total: 6 variables, 5 above threshold (5.0)
```

In this example:
- `total_standard_uncertainty` and `algorithm_standard_uncertainty` have extremely high VIF (~984k), indicating near-perfect linear dependence with other variables. These are derived uncertainty metrics that are mathematically related to the base ice concentration values.
- `raw_ice_conc_values` has VIF ≈ 1.0, meaning it is independent of all other variables — this is expected as it's the raw measurement before any processing.

## Output files

Results are written to the output directory (default: `outputs/vif/`):

| File | Description |
|------|-------------|
| `vif_results.json` | Machine-readable JSON with scores, variable names, threshold, and sample count |
| `vif_report.txt` | Human-readable table matching stdout output |

## Design notes

- VIF is designed as a pre-filter for SHAP-based feature importance (future work) to remove redundant variables before expensive analysis.
- Spatial aggregation (mean across grid cells) before VIF computation follows standard climate science practice ([Diaz-Nieto & Wilby, 2005](https://doi.org/10.1002/joc.1161)).
- The analysis bypasses `CommonDataModule` to avoid requiring a prediction target — only datasets and their variables are needed.

# Sea-ice concentration data selection

IceNet-MP currently uses a continuous 25 km OSI SAF sea-ice concentration record assembled from the reprocessed products that cover the required historical period. The northern and southern full dataset descriptors use the same product transitions for their respective hemispheres.

## Current 25 km record

| Period | Product | Sensor family |
| --- | --- | --- |
| 1978-2002 | OSI-450-a1 | SMMR, SSM/I, SSMIS |
| 2003 to 2011-10-04 | OSI-458 | AMSR-E |
| 2011-10-05 to 2012-07-23 | OSI-450-a1 | SSMIS |
| 2012-07-24 to 2020 | OSI-458 | AMSR-2 |
| 2021-2025 | OSI-438 | AMSR2 |

The descriptors concatenate these sources into one daily `ice_conc` series and keep the product transitions inside the ingestion configuration rather than exposing them to the model as separate input groups.

For the current project baseline this 25 km composition has two practical advantages:

- it provides the long historical record needed for training and holdout evaluation
- it matches the EASE2 25 km grid used by the rest of the current sea-ice pipeline, avoiding an additional resolution change solely for the target field

## Other OSI SAF products

The 10 km SSMIS and AMSR2 products remain useful candidates for higher-resolution experiments, but adopting them as the default target is a separate modelling decision. Their shorter temporal coverage changes the amount of training data available, and a 10 km target would also require an explicit choice about the model/output grid and how other inputs are aligned to it.

They should therefore be evaluated as alternative datasets rather than silently substituted into the existing baseline.

## Configuration locations

The current full-resolution dataset descriptors are:

- `icenet_mp/config/data/datasets/full_sicnorth_osisaf_25p0km_1979_2025_24h_v1.yaml`
- `icenet_mp/config/data/datasets/full_sicsouth_osisaf_25p0km_1979_2025_24h_v1.yaml`

Sample configurations use the same 25 km OSI SAF family over their shorter sample periods.

This page records the dataset choice already represented by the current configuration. Changing the default spatial resolution or product family should be accompanied by a controlled comparison so that any accuracy gain can be separated from the effect of reduced training coverage or changed preprocessing.

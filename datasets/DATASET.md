# Datasets

This file describes the external datasets used in this study and where to download them. The repository does not redistribute raw data files; researchers should download them directly from the official sources listed below.

## 1. GPM IMERG Daily Precipitation

| Item | Detail |
|---|---|
| Product | GPM IMERG Final Run V07, daily late run |
| Variables | Precipitation rate (mm/hr), integrated to daily totals |
| Spatial range | 25°N–35°N, 90°E–110°E (upper Yangtze basin) |
| Temporal range | 2007-05-01 to 2024-10-31 (flood seasons, 1,104 daily files) |
| Format | NetCDF-4 (.nc4), one file per day |
| Total size | ~34.5 GB |
| Download | [https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMRDL.07/](https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMRDL.07/) |
| Access | Free; requires a free NASA Earthdata account |

File naming convention: `imerg_final_daily_YYYYMMDD.nc4`

The five forcing windows are rectangular bounding boxes centered on each reservoir sub-basin. They are not DEM-delineated catchments. Cosine-latitude weighting is applied to convert gridded precipitation to area-averaged daily means.

## 2. CDR² — China Daily River Discharge Records

| Item | Detail |
|---|---|
| Product | CDR² v1.0 (satellite-extended daily river discharge) |
| Variables | Daily discharge (m³/s) for 310 gauges across China |
| Temporal range | 1990–2024 |
| Used stations | 6 gauges along the Jinsha River–Three Gorges cascade |
| Format | CSV (extracted from the full dataset zip) |
| Total size | ~2.8 MB (zip) |
| Download | [https://zenodo.org/records/10886848](https://zenodo.org/records/10886848) |
| Access | Free; Zenodo open access |
| Reference | Lin et al. (2023). *China Daily River Discharge Records (CDR²)*. Zenodo. |

CDR² is used as a constrained satellite-derived reference for runoff-model calibration. It is not used as observed inflow input. Only dates where both GPM precipitation and CDR² discharge are available are used for calibration (25–37 matched dates per forcing window).

## 3. CAMELS-CN — Catchment Attributes and Meteorological Data for China

| Item | Detail |
|---|---|
| Product | CAMELS-CN v1.0 |
| Variables | Basin boundaries (shapefiles), climate attributes, land use |
| Used for | Watershed boundary reference and basin attribute context |
| Format | Shapefile (.shp) + CSV |
| Total size | ~50 MB (boundaries subset) |
| Download | [https://zenodo.org/records/3950679](https://zenodo.org/records/3950679) |
| Access | Free; Zenodo open access |
| Reference | Hao et al. (2021). *CAMELS-CN: Catchment Attributes and Meteorological Data for China*. Zenodo. |

## 4. Reservoir engineering parameters

The five-reservoir cascade parameters (storage capacity, release limits, ramp rates, flood-control levels) are compiled from publicly available sources: reservoir design reports, the Three Gorges Corporation website, and the Ministry of Water Resources. See `datasets/metadata/reservoir_parameters.md` for the compiled parameter table and source citations.

## Data citation

All datasets are publicly available and should be cited according to their respective licenses. This repository does not redistribute raw data; researchers should download data directly from the official sources listed above.

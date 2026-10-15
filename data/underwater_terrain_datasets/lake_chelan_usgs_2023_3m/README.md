# Lake Chelan USGS 2023 measured acoustic map

This directory contains the complete public USGS Lake Chelan survey released in
2026 under DOI [10.5066/P1GE3YDR](https://doi.org/10.5066/P1GE3YDR). The source
survey was acquired from 2023-06-04 through 2023-07-03. It is retained as source
data and has not yet replaced the active OceanSim map.

## Why this dataset is useful

- The lake is about 80 km long, so one continuous route can exercise the current
  72--73 km long-range localization test without repeating a small map.
- Bathymetry and acoustic backscatter were measured during the same NORBIT
  WINGHEAD i67 multibeam campaign and are supplied as five corresponding tiles.
- The five bathymetry tiles are 3 m GeoTIFF grids in NAD83(2011) / UTM zone 10N,
  with elevations referenced to NAVD88.
- The downloaded bathymetry contains 13,774,229 valid cells (about 123.97 km2 at
  nominal cell area) and spans -148.543 m to 335.040 m, a 483.583 m elevation
  range.
- The downloaded backscatter contains 13,880,750 valid cells (about 124.93 km2
  at nominal cell area).
- USGS reports about 0.1 m post-processed horizontal positioning accuracy, about
  0.2 m vertical positioning accuracy, and about 0.02 degree roll/pitch accuracy.
  These are system-level positioning figures; final raster accuracy may be lower
  after all acquisition and processing uncertainties are propagated.

The acquisition used a NORBIT WINGHEAD i67 curved-array broadband multibeam
sonar, an Applanix POS-MV GNSS/INS, measured sound-velocity profiles, and CARIS
processing. This is measured lake-floor data, not a synthetic or inferred DEM.

## Directory layout

- `source_archives/`: the ten original USGS ZIP downloads (bathymetry and
  backscatter tiles A--E).
- `raw/`: extracted GeoTIFF/world files, official metadata, and official preview
  images.
- `source_md5sums.txt`: published MD5 values used to verify every source ZIP.

## Scientific cautions

- Raster resolution (3 m) is not the same as absolute survey accuracy.
- Use `-3.402823e38` as bathymetry NoData and `+3.402823466e38` as backscatter
  NoData; do not interpret either sentinel as a physical measurement.
- Bathymetry and backscatter tile footprints are close but not pixel-identical,
  so they must be registered onto one grid before use by OceanSim.
- Backscatter ranges and offsets differ substantially among tiles. Preserve the
  measured spatial texture, but cross-normalize overlapping tiles before using
  intensity in a global matcher. The values must not be treated as a single
  calibrated absolute target-strength scale without that step.
- This product is multibeam acoustic backscatter. It is useful for the project's
  intensity channel, but it is not an independent towfish side-scan survey.
- The data are public-domain USGS research data and are not for navigation.

Official landing page: <https://www.usgs.gov/data/multibeam-bathymetry-data-and-gridded-surfaces-lake-chelan-washington-collected-during-usgs>

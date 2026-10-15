# Jenkinson Lake USGS July 2023 measured acoustic map

This directory contains public USGS survey products released under DOI
[10.5066/P13BSSDY](https://doi.org/10.5066/P13BSSDY). It is retained as a
high-resolution benchmark and has not yet replaced the active OceanSim map.

## Included products

- Complete-lake 1 m bathymetry: 2,395,198 valid cells (about 2.395 km2), with
  NAVD88 elevations from 1010.065 m to 1058.423 m.
- Complete-lake 1 m acoustic backscatter: 2,417,590 valid cells.
- Upper-basin 0.5 m bathymetry: 1,453,937 valid cells (about 0.363 km2).
- Official USGS text metadata and the three original ZIP archives.

The survey used a 234 kHz SWATHplus-M interferometric side-scan/bathymetric
sonar, an Applanix POS-MV GNSS/INS, measured sound-velocity profiles, BathySwath,
and SonarWiz. USGS reports about 0.1 m post-processed horizontal positioning
accuracy, about 0.2 m vertical positioning accuracy, about 0.02 degree roll and
pitch accuracy, and an estimated sonar depth uncertainty of about 0.05--0.34 m
over a 0--90 m horizontal range. Final products can be less accurate after total
system and processing uncertainty is included.

## Scientific cautions

- The full-lake map is much smaller than the existing Monterey map and is best
  suited to short, high-resolution tests rather than a 73 km one-pass route.
- Backscatter gaps were filled by the publisher with a 5-by-5 focal-mean
  operation; acquisition-line striping remains visible and must not be mistaken
  for natural terrain texture.
- Use the GeoTIFF NoData sentinels rather than treating all finite float32 values
  as observations.
- These are public-domain USGS research data and are not for navigation.

Official landing page: <https://www.usgs.gov/data/bathymetry-and-acoustic-backscatter-data-jenkinson-lake-california-collected-during-three-usgs>

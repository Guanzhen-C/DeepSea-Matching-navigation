# Complete Monterey Canyon acoustic map

This directory is generated from the complete public USGS *Monterey Canyon
and Vicinity* raster extent. It is not the earlier 800 × 600 m demonstration
crop.

- Grid: 5 m, 5,842 columns × 3,534 rows
- UTM footprint: 583,530–612,740 m E, 4,060,670–4,078,340 m N
- Footprint size: 29.210 × 17.670 km
- Valid measured depths: 11,296,546 cells, −1,747.183 to −1.450 m
- Valid registered backscatter: 11,145,229 cells (98.6605% of valid depth)

The rectangular source raster contains USGS nodata outside the mapped
seafloor. Those cells remain `NaN`; the preparation step does not interpolate
or invent terrain across them. Likewise, 151,317 valid depth cells with no
backscatter observation remain `NaN` in `backscatter.npy`.

`backscatter.npy` combines all four published measured products (USGS
SWATHplus, EM-300, Reson 7125, and CSUMB SWATHplus). Because their native
encodings and sonar frequencies differ, each layer is robustly and
monotonically normalized to `[0, 1]` before equal-weight averaging in overlap
cells. It is a measured spatial-texture mosaic, not calibrated absolute target
strength. `backscatter_source_mask.npy` records provenance with bits A=1,
B=2, C=4, and D=8.

Rebuild from the downloaded official archives with:

```bash
PYTHONPATH=. python standalone/prepare_real_bathymetry.py
```

Pass `--download` if the archives are absent. Full source URLs, native nodata
values, archive checksums, per-layer coverage, processing details, and output
array checksums are in `metadata.json`. Verify generated files with:

```bash
cd data/monterey_canyon_full_5m
sha256sum -c sha256sums.txt
```

Source: [USGS data release](https://doi.org/10.5066/F7XD0ZQ4). The data are
public domain with attribution and are not intended for navigation.

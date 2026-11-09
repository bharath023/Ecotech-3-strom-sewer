"""Build dem5m.tif (5 m block-mean DEM, bad cells filled) from a source GeoTIFF.

Run: py make_dem5m.py reprojected_eco.tif [dem5m.tif]
Cells that are nodata or below MIN_VALID_M are treated as bad and filled from
the nearest valid cell. The 1 m source is resampled data, so 5 m loses nothing.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

FACTOR = 5          # 1 m -> 5 m
MIN_VALID_M = 100.0  # Noida ground is ~195-216 m; lower values are edge artefacts
NODATA_OUT = -9999.0


def main(src_path: Path, dst_path: Path) -> None:
    """Aggregate src_path to 5 m and write dst_path."""
    with rasterio.open(src_path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        tr, crs = src.transform, src.crs
        if src.res[0] != 1.0:
            raise ValueError(f"expected 1 m cells, got {src.res}")
    arr[arr < MIN_VALID_M] = np.nan
    h, w = (arr.shape[0] // FACTOR) * FACTOR, (arr.shape[1] // FACTOR) * FACTOR
    blocks = arr[:h, :w].reshape(h // FACTOR, FACTOR, w // FACTOR, FACTOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks
        dem = np.nanmean(blocks, axis=(1, 3))
    idx = distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
    dem = dem[tuple(idx)].astype("float32")
    with rasterio.open(dst_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
                       count=1, dtype="float32", crs=crs, nodata=NODATA_OUT,
                       transform=tr * tr.scale(FACTOR, FACTOR)) as dst:
        dst.write(dem, 1)
    print(f"wrote {dst_path} {dem.shape} min {dem.min():.1f} max {dem.max():.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: py make_dem5m.py reprojected_eco.tif [dem5m.tif]")
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dem5m.tif"))

"""Add `nodes` and `outfall` layers, and GL/depth/cover columns on `pipes`, in the GeoPackage.

Run: py add_outfall_layer.py      (after resize_loop.py)
Node coordinates come from the pipe line endpoints; attributes from nodes_final.csv.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import ecotech3_design as ed  # same folder: reuses the smoothed DEM and sampler

OUT = Path(__file__).resolve().parent / "out"
GPKG = OUT / "ecotech3_network.gpkg"
OUTFALL_NAME = "OUT"


def main() -> None:
    """Build node/outfall layers from the pipes layer and append them to the GeoPackage."""
    pipes = gpd.read_file(GPKG, layer="pipes")
    attrs = pd.read_csv(OUT / "nodes_final.csv").set_index("node")
    recs = []
    for r in pipes.itertuples():
        x0, y0 = r.geometry.coords[0]
        x1, y1 = r.geometry.coords[-1]
        recs.append((r.up, x0, y0))
        recs.append((r.down, x1, y1))
    xy = pd.DataFrame(recs, columns=["node", "x", "y"]).drop_duplicates("node")
    xy = xy.join(attrs, on="node")
    xy["depth_to_inv_m"] = xy["z"] - xy["inv"]
    out_row = pipes[pipes["down"] == OUTFALL_NAME].iloc[0]
    xy.loc[xy["node"] == OUTFALL_NAME, "inv"] = out_row["inv_down"]
    nodes = gpd.GeoDataFrame(xy, geometry=[Point(a, b) for a, b in zip(xy.x, xy.y)],
                             crs=pipes.crs)
    zs, tr = ed.smoothed_dem()
    ox, oy = nodes.loc[nodes["node"] == OUTFALL_NAME, ["x", "y"]].iloc[0]
    nodes.loc[nodes["node"] == OUTFALL_NAME, "z"] = ed.sample(zs, tr, ox, oy)
    nodes["depth_to_inv_m"] = nodes["z"] - nodes["inv"]
    gl = nodes.set_index("node")["z"]
    pipes["gl_up"] = pipes["up"].map(gl).round(2)
    pipes["gl_down"] = pipes["down"].map(gl).round(2)
    pipes["il_up"] = pipes["inv_up"].round(2)
    pipes["il_down"] = pipes["inv_down"].round(2)
    pipes["depth_up_m"] = (pipes["gl_up"] - pipes["il_up"]).round(2)       # GL - invert
    pipes["depth_down_m"] = (pipes["gl_down"] - pipes["il_down"]).round(2)
    pipes["cover_up_m"] = (pipes["depth_up_m"] - pipes["d"]).round(2)      # to crown
    pipes["cover_down_m"] = (pipes["depth_down_m"] - pipes["d"]).round(2)
    pipes.to_file(GPKG, layer="pipes", driver="GPKG")
    nodes.to_file(GPKG, layer="nodes", driver="GPKG")
    outfall = nodes[nodes["node"] == OUTFALL_NAME].copy()
    outfall["label"] = "OUTFALL (west) inv " + outfall["inv"].round(2).astype(str) + " m"
    outfall["q_5yr_m3s"] = out_row["q"]
    outfall.to_file(GPKG, layer="outfall", driver="GPKG")
    print(f"added layers: nodes ({len(nodes)}), outfall (1) at "
          f"{outfall.geometry.iloc[0].x:.0f}, {outfall.geometry.iloc[0].y:.0f}; "
          f"invert {out_row['inv_down']:.2f} m; fed by pipe {out_row['name']}")


if __name__ == "__main__":
    main()

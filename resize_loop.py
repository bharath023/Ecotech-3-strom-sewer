"""Resize Ecotech III sewers from SWMM 5.2 peak flows, then cross-check in SWMM6.

Run: python resize_loop.py   (after ecotech3_design.py has produced ./out)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from pyswmm import Simulation

import ecotech3_design as ed

log = logging.getLogger("resize")
MAX_ITER = 4
LINK_RE = re.compile(r"\s+(P\d+)\s+CONDUIT\s+([\d.]+)\s+\d+\s+\d+:\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def run_swmm5(inp: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a SWMM 5 file; return link table and continuity/flood totals."""
    with Simulation(str(inp)) as sim:
        for _ in sim:
            pass
    rpt = inp.with_suffix(".rpt").read_text(encoding="utf-8", errors="ignore")
    i = rpt.find("Link Flow Summary")
    rows = [m.groups() for ln in rpt[i:i + 12000].splitlines() if (m := LINK_RE.match(ln))]
    links = pd.DataFrame(rows, columns=["pipe", "qmax", "vmax", "q_over_full", "d_over_full"])
    links = links.astype({c: float for c in links.columns[1:]})
    flood = re.search(r"Flooding Loss \.+\s+([\d.]+)\s+([\d.]+)", rpt)
    out = re.search(r"OUT\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", rpt)
    return links, {"flood_m3": float(flood.group(2)) * 1e3,
                   "peak_out": float(out.group(1)), "vol_out_ML": float(out.group(2))}


def main() -> None:
    """Iterate rational design against dynamic-wave peaks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    site = ed.load_site()
    cells = ed.build_cells(site)
    zs, tr = ed.smoothed_dem()
    nodes, pipes, outfall = ed.build_network(cells, zs, tr)
    q_floor: dict[str, float] = {}
    hist = []
    for it in range(MAX_ITER + 1):
        ed.design_pipes(cells, nodes, pipes, outfall, q_floor)
        inp = ed.OUT / "iter_minor.inp"
        ed.write_inp(inp, cells, nodes, pipes, outfall, ed.design_hyetograph(ed.DESIGN_RP),
                     f"Ecotech III iter {it}")
        links, tot = run_swmm5(inp)
        n_over = int((links.q_over_full > 1.0).sum())
        n_fast = int((links.vmax > ed.V_MAX).sum())
        log.info("iter %d: over-capacity %d, v>%.1f %d, flood %.0f m3, peak out %.1f m3/s",
                 it, n_over, ed.V_MAX, n_fast, tot["flood_m3"], tot["peak_out"])
        hist.append({"iter": it, "over_capacity": n_over, "v_gt_max": n_fast, **tot})
        if n_over == 0 and tot["flood_m3"] < 10:
            break
        q_floor = {r.pipe: max(q_floor.get(r.pipe, 0.0), r.qmax) for r in links.itertuples()}
    # final deliverables
    res = ed.design_pipes(cells, nodes, pipes, outfall, q_floor)
    pdf = pd.DataFrame([p.__dict__ for p in pipes])
    pdf.to_csv(ed.OUT / "pipes_final.csv", index=False)
    res.to_csv(ed.OUT / "nodes_final.csv", index=False)
    pd.DataFrame(hist).to_csv(ed.OUT / "resize_history.csv", index=False)
    for rp, tag in ((ed.DESIGN_RP, "minor"), (ed.CHECK_RP, "major")):
        inp = ed.OUT / f"ecotech3_final_{tag}_{rp}yr.inp"
        ed.write_inp(inp, cells, nodes, pipes, outfall, ed.design_hyetograph(rp),
                     f"Ecotech III final design, {rp}-yr 3h storm")
        links, tot = run_swmm5(inp)
        links.to_csv(ed.OUT / f"final_{tag}_links_swmm5.csv", index=False)
        log.info("FINAL %s %d-yr SWMM5: over-capacity %d, v>max %d, flood %.0f m3, peak out %.1f",
                 tag, rp, (links.q_over_full > 1).sum(), (links.vmax > ed.V_MAX).sum(),
                 tot["flood_m3"], tot["peak_out"])
        pd.Series(tot).to_json(ed.OUT / f"final_{tag}_totals_swmm5.json")
    gdf = gpd_pipes(nodes, pipes, outfall)
    gdf.to_file(ed.OUT / "ecotech3_network.gpkg", layer="pipes", driver="GPKG")
    cells.to_file(ed.OUT / "ecotech3_network.gpkg", layer="subcatchments", driver="GPKG")


def gpd_pipes(nodes: pd.DataFrame, pipes, outfall):
    """Pipe lines as a GeoDataFrame with design attributes."""
    import geopandas as gpd
    from shapely.geometry import LineString
    xy = {r.node: (r.x, r.y) for r in nodes.itertuples()}
    xy["OUT"] = (outfall["x"], outfall["y"])
    recs = []
    for p in pipes:
        d = dict(p.__dict__)
        d["geometry"] = LineString([xy[p.up], xy[p.down]])
        recs.append(d)
    return gpd.GeoDataFrame(recs, crs=ed.CRS)


if __name__ == "__main__":
    main()

"""Build a standalone 2D live map (HTML) from the Ecotech III SWMM runs.

Run: py make_live_map.py            (after resize_loop.py; works with or without
add_outfall_layer.py). Output: out/ecotech3_live_map.html - open it by double-click
in Chrome/Edge. Nothing is loaded from the internet.

The script re-runs the final 5-year and 50-year .inp files in SWMM 5.2 and stores
water depth, flooding, pipe flow/fill and subcatchment runoff every SAMPLE_MIN minutes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from pyswmm import Links, Nodes, Simulation, Subcatchments

import ecotech3_design as ed

log = logging.getLogger("livemap")
OUT = ed.OUT
GPKG = OUT / "ecotech3_network.gpkg"
HTML_OUT = OUT / "ecotech3_live_map.html"
SAMPLE_MIN = 2
EVENTS = [
    (f"{ed.DESIGN_RP}-year design storm (minor system)", f"ecotech3_final_minor_{ed.DESIGN_RP}yr.inp"),
    (f"{ed.CHECK_RP}-year storm (major check)", f"ecotech3_final_major_{ed.CHECK_RP}yr.inp"),
]
OUTFALL = "OUT"

TEMPLATE_PATH = Path(__file__).resolve().parent / "live_map_template.html"


def load_geometry() -> dict:
    """Return map geometry (local metres) for subcatchments, nodes and pipes."""
    pipes = gpd.read_file(GPKG, layer="pipes")
    subs = gpd.read_file(GPKG, layer="subcatchments")
    attrs = pd.read_csv(OUT / "nodes_final.csv").set_index("node")
    xy: dict[str, tuple[float, float]] = {}
    for r in pipes.itertuples():
        xy.setdefault(r.up, r.geometry.coords[0])
        xy.setdefault(r.down, r.geometry.coords[-1])
    minx, miny, maxx, maxy = subs.total_bounds
    x0, y0 = float(minx), float(miny)

    def loc(p) -> list[float]:
        return [round(p[0] - x0, 1), round(p[1] - y0, 1)]

    ids = list(xy)
    index = {n: i for i, n in enumerate(ids)}
    out_pipe = pipes[pipes["down"] == OUTFALL].iloc[0]
    nodes = []
    for n in ids:
        if n in attrs.index:
            gl, inv = float(attrs.z[n]), float(attrs.inv[n])
        else:                                      # outfall
            inv = float(out_pipe["inv_down"])
            gl = float(out_pipe["gl_down"]) if "gl_down" in pipes.columns else inv + 10.0
        nodes.append({"id": n, "p": loc(xy[n]), "gl": round(gl, 2), "inv": round(inv, 2),
                      "rim": round(gl - inv, 2)})
    links = []
    for r in pipes.itertuples():
        if r.shape == "BOX":
            qfull = ed.box_full_flow(r.d, r.width, r.slope)[0]
            label = f"{r.d:g} x {r.width:g} m box"
        else:
            qfull = ed.full_flow(r.d, r.slope)[0]
            label = f"D{r.d * 1000:.0f} mm"
        links.append({"id": r.name, "a": index[r.up], "b": index[r.down], "d": float(r.d),
                      "qfull": round(qfull, 2), "label": label, "len": round(r.length, 1),
                      "slope": round(r.slope * 1000, 3), "il": [round(r.inv_up, 2), round(r.inv_down, 2)]})
    subl = []
    for r in subs.itertuples():
        ring = [loc(c) for c in r.geometry.simplify(1.0).exterior.coords]
        subl.append({"id": r.sc, "ha": round(float(r.area_ha), 2), "C": round(float(r.C), 2),
                     "imp": round(float(r.imperv), 0), "ring": ring})
    w = round(float(maxx - minx), 1)
    h = round(float(maxy - miny), 1)
    # affine local metres -> Web Mercator (for web tiles): X = a*x + c*y + e, Y = b*x + d*y + f
    to_merc = Transformer.from_crs(ed.CRS, 3857, always_xy=True)
    m00, m10, m01 = (to_merc.transform(x0, y0), to_merc.transform(x0 + w, y0),
                     to_merc.transform(x0, y0 + h))
    merc = {"a": (m10[0] - m00[0]) / w, "b": (m10[1] - m00[1]) / w,
            "c": (m01[0] - m00[0]) / h, "d": (m01[1] - m00[1]) / h, "e": m00[0], "f": m00[1]}
    lat = Transformer.from_crs(ed.CRS, 4326, always_xy=True).transform(x0 + w / 2, y0 + h / 2)[1]
    return {"w": w, "h": h, "merc": merc, "lat": round(lat, 5), "subs": subl, "nodes": nodes, "links": links,
            "ids": ids, "sub_ids": [s["id"] for s in subl], "link_ids": [l["id"] for l in links]}


def run_event(inp: Path, geo: dict) -> dict:
    """Run one SWMM event and return time-series arrays sampled every SAMPLE_MIN."""
    t, rain, outq, flood = [], [], [], []
    nd, nf, lq, ld, sr = [], [], [], [], []
    with Simulation(str(inp)) as sim:
        nodes, links, subs = Nodes(sim), Links(sim), Subcatchments(sim)
        sim.step_advance(SAMPLE_MIN * 60)
        start = sim.start_time
        for _ in sim:
            minutes = (sim.current_time - start).total_seconds() / 60.0
            t.append(round(minutes, 1))
            rain.append(round(subs[geo["sub_ids"][0]].rainfall, 1))
            outq.append(round(nodes[OUTFALL].total_inflow, 2))
            row_d, row_f = [], []
            for n in geo["ids"]:
                nn = nodes[n]
                row_d.append(round(nn.depth, 2) if n != OUTFALL else 0.0)
                row_f.append(round(nn.flooding, 3) if n != OUTFALL else 0.0)
            nd.append(row_d)
            nf.append(row_f)
            flood.append(round(sum(row_f), 3))
            lq.append([round(links[k].flow, 2) for k in geo["link_ids"]])
            ld.append([round(links[k].depth, 2) for k in geo["link_ids"]])
            sr.append([round(subs[k].runoff, 3) for k in geo["sub_ids"]])
    log.info("%s: %d samples, peak outfall %.1f m3/s, peak flooding %.2f m3/s", inp.name,
             len(t), max(outq), max(flood))
    return {"t": t, "rain": rain, "outq": outq, "flood": flood, "nd": nd, "nf": nf,
            "lq": lq, "ld": ld, "sr": sr}


def main() -> None:
    """Generate the HTML viewer."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    geo = load_geometry()
    events = []
    for label, name in EVENTS:
        ev = run_event(OUT / name, geo)
        ev["label"] = label
        events.append(ev)
    data = {"geo": geo, "events": events, "sample_min": SAMPLE_MIN,
            "note": "Concept design on an assumed 350 m grid; DEM is a DSM. Indicative only."}
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    HTML_OUT.write_text(html, encoding="utf-8")
    log.info("wrote %s (%.0f KB)", HTML_OUT, HTML_OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()

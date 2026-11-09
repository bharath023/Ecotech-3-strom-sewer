"""Ecotech III (Greater Noida) conceptual storm sewer design.

Pipeline: land-use parcels -> grid subcatchments -> gravity network to a western
outfall -> rational-method pipe sizing (CPHEEO-style criteria) -> SWMM INP files.

Run: py make_dem5m.py reprojected_eco.tif   (once), then py ecotech3_design.py
All CRS work is in EPSG:32643 (UTM 43N, metres).

ASSUMPTIONS (all flagged in the report): see the constants block below.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

log = logging.getLogger("ecotech3")

# ----------------------------------------------------------------- paths / CRS
WORK = Path(__file__).resolve().parent
OUT = WORK / "out"
LANDUSE_SHP = WORK / "Landuse_Masterplan.shp"   # keep .dbf/.shx/.prj/.cpg beside it
DEM_5M = WORK / "dem5m.tif"
CRS = 32643

# ------------------------------------------------------------------- geometry
OUTFALL_XY = (739512.0, 3158863.0)   # western boundary cell chosen by user
CELL_M = 350.0                       # ASSUMED collector grid spacing
MIN_CELL_HA = 3.0                    # smaller slivers merge into a neighbour
DEM_SMOOTH_SIGMA_M = 150.0           # DEM is a DSM with mounds/pits: smooth hard
MIN_COVER_M = 1.2                    # cover to pipe crown (ASSUMED)

# ------------------------------------------------------------------- hydrology
P24_50_MM = 240.0                    # CWC 1(e) Plate 9 read-off (confirm!)
# CWC 1(e) Fig.10 ratios (50-yr T-hr / 24-hr), duration in hours
CWC_RATIO = {1: 0.340, 2: 0.475, 3: 0.560, 4: 0.607, 6: 0.680, 9: 0.770,
             12: 0.840, 18: 0.930, 24: 1.000}
SUBHOUR_EXP = math.log(0.56 / 0.34) / math.log(3.0)   # EXTRAPOLATED below 1 h
RP_FACTOR = {50: 1.00, 10: 0.73, 5: 0.60}  # ASSUMED Gumbel-style scaling vs 50-yr
DESIGN_RP = 5                        # return period (yr) used to SIZE the sewers
CHECK_RP = 50                        # return period (yr) used for the flooding CHECK
STORM_MIN = 180                      # design storm duration (min)
PEAK_FRAC = 0.4                      # position of the peak within the storm (0-1)
RAIN_FILE = WORK / "rainfall_input.csv"   # YOUR rainfall: edit this file (see README)
T_INLET_MIN = 15.0                   # ASSUMED inlet time for plots
LANDUSE_PARAMS = {                   # runoff coeff C (CPHEEO industrial 0.60-0.90)
    "INDUSTRIAL": {"C": 0.75, "imperv": 70.0},
    "INSTITUTIONAL": {"C": 0.60, "imperv": 55.0},
}
CN_PERVIOUS = 80.0                   # ASSUMED alluvial silty loam

# ------------------------------------------------------------------- hydraulics
MANNING_N = 0.013
V_MIN, V_MAX = 0.6, 3.0              # m/s (CPHEEO summary: 0.6 / 3-4 concrete)
DIAMETERS_M = [0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1, 2.4]
BOX_HEIGHTS_M = [2.0, 2.5, 3.0]
BOX_WIDTHS_M = [round(2.0 + 0.5 * k, 1) for k in range(0, 25)]   # 2.0 .. 14.0 m
MAX_TRUNK_SLOPE = 0.001              # cap when steepening a trunk beyond ground fall


# =============================================================== rainfall (IDF)
def duration_ratio(t_min: float) -> float:
    """CWC 1(e) ratio of T-hr to 24-hr point rainfall; extrapolated < 1 h."""
    t_h = t_min / 60.0
    if t_h <= 1.0:
        return CWC_RATIO[1] * t_h ** SUBHOUR_EXP
    hrs = sorted(CWC_RATIO)
    return float(np.interp(t_h, hrs, [CWC_RATIO[h] for h in hrs]))


_RAIN: dict[int, tuple[np.ndarray, np.ndarray]] | None = None   # rp -> (ln minutes, ln mm/h)


def load_rainfall_table(path: Path = RAIN_FILE) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read the user's IDF table (empty dict if the file does not exist).

    CSV columns: return_period_yr, duration_min, and intensity_mm_h OR depth_mm.
    Lines starting with # are comments.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"return_period_yr", "duration_min"} <= set(df.columns):
        raise ValueError(f"{path.name}: need columns return_period_yr, duration_min "
                         "and intensity_mm_h (or depth_mm)")
    if "intensity_mm_h" in df.columns:
        df["i"] = df["intensity_mm_h"].astype(float)
    elif "depth_mm" in df.columns:
        df["i"] = df["depth_mm"].astype(float) / (df["duration_min"].astype(float) / 60.0)
    else:
        raise ValueError(f"{path.name}: add an intensity_mm_h or depth_mm column")
    if (df[["return_period_yr", "duration_min", "i"]].astype(float) <= 0).any().any():
        raise ValueError(f"{path.name}: all values must be positive numbers")
    table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rp, g in df.groupby("return_period_yr"):
        g = g.sort_values("duration_min")
        if len(g) < 2:
            raise ValueError(f"{path.name}: return period {rp} needs at least two durations")
        if (np.diff(g["i"].to_numpy()) > 0).any():
            log.warning("%s: RP %s intensity rises with duration - please check", path.name, rp)
        table[int(rp)] = (np.log(g["duration_min"].to_numpy(float)), np.log(g["i"].to_numpy(float)))
    return table


def rain_table() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Cached user table (loaded once)."""
    global _RAIN
    if _RAIN is None:
        _RAIN = load_rainfall_table()
        if _RAIN:
            log.info("rainfall: using %s (return periods %s)", RAIN_FILE.name, sorted(_RAIN))
        else:
            log.warning("rainfall: %s not found - using ASSUMED CWC-scaled rainfall", RAIN_FILE.name)
    return _RAIN


def _assumed_depth_mm(t_min: float, rp: int) -> float:
    """Fallback depth (mm): CWC 1(e) ratios x assumed return-period factor."""
    return P24_50_MM * RP_FACTOR[rp] * duration_ratio(t_min)


def intensity_mm_h(t_min: float, rp: int) -> float:
    """Average intensity (mm/h) for duration t_min (log-log interpolation of the table)."""
    tab = rain_table()
    if not tab:
        return _assumed_depth_mm(t_min, rp) / (t_min / 60.0)
    if rp not in tab:
        raise ValueError(f"return period {rp} yr is not in {RAIN_FILE.name}; "
                         f"available: {sorted(tab)}")
    lx, ly = tab[rp]
    x = math.log(t_min)
    if x <= lx[0]:                                  # extrapolate with the end slope
        y = ly[0] + (ly[1] - ly[0]) / (lx[1] - lx[0]) * (x - lx[0])
    elif x >= lx[-1]:
        y = ly[-1] + (ly[-1] - ly[-2]) / (lx[-1] - lx[-2]) * (x - lx[-1])
    else:
        y = float(np.interp(x, lx, ly))
    return math.exp(y)


def depth_mm(t_min: float, rp: int) -> float:
    """Point rainfall depth (mm) for duration t_min and return period rp."""
    return intensity_mm_h(t_min, rp) * t_min / 60.0


def write_rainfall_used() -> None:
    """Save the rainfall actually used (for the record) to out/rainfall_used.csv."""
    src = RAIN_FILE.name if rain_table() else "ASSUMED CWC-scaled"
    rows = [{"return_period_yr": rp, "duration_min": t, "intensity_mm_h": round(intensity_mm_h(t, rp), 1),
             "depth_mm": round(depth_mm(t, rp), 1), "source": src}
            for rp in (DESIGN_RP, CHECK_RP) for t in (5, 10, 15, 30, 60, 120, 180)]
    pd.DataFrame(rows).to_csv(OUT / "rainfall_used.csv", index=False)


def design_hyetograph(rp: int, total_min: int = STORM_MIN, step: int = 5,
                      peak_frac: float = PEAK_FRAC) -> list[float]:
    """Alternating-block hyetograph; returns intensity (mm/h) per step."""
    n = total_min // step
    inc = [depth_mm(step * (k + 1), rp) - depth_mm(step * k, rp) if k else depth_mm(step, rp)
           for k in range(n)]
    ordered = sorted(inc, reverse=True)
    out = [0.0] * n
    peak_idx = int(peak_frac * n)
    left, right = peak_idx, peak_idx + 1
    for k, v in enumerate(ordered):
        if k % 2 == 0 and left >= 0:
            out[left] = v
            left -= 1
        elif right < n:
            out[right] = v
            right += 1
        else:
            out[left] = v
            left -= 1
    return [v * 60.0 / step for v in out]


# ============================================================ subcatchments
def load_site() -> gpd.GeoDataFrame:
    """Return Ecotech 3 parcels (INDUSTRIAL / INSTITUTIONAL) in UTM 43N."""
    gdf = gpd.read_file(LANDUSE_SHP).to_crs(CRS)
    gdf["geometry"] = gdf.geometry.force_2d().make_valid()
    site = gdf[gdf["name"].str.strip().str.upper() == "ECOTECH 3"].copy()
    site["lu"] = site["type"].str.strip().str.upper()
    unknown = set(site["lu"]) - set(LANDUSE_PARAMS)
    if unknown:
        raise ValueError(f"No parameters for land use: {unknown}")
    return site[["lu", "geometry"]].reset_index(drop=True)


def build_cells(site: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Grid the site in its principal-axis frame and merge slivers."""
    union = unary_union(site.geometry)
    rect = union.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]),
              math.degrees(math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i]))) for i in range(4)]
    angle = max(edges)[1]
    origin = rect.centroid
    rot = affinity.rotate(union, -angle, origin=origin)
    minx, miny, maxx, maxy = rot.bounds
    cells = []
    for gx in np.arange(minx, maxx, CELL_M):
        for gy in np.arange(miny, maxy, CELL_M):
            c = box(gx, gy, gx + CELL_M, gy + CELL_M).intersection(rot)
            if not c.is_empty and c.area > 1.0:
                cells.append(affinity.rotate(c, angle, origin=origin))
    gdf = gpd.GeoDataFrame(geometry=cells, crs=CRS)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    # merge slivers into the neighbour sharing the longest boundary
    while True:
        areas = gdf.area / 1e4
        small = areas[areas < MIN_CELL_HA]
        if small.empty:
            break
        i = small.idxmin()
        geom = gdf.geometry[i]
        best, best_len = None, 0.0
        for j in gdf.index[gdf.index != i]:
            shared = geom.intersection(gdf.geometry[j]).length
            if shared > best_len:
                best, best_len = j, shared
        if best is None:           # isolated sliver: drop with a warning
            log.warning("dropping isolated sliver %.2f ha", areas[i])
            gdf = gdf.drop(index=i).reset_index(drop=True)
            continue
        gdf.loc[best, "geometry"] = unary_union([gdf.geometry[best], geom])
        gdf = gdf.drop(index=i).reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.apply(lambda g: g if g.geom_type == "Polygon"
                                         else max(g.geoms, key=lambda p: p.area))
    gdf["sc"] = [f"S{k + 1:02d}" for k in range(len(gdf))]
    # land-use areas per cell
    ind, inst = [], []
    for geom in gdf.geometry:
        a = {"INDUSTRIAL": 0.0, "INSTITUTIONAL": 0.0}
        for lu, pg in zip(site["lu"], site.geometry):
            a[lu] += geom.intersection(pg).area
        ind.append(a["INDUSTRIAL"])
        inst.append(a["INSTITUTIONAL"])
    gdf["a_ind"], gdf["a_inst"] = ind, inst
    gdf["area_ha"] = (gdf["a_ind"] + gdf["a_inst"]) / 1e4
    wt = lambda k: (gdf["a_ind"] * LANDUSE_PARAMS["INDUSTRIAL"][k]
                    + gdf["a_inst"] * LANDUSE_PARAMS["INSTITUTIONAL"][k]) / (gdf["a_ind"] + gdf["a_inst"])
    gdf["C"] = wt("C")
    gdf["imperv"] = wt("imperv")
    return gdf


# ============================================================== terrain
def smoothed_dem() -> tuple[np.ndarray, rasterio.Affine]:
    """Return smoothed DEM array and transform (smoothing removes DSM clutter)."""
    with rasterio.open(DEM_5M) as src:
        arr = src.read(1).astype("float64")
        tr = src.transform
        res = src.res[0]
    return gaussian_filter(arr, DEM_SMOOTH_SIGMA_M / res), tr


def sample(arr: np.ndarray, tr: rasterio.Affine, x: float, y: float) -> float:
    """Sample array at map coordinates."""
    col, row = ~tr * (x, y)
    return float(arr[int(row), int(col)])


# =============================================================== network
@dataclass
class Pipe:
    """Designed gravity pipe."""
    name: str
    up: str
    down: str
    length: float
    q: float = 0.0
    d: float = 0.0            # diameter (circular) or height (box), m
    width: float = 0.0        # box width (m); 0 for circular
    slope: float = 0.0
    v_full: float = 0.0
    inv_up: float = 0.0
    inv_down: float = 0.0
    note: str = ""
    shape: str = "CIRC"


def full_flow(d: float, s: float) -> tuple[float, float]:
    """Manning full-pipe (Q m3/s, V m/s) for diameter d (m) and slope s (m/m)."""
    v = (1.0 / MANNING_N) * (d / 4.0) ** (2.0 / 3.0) * math.sqrt(s)
    return v * math.pi * d * d / 4.0, v


def box_full_flow(h: float, b: float, s: float) -> tuple[float, float]:
    """Manning full-section (Q, V) for a closed rectangular box h x b (m)."""
    area, perim = h * b, 2.0 * (h + b)
    v = (1.0 / MANNING_N) * (area / perim) ** (2.0 / 3.0) * math.sqrt(s)
    return v * area, v


def box_slope_for_v(h: float, b: float, v: float) -> float:
    """Slope giving full-section velocity v in a box h x b."""
    return (v * MANNING_N / ((h * b) / (2.0 * (h + b))) ** (2.0 / 3.0)) ** 2


def slope_for_v(d: float, v: float) -> float:
    """Slope giving full-pipe velocity v in a pipe of diameter d."""
    return (v * MANNING_N / (d / 4.0) ** (2.0 / 3.0)) ** 2


def build_network(cells: gpd.GeoDataFrame, zs: np.ndarray, tr: rasterio.Affine):
    """Return (nodes df, pipes list, tree predecessor map, outfall info)."""
    pts = [g.representative_point() for g in cells.geometry]
    nodes = pd.DataFrame({"node": cells["sc"].str.replace("S", "J"), "sc": cells["sc"],
                          "x": [p.x for p in pts], "y": [p.y for p in pts]})
    nodes["z"] = [sample(zs, tr, x, y) for x, y in zip(nodes.x, nodes.y)]
    # adjacency by shared boundary
    G = nx.DiGraph()
    for i, gi in enumerate(cells.geometry):
        for j in range(i + 1, len(cells)):
            if gi.intersection(cells.geometry[j]).length > 50.0:
                d = math.hypot(nodes.x[i] - nodes.x[j], nodes.y[i] - nodes.y[j])
                for a, b in ((i, j), (j, i)):
                    rise = max(0.0, nodes.z[b] - nodes.z[a])
                    G.add_edge(a, b, cost=d + 500.0 * rise, dist=d)
    ox, oy = OUTFALL_XY
    dists = np.hypot(nodes.x - ox, nodes.y - oy)
    last = int(dists.idxmin())
    # shortest-path tree toward `last` (reverse graph, predecessors = downstream hop)
    pred, _ = nx.dijkstra_predecessor_and_distance(G.reverse(copy=True), last, weight="cost")
    down = {n: p[0] for n, p in pred.items() if p}
    if len(down) != len(nodes) - 1:
        raise RuntimeError("network not fully connected to outfall node")
    zo = sample(zs, tr, ox, oy)
    pipes = []
    for u, d in down.items():
        pipes.append(Pipe(f"P{len(pipes) + 1:02d}", nodes.node[u], nodes.node[d],
                          G[u][d]["dist"]))
    pipes.append(Pipe(f"P{len(pipes) + 1:02d}", nodes.node[last], "OUT",
                      float(dists[last])))
    outfall = {"x": ox, "y": oy, "z": zo, "last": nodes.node[last]}
    return nodes, pipes, outfall


def design_pipes(cells, nodes, pipes, outfall,
                 q_floor: dict[str, float] | None = None) -> pd.DataFrame:
    """Rational-method sizing + crown-matched inverts. Returns node results."""
    zmap = dict(zip(nodes.node, nodes.z))
    zmap["OUT"] = outfall["z"]
    area_ca = dict(zip(nodes.node, cells["area_ha"] * cells["C"]))
    area_ha = dict(zip(nodes.node, cells["area_ha"]))
    down_of = {p.up: p for p in pipes}
    ups: dict[str, list[Pipe]] = {n: [] for n in list(nodes.node) + ["OUT"]}
    for p in pipes:
        ups[p.down].append(p)

    # topological order: leaves first
    order, seen = [], set()
    def visit(n: str) -> None:
        for p in ups[n]:
            if p.up not in seen:
                visit(p.up)
        if n not in seen and n != "OUT":
            seen.add(n)
            order.append(n)
    visit("OUT")

    tc: dict[str, float] = {}
    cum_ca: dict[str, float] = {}
    cum_ha: dict[str, float] = {}
    crown_in: dict[str, float] = {}
    for n in order:
        tc_n = T_INLET_MIN
        ca, ha = area_ca[n], area_ha[n]
        for p in ups[n]:
            tc_n = max(tc_n, tc[p.up] + p.length / p.v_full / 60.0)
            ca += cum_ca[p.up]
            ha += cum_ha[p.up]
        tc[n], cum_ca[n], cum_ha[n] = tc_n, ca, ha
        p = down_of[n]
        i_mm = intensity_mm_h(tc_n, DESIGN_RP)
        p.q = ca * i_mm / 360.0                       # Q = C i A / 360  (A ha)
        if q_floor:                                   # simulation-guided resize
            p.q = max(p.q, q_floor.get(p.name, 0.0))
        g_slope = max((zmap[p.up] - zmap[p.down]) / p.length, 0.0)
        s_floor = max(g_slope, 0.0)
        chosen = None
        for d in DIAMETERS_M:                         # circular pipes first
            sl = min(max(s_floor, slope_for_v(d, V_MIN * 1.1)), slope_for_v(d, V_MAX))
            qf, vf = full_flow(d, sl)
            if qf >= p.q and V_MIN <= vf <= V_MAX:
                chosen = ("CIRC", d, 0.0, sl, vf)
                break
        if chosen is None:                            # then single-cell boxes
            cands = sorted(((h * b, h, b) for h in BOX_HEIGHTS_M for b in BOX_WIDTHS_M))
            for _, h, b in cands:
                sl = min(max(s_floor, box_slope_for_v(h, b, V_MIN * 1.1)),
                         box_slope_for_v(h, b, V_MAX))
                qf, vf = box_full_flow(h, b, sl)
                if qf >= p.q and V_MIN <= vf <= V_MAX:
                    chosen = ("BOX", h, b, sl, vf)
                    break
        if chosen is None:                            # steepen a box, slope capped
            cands = sorted(((h * b, h, b) for h in BOX_HEIGHTS_M for b in BOX_WIDTHS_M))
            for _, h, b in cands:
                area, perim = h * b, 2.0 * (h + b)
                k_cap = area * (area / perim) ** (2.0 / 3.0) / MANNING_N   # Q = k sqrt(s)
                s_req = (p.q / k_cap) ** 2
                sl = max(s_floor, box_slope_for_v(h, b, V_MIN * 1.1), s_req)
                qf, vf = box_full_flow(h, b, sl)
                if sl <= MAX_TRUNK_SLOPE and vf <= V_MAX:
                    chosen = ("BOX", h, b, sl, vf)
                    p.note = f"laid steeper than ground ({sl * 1000:.2f} per mille)"
                    break
        if chosen is None:
            h, b = BOX_HEIGHTS_M[-1], BOX_WIDTHS_M[-1]
            qf, vf = box_full_flow(h, b, MAX_TRUNK_SLOPE)
            chosen = ("BOX", h, b, MAX_TRUNK_SLOPE, vf)
            p.note = f"UNDERSIZED: capacity {qf:.1f} < {p.q:.1f} m3/s"
        p.shape, p.d, p.width, p.slope, p.v_full = chosen
        if p.shape == "BOX":
            p.note = (p.note + " " if p.note else "") + "closed box / lined drain"
        # invert: crown matches lowest incoming crown, and cover is respected
        crown_up = zmap[n] - MIN_COVER_M
        if ups[n]:
            crown_up = min(crown_up, min(crown_in[q.up] for q in ups[n]))
        p.inv_up = crown_up - p.d
        p.inv_down = p.inv_up - p.slope * p.length
        crown_in[n] = p.inv_down + p.d                # crown at downstream node
    # tidy: store downstream crown by node for the next pipe (already via crown_in[n])
    rows = []
    for n in order:
        p = down_of[n]
        rows.append({"node": n, "z": zmap[n], "inv": p.inv_up, "tc_min": tc[n],
                     "cum_ha": cum_ha[n], "q_design": p.q})
    return pd.DataFrame(rows)


# =============================================================== SWMM writer
def write_inp(path: Path, cells, nodes, pipes, outfall, hyet: list[float],
              title: str) -> None:
    """Write a SWMM 5.x input file (CMS, dynamic wave, CN infiltration)."""
    z = dict(zip(nodes.node, nodes.z))
    inv_node = {p.up: p.inv_up for p in pipes}
    lines = [
        "[TITLE]", title, "",
        "[OPTIONS]",
        "FLOW_UNITS CMS", "INFILTRATION CURVE_NUMBER", "FLOW_ROUTING DYNWAVE",
        "LINK_OFFSETS DEPTH", "START_DATE 01/01/2026", "START_TIME 00:00:00",
        "END_DATE 01/01/2026", "END_TIME 06:00:00", "REPORT_START_DATE 01/01/2026",
        "REPORT_START_TIME 00:00:00", "SWEEP_START 01/01", "SWEEP_END 12/31",
        "WET_STEP 00:01:00", "DRY_STEP 00:05:00", "ROUTING_STEP 00:00:05",
        "ALLOW_PONDING NO", "MIN_SURFAREA 12.566", "",
        "[RAINGAGES]", ";;Name Format Interval SCF Source",
        "RG1 INTENSITY 0:05 1.0 TIMESERIES STORM", "",
        "[TIMESERIES]",
    ]
    for k, v in enumerate(hyet):
        t = k * 5
        lines.append(f"STORM {t // 60:02d}:{t % 60:02d} {v:.3f}")
    lines += ["STORM 03:00 0.000", "", "[SUBCATCHMENTS]",
              ";;Name Raingage Outlet Area %Imperv Width %Slope CurbLen"]
    for _, r in cells.iterrows():
        width = r.area_ha * 1e4 / 150.0              # overland flow length 150 m
        lines.append(f"{r.sc} RG1 {r.sc.replace('S', 'J')} {r.area_ha:.3f} "
                     f"{r.imperv:.1f} {width:.0f} 0.5 0")
    lines += ["", "[SUBAREAS]", ";;Sub N-Imp N-Perv S-Imp S-Perv PctZero RouteTo"]
    lines += [f"{sc} 0.013 0.15 1.5 5.0 25 OUTLET" for sc in cells.sc]
    lines += ["", "[INFILTRATION]", ";;Sub CN Ksat DryTime"]
    lines += [f"{sc} {CN_PERVIOUS} 0.5 7" for sc in cells.sc]
    lines += ["", "[JUNCTIONS]", ";;Name Invert MaxDepth Init Surcharge Aponded"]
    for n in nodes.node:
        lines.append(f"{n} {inv_node[n]:.3f} {z[n] - inv_node[n]:.3f} 0 0 0")
    out_inv = [p for p in pipes if p.down == "OUT"][0].inv_down
    lines += ["", "[OUTFALLS]", ";;Name Invert Type", f"OUT {out_inv:.3f} FREE",
              "", "[CONDUITS]", ";;Name From To Length N InOff OutOff InitFlow MaxFlow"]
    for p in pipes:
        lines.append(f"{p.name} {p.up} {p.down} {p.length:.1f} {MANNING_N} 0 0 0 0")
    lines += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels"]
    for p in pipes:
        if p.shape == "BOX":
            lines.append(f"{p.name} RECT_CLOSED {p.d:.2f} {p.width:.2f} 0 0 1")
        else:
            lines.append(f"{p.name} CIRCULAR {p.d:.2f} 0 0 0 1")
    lines += ["", "[COORDINATES]", ";;Node X Y"]
    lines += [f"{r.node} {r.x:.1f} {r.y:.1f}" for r in nodes.itertuples()]
    lines.append(f"OUT {outfall['x']:.1f} {outfall['y']:.1f}")
    lines += ["", "[Polygons]", ";;Sub X Y"]
    for _, r in cells.iterrows():
        for x, y in list(r.geometry.exterior.coords):
            lines.append(f"{r.sc} {x:.1f} {y:.1f}")
    lines += ["", "[REPORT]", "INPUT NO", "CONTROLS NO", "SUBCATCHMENTS NONE",
              "NODES ALL", "LINKS ALL", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full design pipeline and write outputs to ./out."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(exist_ok=True)
    site = load_site()
    cells = build_cells(site)
    log.info("site %.2f km2 -> %d subcatchments (%.1f-%.1f ha)",
             cells.area_ha.sum() / 100, len(cells), cells.area_ha.min(), cells.area_ha.max())
    zs, tr = smoothed_dem()
    nodes, pipes, outfall = build_network(cells, zs, tr)
    res = design_pipes(cells, nodes, pipes, outfall)
    pdf = pd.DataFrame([p.__dict__ for p in pipes])
    pdf.to_csv(OUT / "pipes_design.csv", index=False)
    res.to_csv(OUT / "nodes_design.csv", index=False)
    cells.to_file(OUT / "ecotech3_network.gpkg", layer="subcatchments", driver="GPKG")
    for rp, tag in ((DESIGN_RP, "minor"), (CHECK_RP, "major")):
        write_inp(OUT / f"ecotech3_{tag}_{rp}yr.inp", cells, nodes, pipes, outfall,
                  design_hyetograph(rp), f"Ecotech III storm sewer, {rp}-yr 3h storm")
    log.info("design Q peak at outfall %.2f m3/s; diameters %s", pdf.q.max(),
             sorted(pdf.d.unique()))
    write_rainfall_used()
    for rp in sorted(rain_table()) if rain_table() else sorted(RP_FACTOR):
        log.info("IDF RP%d: 15 min %.0f mm/h, 30 min %.0f, 60 min %.0f", rp,
                 intensity_mm_h(15, rp), intensity_mm_h(30, rp), intensity_mm_h(60, rp))


if __name__ == "__main__":
    main()

# Ecotech III (Greater Noida) – conceptual storm sewer design

Screening-level storm sewer design and SWMM simulation for the Ecotech III area,
with a standalone 2D live map of the simulation. **Concept design only – not for construction.**

## Enter your own rainfall
Edit **`rainfall_input.csv`** (next to the scripts) before running steps 2-5:
```
return_period_yr,duration_min,intensity_mm_h
5,15,104.3
5,30,71.5
...
```
- Use an IDF table (e.g. IMD / CPHEEO values for Delhi-NCR): intensity in mm/h, or a `depth_mm` column instead.
- At least two durations per return period; any number of return periods. Values between durations are interpolated (log-log), outside your range extrapolated.
- Choose the return periods at the top of `ecotech3_design.py`: `DESIGN_RP` (sizing, default 5) and `CHECK_RP` (flood check, default 50); the storm length and peak position are `STORM_MIN` and `PEAK_FRAC`.
- The file shipped in this repo contains **placeholder values** (assumed CWC-scaled numbers, not official IMD data). Replace them.
- If the file is missing, the scripts fall back to the assumed CWC-scaled rainfall and print a warning.
- The rainfall actually used is saved to `out/rainfall_used.csv`.

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1 | `make_dem5m.py reprojected_eco.tif` | `dem5m.tif` (5 m DEM) |
| 2 | `ecotech3_design.py` | subcatchments, network, rational-method sizing, SWMM `.inp` files |
| 3 | `resize_loop.py` | pipes resized from SWMM 5.2 peak flows; final `.inp`, CSVs, GeoPackage |
| 4 | `add_outfall_layer.py` (optional) | `nodes` + `outfall` layers, GL/IL/cover columns on `pipes` |
| 5 | `make_live_map.py` | `out/ecotech3_live_map.html` (open in Chrome/Edge) |

`live_map_template.html` is only the HTML shell used by step 5 – do not open it directly.

## Quick start (Windows)
```
py -m pip install -r requirements.txt
py make_dem5m.py reprojected_eco.tif
py ecotech3_design.py
py resize_loop.py
py add_outfall_layer.py
py make_live_map.py
```
Place these input files next to the scripts (they are **not** in the repo):
- `reprojected_eco.tif` – 1 m DEM, EPSG:32643
- `Landuse_Masterplan.shp` with `.dbf`, `.shx`, `.prj`, `.cpg` – GNIDA master plan land use

## Key assumptions (change in the constants block of `ecotech3_design.py`)
- Outfall on the western boundary at (739512 E, 3158863 N, EPSG:32643).
- Subcatchments are an **assumed 350 m collector grid**; internal plot roads are not in the master plan.
- Default (placeholder) rainfall: 50-year 24-h point rainfall 240 mm read from CWC Subzone 1(e) Plate 9 (to be confirmed);
  CWC duration ratios (Fig. 10); the 5-year depth is the 50-year depth × 0.60 (**assumed**);
  depths below 1 h are **extrapolated**. Replace with IMD IDF values in `rainfall_input.csv`.
- Runoff coefficient C = 0.75 (industrial) / 0.60 (institutional); Manning n = 0.013;
  velocity limits 0.6–3.0 m/s; minimum cover 1.2 m. Check against the CPHEEO storm-water manual.
- The DEM is a surface model (buildings/mounds), smoothed at 150 m for ground levels – inverts are indicative only.

## Known limitations
- The network is deep (outfall invert about 10 m below ground) because the site is very flat; confirm receiving water level.
- After resizing, a few conduits still exceed 3 m/s in the dynamic run.
- The live map shows 1D results on a map; it is not a 2D overland-flow simulation.
- Results depend on the assumed grid (about ±10 % on peak flows between machines/library versions).

## Data
The land-use layer comes from the GNIDA master plan service; the DEM is user-supplied.
Check the licence/terms of both before redistributing them.

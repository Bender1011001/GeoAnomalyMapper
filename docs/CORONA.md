# CORONA: free 1960s ~2 m satellite imagery, in five lines of Python

`archaeo_intel/corona.py` reads the University of Arkansas CAST **CORONA
Atlas** archive — 217 declassified reconnaissance missions (1960-72, KH-4B
down to ~1.8 m) served as open HTTPS, no account — and turns raw film strips
into georeferenced GeoTIFFs you can drop into QGIS or Google Earth.

Why you'd care: the frames pre-date mechanized agriculture, dams, sprawl,
and recent conflict. Archaeological and landscape features long erased on
the ground are often crisply visible. Resolution is 4-15x finer than today's
free optical data (Sentinel-2, 10 m).

## Quick start

```python
from archaeo_intel import corona

corona.list_missions()                      # 217 mission ids
strips = corona.list_strips("1102-1025d")   # Dec 1967, 76 strips

url = strips[9]
img = corona.quicklook(url)                 # whole 217-km strip, ~2 s
crop = corona.read_window(url, 70000, 0, 14000, 7391, factor=4)
```

Nothing is downloaded in bulk: quicklooks come from the `.ovr` overview
pyramid via HTTP range reads; full-resolution windows cost only what you
crop (a full strip is ~800 MB — you almost never need one).

## Georeferencing (human-in-the-loop)

The strips are raw film scans with **no coordinates**. You supply 3+ ground
control points (GCPs) — landmarks you can identify in both the strip and any
modern map (road junctions, wadi confluences, building corners):

```python
g = None
g = corona.add_gcp(g, col=71204, row=3010, lon=39.6021, lat=36.6614, note="wadi fork")
g = corona.add_gcp(g, col=78990, row=1500, lon=39.7409, lat=36.6903)
g = corona.add_gcp(g, col=74450, row=5522, lon=39.6666, lat=36.6008)
# ... 6+ well-spread points recommended

rep = corona.fit_report(g, order=1)         # per-GCP residuals in meters
print(rep["residuals_m"], rep["rms_m"])     # one huge residual = a mis-click

img, bbox = corona.warp_to_grid(url, g, res_m=4.0)
corona.save_geotiff("site_1967.tif", img, bbox)   # EPSG:4326, QGIS-ready
```

Practical accuracy notes:

- KH-4B optics are **panoramic**: scale varies along the strip. An affine
  fit (`order=1`) is good over ~10-20 km windows; use `order=2` with 6+
  well-spread GCPs for larger areas, and prefer several local fits over one
  global fit.
- `fit_report` is your friend: a single GCP with residuals several times
  the others is a mis-click — delete it and refit.
- Expect ~1-3 native pixels (2-6 m) of local accuracy from a careful
  6-8-GCP affine over a 10 km window.

## CLI

```
python -m archaeo_intel.corona missions
python -m archaeo_intel.corona strips 1102-1025d
python -m archaeo_intel.corona quicklook <ntf_url> -o strip.png
python -m archaeo_intel.corona window <ntf_url> 70000 0 14000 7391 --factor 4 -o crop.png
python -m archaeo_intel.corona warp <ntf_url> gcps.json --res-m 4 -o site.tif
```

(`gcps.json` is written by `corona.save_gcps(path, gcps, ntf_url)`.)

## Finding coverage

The archive carries no footprint metadata, so coverage identification is a
quicklook exercise: montage a mission's strips (2 s each) and recognize
geography. Strips within a mission march ~15 km per strip number and span
~215 x 15 km each. (A worked example: mission 1102-1025d strip 016 contains
the unmistakable Jebel Abd al-Aziz anticline, which anchors the whole pass
across the Syrian Jazira.) Alternatively, USGS EarthExplorer's "Declass"
datasets offer footprint search with a free account.

## Data credit and license

Imagery is served by the **CORONA Atlas of the Middle East** project, CAST,
University of Arkansas (https://corona.cast.uark.edu) — credit them when you
use it (CC BY-SA per their site; the underlying USGS declassified film is
public domain), and cite: Casana, J. & Cothren, J., "The CORONA Atlas
Project", in *Mapping Archaeological Landscapes from Space* (2013).

Please be a good citizen of their bandwidth: prefer quicklooks and windowed
reads over bulk strip downloads.

## Known limitations

- No automatic georeferencing yet: matching 1967 film to modern imagery
  automatically is an open problem (three documented failed attempts live in
  the project notes); GCPs are human-picked for now.
- No footprint index: coverage discovery is visual (see above).
- Film artifacts (vignetting, frame joins, panoramic distortion) are left
  as-is; treat sub-pixel measurements with suspicion.

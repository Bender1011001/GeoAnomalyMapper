"""Regenerate the figures embedded in README.md.

Two figures, each answering one question a technical visitor asks in the first
ten seconds:

  wink_validation.png  "does it actually detect a real thing?"
  funnel.png           "does it scale, and does it filter?"

Run:  python tools/make_readme_figures.py
Both are committed to docs/img/ (whitelisted in .gitignore) so the repo shows
evidence without anyone needing credentials or a download.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Circle
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
TIF = ROOT / "data" / "insar_wink" / "wink_hires_stack_rate_m_yr.tif"
FIND = ROOT / "data" / "insar_wink" / "FINAL_deformation_findings.json"

# Published Wink Sink locations (Winkler County TX; public record).
WINK1 = (31.7772, -103.1119)   # collapsed 1980
WINK2 = (31.7625, -103.1003)   # collapsed 2002


def wink_validation() -> Path:
    """Two panels: regional context, then a zoom on the blind detection.

    The features are small (a fast bowl is tens of metres), so a single
    full-scene view renders them as invisible specks. Context + zoom is the
    honest way to show both that the scene is mostly quiet and that the hit is
    real.
    """
    f = json.loads(FIND.read_text())
    val, det = f["validation"], f["validation"]["detection"]

    with rasterio.open(TIF) as ds:
        rate = ds.read(1) * 100.0                     # m/yr -> cm/yr
        bounds, crs = ds.bounds, ds.crs
    # InSAR velocity is relative, so an absolute offset is meaningless.
    # Referencing to the scene median makes "negative = sinking relative to its
    # surroundings" literally true, which is the claim we actually make.
    rate = rate - np.nanmedian(rate)

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xy = lambda lat, lon: tf.transform(lon, lat)      # noqa: E731
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    dx, dy = xy(det["lat"], det["lon"])
    PAD = 2600.0                                      # zoom half-width, metres

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.6, 6.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.06, 1]})

    kw = dict(extent=extent, origin="upper", cmap="RdBu_r", vmin=-4, vmax=4,
              interpolation="nearest")
    axA.imshow(rate, **kw)
    im = axB.imshow(rate, **kw)

    # --- panel A: regional context -------------------------------------------
    axA.add_patch(plt.Rectangle((dx - PAD, dy - PAD), 2 * PAD, 2 * PAD,
                                fill=False, ec="#111", lw=1.8, ls="--", zorder=6))
    axA.annotate("panel B", (dx + PAD, dy + PAD), xytext=(8, 8),
                 textcoords="offset points", fontsize=9.5, fontweight="bold")
    for c, col in zip(f["unknown_candidates_hires_box"][:3],
                      ("#c81e1e", "#c81e1e", "#c81e1e")):
        cx, cy = xy(c["lat"], c["lon"])
        axA.add_patch(Circle((cx, cy), 340, fill=False, ec=col, lw=1.8, zorder=6))
    axA.annotate("3 uncatalogued bowls\n(−21.8, −18.5, −8.0 cm/yr)\nnot in any"
                 " public inventory",
                 xy=xy(*(31.7502, -103.0274)), xytext=(-6, -132),
                 textcoords="offset points", fontsize=9.2, color="#8c1010",
                 ha="center", zorder=9,
                 bbox=dict(fc="white", ec="#c81e1e", lw=1.1, alpha=.95, pad=3.2),
                 arrowprops=dict(arrowstyle="->", color="#c81e1e", lw=1.5))
    axA.set_title("A — full scene, 295 km²", fontsize=11.5, fontweight="bold")

    # --- panel B: the blind hit ----------------------------------------------
    for (lat, lon), lab, off in ((WINK1, "Wink Sink 1 (1980)", (13, 9)),
                                 (WINK2, "Wink Sink 2 (2002)", (13, -20))):
        x, y = xy(lat, lon)
        axB.plot(x, y, marker="^", ms=14, mfc="#ffd400", mec="k", mew=1.5, zorder=7)
        axB.annotate(lab, (x, y), xytext=off, textcoords="offset points",
                     fontsize=10, fontweight="bold", zorder=8,
                     bbox=dict(fc="white", ec="none", alpha=.82, pad=1.8))
    axB.add_patch(Circle((dx, dy), 300, fill=False, ec="#00a028", lw=3.2, zorder=9))
    axB.annotate(f"DETECTED BLIND — {det['rate_cm_yr']} cm/yr\n"
                 f"{val['offset_from_published_location_km']} km from the published\n"
                 "active subsidence area",
                 (dx, dy), xytext=(0, -150), textcoords="offset points",
                 fontsize=10.8, fontweight="bold", color="#00631a", ha="center",
                 zorder=10,
                 bbox=dict(fc="white", ec="#00a028", lw=1.6, alpha=.96, pad=4.5),
                 arrowprops=dict(arrowstyle="->", color="#00a028", lw=2.2))
    axB.set_xlim(dx - PAD, dx + PAD); axB.set_ylim(dy - PAD, dy + PAD)
    axB.set_title("B — zoom on the detection", fontsize=11.5, fontweight="bold")

    for a in (axA, axB):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#888")

    cb = fig.colorbar(im, ax=(axA, axB), shrink=.9, pad=.015, aspect=34)
    cb.set_label("line-of-sight velocity (cm/yr), scene-median referenced\n"
                 "negative = subsiding", fontsize=9.8)
    fig.suptitle("Ground-truth validation — Wink Sinks, Winkler County, TX\n"
                 "Sentinel-1 InSAR, 12 short pairs, 40 m. The detector was never "
                 "told where to look.",
                 fontsize=13.2, fontweight="bold")
    fig.text(.012, -.028,
             "Noise floor (robust σ) 1.0 cm/yr · 5 bowls in the 295 km² box · "
             "probability of a chance hit within 1 km ≈ 5% · the 1980/2002 rims "
             "read quiescent today, matching published post-2016 deceleration.",
             fontsize=8.5, color="#444")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "wink_validation.png"
    fig.savefig(p, dpi=125, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


def funnel() -> Path:
    stages = [("Raw anomalies\nfrom 357 tiles", 33458, "#b9c6d6"),
              ("Localized\n(Mogi: not regional aquifer)", 11138, "#8fa6bf"),
              ("Physically plausible\n(velocity and cumulative agree)", 4754, "#6d88a8"),
              ("Flagged by vision model\nfor full-res review", 90, "#456c94"),
              ("Survived human review", 1, "#c8501e")]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ys = np.arange(len(stages))[::-1]
    widths = [np.log10(max(n, 1)) + 1 for _, n, _ in stages]
    for y, (lab, n, c), w in zip(ys, stages, widths):
        ax.barh(y, w, height=.62, color=c, edgecolor="white", lw=1.5)
        ax.text(w + .09, y, f"{n:,}", va="center", ha="left",
                fontsize=14, fontweight="bold", color="#20303f")
        ax.text(-.12, y, lab, va="center", ha="right", fontsize=10.2, color="#20303f")
    ax.set_xlim(-0.02, max(widths) + 1.5); ax.set_ylim(-.7, len(stages) - .3)
    ax.axis("off")
    ax.set_title("Automated triage over the four-region arid sweep\n"
                 "33,458 raw detections reduced to one candidate worth a human's "
                 "time — for $0.25 of model inference",
                 fontsize=12.5, fontweight="bold", loc="left", pad=15)
    ax.text(0, -.62, "Bar length is log-scaled. The single survivor resolved to an "
            "active evaporite-karst collapse in Nash Draw, NM — a documented "
            "process. Zero unexplained detections is the correct outcome for a "
            "working detector.",
            transform=ax.transData, fontsize=8.6, color="#555")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "funnel.png"
    fig.savefig(p, dpi=125, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (wink_validation, funnel):
        print("wrote", fn())

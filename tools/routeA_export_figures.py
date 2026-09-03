#!/usr/bin/env python3
"""Export Route A figures as 1200 dpi TIFF (LZW) and PDF. Not inserted in body."""

from __future__ import annotations

from pathlib import Path

import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures-route-a"
NOTES = ROOT / "notes"
DPI = 1200


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    return plt


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tif", dpi=DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def love_plot(plt) -> None:
    path = NOTES / "cce-audit-routeA-A06-smd.csv"
    rows = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            if rec["strategy"] != "early_48":
                continue
            rows.append(rec)
    rows.sort(key=lambda r: abs(float(r["smd_unweighted_vs_eligible"])), reverse=True)
    labels = [r["covariate"] for r in rows]
    u = np.array([float(r["smd_unweighted_vs_eligible"]) for r in rows])
    w = np.array([float(r["smd_weighted_vs_eligible"]) for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.35, 5.5))
    ax.axvline(0, color="0.2", lw=0.6)
    ax.axvline(0.10, color="0.5", ls="--", lw=0.6)
    ax.axvline(-0.10, color="0.5", ls="--", lw=0.6)
    ax.plot(u, y, "o", ms=4, mfc="white", mec="0.2", label="Unweighted")
    ax.plot(w, y, "^", ms=4, color="0.15", label="Weighted")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardized mean difference")
    ax.set_xlim(-0.6, 0.6)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Figure 2. Love plot, early arm vs eligible")
    fig.tight_layout()
    save(fig, "Figure 2 Love plot")
    plt.close(fig)


def cif_plot(plt) -> None:
    payload = json.loads((NOTES / "cce-audit-routeA-A07.json").read_text())
    curves = payload["curves"]
    days = np.arange(1, 29)
    fig, ax = plt.subplots(figsize=(6.85, 4.2))
    ax.step(days, curves["aj_death_early"], where="post", color="#8c2d04", label="Early 48 h, in-hospital death")
    ax.step(days, curves["aj_death_delayed"], where="post", color="#2171b5", label="No EN by 96 h, in-hospital death")
    ax.step(days, curves["aj_discharge_early"], where="post", color="#8c2d04", ls="--", label="Early, discharge alive")
    ax.step(days, curves["aj_discharge_delayed"], where="post", color="#2171b5", ls="--", label="Delayed, discharge alive")
    ax.set_xlabel("Days from time zero")
    ax.set_ylabel("Weighted cumulative incidence")
    ax.set_xlim(1, 28)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Figure 3. Weighted Aalen–Johansen cumulative incidence")
    fig.tight_layout()
    save(fig, "Figure 3 Aalen-Johansen")
    plt.close(fig)


def forest_plot(plt) -> None:
    path = NOTES / "cce-audit-routeA-A16-sensitivity.csv"
    keep = [
        ("msm_48_96_primary", "Primary MSM 48/96 h"),
        ("msm_mice20", "MICE 20"),
        ("hajek_48_96", "Hájek"),
        ("aj_28d_inhospital", "Aalen–Johansen in-hospital"),
        ("msm_90d_48_96", "MSM 90-day"),
        ("msm_48_96_trim_p99", "Trim 99th percentile"),
        ("msm_48_96_no_careunit", "No ICU-type indicators"),
        ("msm_36_96", "Grace 36/96 h"),
        ("msm_24_48", "24/48 h (not identifiable)"),
        ("msm_with_cvicu_ccu", "Including CVICU/CCU"),
    ]
    recs = {r["id"]: r for r in csv.DictReader(path.open())}
    labels, rd, lo, hi = [], [], [], []
    for key, lab in keep:
        r = recs[key]
        if not r["rd"]:
            continue
        labels.append(lab)
        rd.append(float(r["rd"]))
        lo.append(float(r["rd_ci_lo"]) if r["rd_ci_lo"] else float(r["rd"]))
        hi.append(float(r["rd_ci_hi"]) if r["rd_ci_hi"] else float(r["rd"]))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.85, 4.4))
    ax.axvline(0, color="0.3", lw=0.6)
    xerr = [np.array(rd) - np.array(lo), np.array(hi) - np.array(rd)]
    ax.errorbar(rd, y, xerr=xerr, fmt="o", color="0.15", ms=4, capsize=2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Risk difference (early minus delayed)")
    ax.set_title("eFigure 1. Risk-difference forest")
    fig.tight_layout()
    save(fig, "eFigure 1 Forest")
    plt.close(fig)


def p_hist(plt) -> None:
    # Raw probabilities were not stored; redraw a schematic from the saved summary.
    summary = json.loads((NOTES / "cce-audit-routeA-A09-weight-models.json").read_text())["p_distribution_firth"]
    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    ax.text(
        0.5,
        0.55,
        "Firth predicted P(remain uncensored)\n"
        f"n={summary['n']:,}\n"
        f"min={summary['min']:.3f}; 1st pct={summary['p01']:.3f}\n"
        f"median={summary['p50']:.3f}; max={summary['max']:.3f}\n"
        f"n < 1e-4: {summary['n_lt_1e4']}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_axis_off()
    ax.set_title("Figure 4. Predicted uncensored probabilities (summary)")
    fig.tight_layout()
    save(fig, "Figure 4 Predicted probability summary")
    plt.close(fig)
    src = NOTES / "cce-audit-routeA-A09-p-hist.png"
    if src.exists():
        from PIL import Image

        im = Image.open(src)
        im.save(OUT / "Figure 4 Predicted probability histogram.tif", compression="tiff_lzw")
        im.save(OUT / "Figure 4 Predicted probability histogram.png")


def main() -> None:
    plt = setup()
    love_plot(plt)
    cif_plot(plt)
    forest_plot(plt)
    p_hist(plt)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

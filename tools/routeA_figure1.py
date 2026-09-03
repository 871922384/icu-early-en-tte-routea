#!/usr/bin/env python3
"""Figure 1 clone-censor flow. Counts from Table 3. No effect estimates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures-route-a"
CCE = OUT / "cce"
DPI = 1200


def box(ax, x, y, w, h, text, fc="white"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8,
        edgecolor="0.15",
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, fontname="DejaVu Sans", color="0.1")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.7, mutation_scale=8),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CCE.mkdir(parents=True, exist_ok=True)
    fig_w = 6.85
    fig_h = 8.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    plt.rcParams["font.family"] = "DejaVu Sans"

    box(ax, 2.7, 11.1, 4.6, 0.7, "Eligible first ICU stays\nn = 11,259")
    arrow(ax, 5.0, 11.1, 5.0, 10.55)

    box(ax, 0.3, 9.55, 4.4, 0.85, "Early-EN clone\nInitiate EN by 48 h\nn = 11,259", fc="#f7f7f7")
    box(ax, 5.3, 9.55, 4.4, 0.85, "Delayed clone\nNo EN before 96 h\nn = 11,259", fc="#f7f7f7")
    arrow(ax, 2.5, 9.55, 2.5, 8.95)
    arrow(ax, 7.5, 9.55, 7.5, 8.95)

    box(ax, 0.3, 8.15, 4.4, 0.7, "Artificially censored\nn = 7,582")
    box(ax, 5.3, 8.15, 4.4, 0.7, "Artificially censored\nn = 3,933")
    arrow(ax, 2.5, 8.15, 2.5, 7.55)
    arrow(ax, 7.5, 8.15, 7.5, 7.55)

    box(ax, 0.3, 6.75, 4.4, 0.7, "Uncensored\nn = 3,677", fc="#deebf7")
    box(ax, 5.3, 6.75, 4.4, 0.7, "Uncensored\nn = 7,326", fc="#deebf7")

    # splits
    arrow(ax, 1.1, 6.75, 0.9, 6.15)
    arrow(ax, 2.5, 6.75, 2.5, 6.15)
    arrow(ax, 3.9, 6.75, 4.1, 6.15)
    box(ax, 0.15, 5.15, 1.5, 0.95, "Initiated EN\nn = 2,599\n(70.7%)")
    box(ax, 1.75, 5.15, 1.5, 0.95, "Died in grace\nbefore EN\nn = 1,085\n(29.5%)")
    box(ax, 3.35, 5.15, 1.5, 0.95, "Discharged\nin grace\nn = 14")

    arrow(ax, 5.85, 6.75, 5.7, 6.15)
    arrow(ax, 7.0, 6.75, 6.9, 6.15)
    arrow(ax, 8.15, 6.75, 8.1, 6.15)
    arrow(ax, 9.15, 6.75, 9.25, 6.15)
    box(ax, 5.15, 5.15, 1.15, 0.95, "Initiated\nEN\nn = 10")
    box(ax, 6.35, 5.15, 1.15, 0.95, "Died in\ngrace\nn = 1,342")
    box(ax, 7.55, 5.15, 1.15, 0.95, "Discharged\nin grace\nn = 1,327")
    box(ax, 8.75, 5.15, 1.15, 0.95, "Never EN,\nsurvived\nn = 5,982")

    arrow(ax, 2.5, 5.15, 2.5, 4.55)
    arrow(ax, 7.5, 5.15, 7.5, 4.55)
    box(
        ax,
        0.3,
        3.35,
        4.4,
        1.1,
        "Analyzed (early)\nSum of weights = 3,348\nKish ESS = 2,650",
        fc="#fff7bc",
    )
    box(
        ax,
        5.3,
        3.35,
        4.4,
        1.1,
        "Analyzed (delayed)\nSum of weights = 7,008\nKish ESS = 6,353",
        fc="#fff7bc",
    )
    ax.text(
        5.0,
        2.7,
        "Clone counts are not mutually exclusive across arms.\n"
        "Grace-period deaths are compatible with both strategies.\n"
        "Split boxes among uncensored clones are descriptive and not a partition.",
        ha="center",
        va="top",
        fontsize=7,
        color="0.25",
    )
    fig.tight_layout()
    stem = "Figure 1 Clone flow"
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tif", dpi=DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(CCE / "Figure1.tif", dpi=DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(CCE / "Figure1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", CCE / "Figure1.tif")


if __name__ == "__main__":
    main()

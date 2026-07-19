"""
style.py
========

Shared figure styling so every plot in the repo looks consistent and
publication-ready (serif-free sans, consistent DPI, consistent palette).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "control": "#3B76D6",
    "stress": "#D6483B",
    "up": "#D6483B",
    "down": "#3B76D6",
    "ns": "#B0B0B0",
}


def apply_style() -> None:
    """Apply repo-wide matplotlib/seaborn styling. Call once before plotting."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "font.family": "sans-serif",
    })


def save_figure(fig, name: str, figures_dir, dpi: int = 300) -> None:
    """Save fig as both PNG and PDF under figures_dir/{png,pdf}/name.{ext}."""
    png_dir = figures_dir / "png"
    pdf_dir = figures_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")

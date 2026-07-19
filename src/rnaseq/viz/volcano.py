"""
volcano.py
==========

Volcano plot: log2FoldChange (x) vs -log10(padj) (y). Standard DE visual —
shows effect size and significance simultaneously.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .style import PALETTE, apply_style, save_figure


def plot_volcano(
    results_df: pd.DataFrame,
    alpha: float,
    lfc_threshold: float,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "volcano_plot",
):
    """
    results_df: DESeq2 results with log2FoldChange, padj columns.
    Points colored by significance status (up / down / not significant).
    Saves PNG + PDF, returns the figure.
    """
    apply_style()
    df = results_df.copy()
    df = df.dropna(subset=["padj", "log2FoldChange"])
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    is_sig = (df["padj"] < alpha) & (df["log2FoldChange"].abs() >= lfc_threshold)
    status = np.where(
        is_sig & (df["log2FoldChange"] > 0), "up",
        np.where(is_sig & (df["log2FoldChange"] < 0), "down", "ns"),
    )
    df["status"] = status

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [("ns", PALETTE["ns"]), ("down", PALETTE["down"]), ("up", PALETTE["up"])]:
        subset = df[df["status"] == label]
        ax.scatter(
            subset["log2FoldChange"], subset["neg_log10_padj"],
            s=14, alpha=0.6, color=color,
            label=f"{label} (n={len(subset)})",
        )

    ax.axhline(-np.log10(alpha), color="gray", linestyle="--", linewidth=1)
    ax.axvline(lfc_threshold, color="gray", linestyle="--", linewidth=1)
    ax.axvline(-lfc_threshold, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("log2 Fold Change (stress vs control)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Volcano Plot — Differential Expression")
    ax.legend(loc="upper right", markerscale=2)

    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig

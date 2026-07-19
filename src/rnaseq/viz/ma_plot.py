"""
ma_plot.py
==========

MA plot: mean expression (x, log scale) vs log2FoldChange (y). Reveals
fold-change bias at low expression levels — standard DESeq2 diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .style import PALETTE, apply_style, save_figure


def plot_ma(
    results_df: pd.DataFrame,
    alpha: float,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "ma_plot",
):
    """
    results_df: DESeq2 results with baseMean, log2FoldChange, padj columns.
    Significant genes (padj < alpha) highlighted.
    """
    apply_style()
    df = results_df.copy()
    df = df.dropna(subset=["baseMean", "log2FoldChange"])
    df = df[df["baseMean"] > 0]

    is_sig = df["padj"] < alpha

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df.loc[~is_sig, "baseMean"], df.loc[~is_sig, "log2FoldChange"],
        s=10, alpha=0.4, color=PALETTE["ns"], label=f"not significant (n={(~is_sig).sum()})",
    )
    ax.scatter(
        df.loc[is_sig, "baseMean"], df.loc[is_sig, "log2FoldChange"],
        s=14, alpha=0.7, color=PALETTE["up"], label=f"significant (n={is_sig.sum()})",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Mean of normalized counts (log scale)")
    ax.set_ylabel("log2 Fold Change")
    ax.set_title("MA Plot")
    ax.legend(loc="upper right", markerscale=2)

    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig

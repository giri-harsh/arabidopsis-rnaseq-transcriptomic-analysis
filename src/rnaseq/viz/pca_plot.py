"""
pca_plot.py
===========

Renders PCA scatter from ml/pca.py output. Colored by condition, annotated
with % variance explained per axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .style import PALETTE, apply_style, save_figure


def plot_pca(
    pca_coords: pd.DataFrame,
    explained_variance_ratio,
    metadata: pd.DataFrame,
    condition_column: str,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "pca_plot",
):
    """
    pca_coords: sample x component DataFrame (e.g. PC1, PC2 columns), indexed by sample id.
    explained_variance_ratio: array-like, fraction of variance per component.
    metadata: sample metadata for condition coloring.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    conditions = metadata.loc[pca_coords.index, condition_column]
    for condition in conditions.unique():
        mask = conditions == condition
        ax.scatter(
            pca_coords.loc[mask, "PC1"], pca_coords.loc[mask, "PC2"],
            s=90, alpha=0.85, color=PALETTE.get(condition, "#555555"),
            label=condition, edgecolor="white", linewidth=0.5,
        )

    for sample_id, row in pca_coords.iterrows():
        ax.annotate(sample_id, (row["PC1"], row["PC2"]), fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance)")
    ax.set_title("Principal Component Analysis — Samples")
    ax.legend(title=None, loc="best")

    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig

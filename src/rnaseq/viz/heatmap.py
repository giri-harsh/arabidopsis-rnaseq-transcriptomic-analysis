"""
heatmap.py
==========

Clustered heatmap of top significant DEGs (row-z-scored normalized counts),
hierarchically clustered on both axes. Standard way to show DEG expression
patterns across samples/conditions at a glance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from .style import PALETTE, apply_style


def plot_deg_heatmap(
    normalized_counts: pd.DataFrame,
    significant_genes: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_column: str,
    top_n: int,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "deg_heatmap",
):
    """
    normalized_counts: gene x sample (filtered/normalized).
    significant_genes: DESeq2 results subset (index = gene_id), sorted by padj.
    metadata: sample metadata for the condition color bar.
    top_n: number of top (lowest padj) significant genes to display.
    """
    apply_style()

    gene_ids = significant_genes.index[:top_n]
    gene_ids = [g for g in gene_ids if g in normalized_counts.index]

    if len(gene_ids) < 2:
        raise ValueError("Need at least 2 significant genes present in normalized_counts to plot heatmap.")

    expr = normalized_counts.loc[gene_ids]
    log_expr = np.log2(expr + 1)
    z_expr = log_expr.sub(log_expr.mean(axis=1), axis=0).div(log_expr.std(axis=1).replace(0, 1), axis=0)

    condition_colors = metadata[condition_column].map(
        lambda c: PALETTE.get(c, "#888888")
    )

    g = sns.clustermap(
        z_expr,
        cmap="RdBu_r",
        center=0,
        col_colors=condition_colors,
        figsize=(10, max(6, 0.18 * len(gene_ids))),
        yticklabels=len(gene_ids) <= 60,
        xticklabels=True,
        cbar_kws={"label": "z-score (log2 normalized expression)"},
    )
    g.fig.suptitle(f"Clustered Heatmap — Top {len(gene_ids)} Significant DEGs", y=1.02)

    figures_dir = Path(figures_dir)
    (figures_dir / "png").mkdir(parents=True, exist_ok=True)
    (figures_dir / "pdf").mkdir(parents=True, exist_ok=True)
    g.savefig(figures_dir / "png" / f"{name}.png", dpi=dpi, bbox_inches="tight")
    g.savefig(figures_dir / "pdf" / f"{name}.pdf", bbox_inches="tight")
    return g

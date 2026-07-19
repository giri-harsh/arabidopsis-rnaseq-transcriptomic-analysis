"""
qc_plots.py
===========

QC visuals: library size bars, expression distribution, boxplots
before/after normalization, sample correlation heatmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import PALETTE, apply_style, save_figure


def plot_library_sizes(
    library_sizes: pd.Series,
    metadata: pd.DataFrame,
    condition_column: str,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "library_sizes",
):
    """Bar chart of total raw read count per sample, colored by condition."""
    apply_style()
    conditions = metadata.loc[library_sizes.index, condition_column]
    colors = conditions.map(lambda c: PALETTE.get(c, "#555555"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(library_sizes.index, library_sizes.values, color=colors)
    ax.set_ylabel("Total raw read count")
    ax.set_title("Library Size per Sample")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig


def plot_normalization_boxplots(
    raw_counts: pd.DataFrame,
    normalized_counts: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "normalization_boxplots",
):
    """Side-by-side log2(count+1) boxplots before/after size-factor normalization."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    log_raw = np.log2(raw_counts + 1)
    log_norm = np.log2(normalized_counts + 1)

    axes[0].boxplot(log_raw.values, labels=log_raw.columns, showfliers=False)
    axes[0].set_title("Before Normalization")
    axes[0].set_ylabel("log2(count + 1)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].boxplot(log_norm.values, labels=log_norm.columns, showfliers=False)
    axes[1].set_title("After Size-Factor Normalization")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Count Distribution Before vs After Normalization")
    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig


def plot_expression_distribution(
    normalized_counts: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "expression_distribution",
):
    """Density plot of log2(normalized count + 1) per sample — overall expression shape."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    log_counts = np.log2(normalized_counts + 1)
    for col in log_counts.columns:
        sns.kdeplot(log_counts[col], ax=ax, alpha=0.6, linewidth=1.2, label=col)
    ax.set_xlabel("log2(normalized count + 1)")
    ax.set_title("Expression Distribution per Sample")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig


def plot_sample_correlation(
    correlation_matrix: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 300,
    name: str = "sample_correlation",
):
    """Heatmap of the sample-sample correlation matrix."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                vmin=correlation_matrix.values.min(), vmax=1.0, square=True)
    ax.set_title("Sample-Sample Correlation (Spearman)")
    fig.tight_layout()
    save_figure(fig, name, figures_dir, dpi=dpi)
    return fig

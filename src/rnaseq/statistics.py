"""
statistics.py
=============

Post-DESeq2 statistical summaries: significant DEG extraction, summary
counts, sample correlation matrix. Consumed by report.py and viz modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DEGSummary:
    """
    full_table: complete DESeq2 results (all tested genes).
    significant: subset passing padj < alpha AND |log2FC| >= lfc_threshold.
    upregulated / downregulated: significant split by direction.
    n_tested, n_significant, n_up, n_down: counts.
    """

    full_table: pd.DataFrame
    significant: pd.DataFrame
    upregulated: pd.DataFrame
    downregulated: pd.DataFrame

    @property
    def n_tested(self) -> int:
        return int(self.full_table["pvalue"].notna().sum())

    @property
    def n_significant(self) -> int:
        return self.significant.shape[0]

    @property
    def n_up(self) -> int:
        return self.upregulated.shape[0]

    @property
    def n_down(self) -> int:
        return self.downregulated.shape[0]


def extract_significant_degs(
    results_df: pd.DataFrame,
    config: PipelineConfig,
) -> DEGSummary:
    """
    Apply significance filter: padj < alpha AND |log2FoldChange| >= lfc_threshold.

    Genes with NA padj (e.g. filtered by DESeq2's independent filtering or
    outlier handling) are excluded from the significant set but retained in
    full_table.
    """
    df = results_df.copy()
    valid = df["padj"].notna()

    is_sig = valid & (df["padj"] < config.alpha) & (df["log2FoldChange"].abs() >= config.lfc_threshold)
    significant = df.loc[is_sig].sort_values("padj")

    upregulated = significant.loc[significant["log2FoldChange"] > 0]
    downregulated = significant.loc[significant["log2FoldChange"] < 0]

    logger.info(
        "DEG summary: %d significant (padj<%.3f, |log2FC|>=%.2f) — %d up, %d down",
        significant.shape[0], config.alpha, config.lfc_threshold,
        upregulated.shape[0], downregulated.shape[0],
    )

    return DEGSummary(
        full_table=df,
        significant=significant,
        upregulated=upregulated,
        downregulated=downregulated,
    )


def sample_correlation_matrix(normalized_counts: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """
    Sample-sample correlation on log-transformed normalized counts.
    Used as a QC diagnostic: replicates should cluster (high correlation).
    """
    log_counts = np.log2(normalized_counts + 1)
    return log_counts.corr(method=method)


def summary_statistics(deg_summary: DEGSummary, preprocessing_n_removed: int, total_genes: int) -> dict:
    """Compact dict of headline numbers for the report / JSON export."""
    return {
        "total_genes_input": int(total_genes),
        "genes_removed_low_count": int(preprocessing_n_removed),
        "genes_tested": deg_summary.n_tested,
        "significant_degs": deg_summary.n_significant,
        "upregulated": deg_summary.n_up,
        "downregulated": deg_summary.n_down,
    }

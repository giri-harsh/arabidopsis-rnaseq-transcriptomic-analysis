"""
deseq_pipeline.py
=================

PyDESeq2 wrapper: dispersion estimation → Wald test → BH FDR.

Uses raw (unfiltered-by-us-beyond-basic) counts + design ~condition.
PyDESeq2 internally: computes its own size factors, fits per-gene
dispersions (gene-wise → trend → MAP shrinkage, DESeq2 methodology),
fits NB GLM, runs Wald test per gene, applies BH correction on p-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from .config import PipelineConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DESeqResult:
    """
    dds: fitted DeseqDataSet (holds normalized counts, dispersions, size factors).
    stats: fitted DeseqStats (Wald test results).
    results_df: gene table — baseMean, log2FoldChange, lfcSE, stat, pvalue, padj.
    contrast: (factor, test_level, reference_level) used for the Wald test.
    """

    dds: DeseqDataSet
    stats: DeseqStats
    results_df: pd.DataFrame
    contrast: tuple


def run_deseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> DESeqResult:
    """
    Run full DESeq2 workflow via PyDESeq2.

    counts: gene x sample raw integer counts (genes as rows).
    metadata: sample x fields, indexed by sample id, matching counts.columns.
    config.condition_column: design factor name.
    config.reference_level: baseline level for the Wald contrast.

    Returns DESeqResult with a results_df sorted by padj ascending.
    """
    # PyDESeq2 expects samples as rows, genes as columns.
    counts_for_dds = counts.T
    condition_col = config.condition_column
    levels = sorted(metadata[condition_col].unique().tolist())

    if config.reference_level not in levels:
        raise ValueError(
            f"reference_level '{config.reference_level}' not among condition "
            f"levels {levels}. Set config.reference_level to one of these."
        )
    test_level = next(l for l in levels if l != config.reference_level)

    logger.info(
        "Fitting DESeq2 model: design=~%s, reference=%s, test=%s, genes=%d, samples=%d",
        condition_col, config.reference_level, test_level,
        counts_for_dds.shape[1], counts_for_dds.shape[0],
    )

    dds = DeseqDataSet(
        counts=counts_for_dds,
        metadata=metadata,
        design=f"~{condition_col}",
        refit_cooks=True,
    )
    dds.deseq2()  # size factors -> gene-wise dispersion -> trend fit -> MAP dispersion -> GLM fit

    contrast = [condition_col, test_level, config.reference_level]
    stats = DeseqStats(dds, contrast=contrast, alpha=config.alpha)
    stats.summary()  # Wald test + BH correction (padj)

    results_df = stats.results_df.copy()
    results_df = results_df.sort_values("padj", na_position="last")
    results_df.index.name = "gene_id"

    n_tested = results_df["pvalue"].notna().sum()
    n_sig = (results_df["padj"] < config.alpha).sum()
    logger.info(
        "DESeq2 complete: %d genes tested, %d significant at padj<%.3f",
        n_tested, n_sig, config.alpha,
    )

    return DESeqResult(dds=dds, stats=stats, results_df=results_df, contrast=tuple(contrast))

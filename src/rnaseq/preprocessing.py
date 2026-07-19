"""
preprocessing.py
================

Filter low-count genes → compute library sizes → normalize → QC checks.

Why: raw counts unusable directly for DE. Low-count genes → unreliable
dispersion. Library size differs per sample → must normalize before
comparing expression across samples. QC before/after normalization proves
normalization worked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """
    Bundle of preprocessing outputs.

    filtered_counts: gene x sample raw counts, low-count genes removed.
    size_factors: per-sample normalization factors (median-of-ratios, DESeq2-style).
    normalized_counts: filtered_counts / size_factors (for QC/visualization only —
        PyDESeq2 recomputes its own size factors internally for the GLM fit).
    library_sizes: raw total counts per sample, pre-filtering.
    n_genes_removed: genes dropped by the filter.
    """

    filtered_counts: pd.DataFrame
    size_factors: pd.Series
    normalized_counts: pd.DataFrame
    library_sizes: pd.Series
    n_genes_removed: int


def compute_library_sizes(counts: pd.DataFrame) -> pd.Series:
    """Total raw read count per sample. Basic sequencing-depth QC metric."""
    return counts.sum(axis=0)


def filter_low_count_genes(
    counts: pd.DataFrame,
    min_total_count: int = 10,
    min_samples_expressed: int = 2,
) -> pd.DataFrame:
    """
    Drop genes with insufficient signal for reliable dispersion estimation.

    Kept if: sum(counts) >= min_total_count AND
             (#samples with count > 0) >= min_samples_expressed.
    """
    total_ok = counts.sum(axis=1) >= min_total_count
    expressed_ok = (counts > 0).sum(axis=1) >= min_samples_expressed
    keep = total_ok & expressed_ok

    n_before = counts.shape[0]
    filtered = counts.loc[keep]
    n_removed = n_before - filtered.shape[0]

    logger.info(
        "Gene filter: kept %d / %d genes (removed %d, %.1f%%)",
        filtered.shape[0], n_before, n_removed,
        100 * n_removed / n_before if n_before else 0.0,
    )
    return filtered


def compute_size_factors(counts: pd.DataFrame) -> pd.Series:
    """
    Median-of-ratios normalization (DESeq2's own method).

    Steps:
      1. geometric mean of each gene across all samples (reference row)
      2. per-sample per-gene ratio = count / reference
      3. size factor = median of ratios across genes (finite, ref>0 only)

    Robust to a handful of extremely highly-expressed genes, unlike simple
    total-count (library size) normalization.
    """
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_means = log_counts.mean(axis=1)

    ratios = log_counts.sub(log_geo_means, axis=0)
    finite_ratios = ratios[np.isfinite(log_geo_means)]

    size_factors = np.exp(finite_ratios.median(axis=0))
    size_factors = size_factors.fillna(1.0)

    logger.info("Computed size factors: %s", size_factors.round(3).to_dict())
    return size_factors


def normalize_counts(counts: pd.DataFrame, size_factors: pd.Series) -> pd.DataFrame:
    """Divide each sample's counts by its size factor. QC/visualization only."""
    return counts.div(size_factors, axis=1)


def run_preprocessing(
    counts: pd.DataFrame,
    config: PipelineConfig,
) -> PreprocessingResult:
    """Full preprocessing pipeline: filter → library sizes → size factors → normalize."""
    library_sizes = compute_library_sizes(counts)

    filtered = filter_low_count_genes(
        counts,
        min_total_count=config.min_total_count,
        min_samples_expressed=config.min_samples_expressed,
    )

    size_factors = compute_size_factors(filtered)
    normalized = normalize_counts(filtered, size_factors)

    return PreprocessingResult(
        filtered_counts=filtered,
        size_factors=size_factors,
        normalized_counts=normalized,
        library_sizes=library_sizes,
        n_genes_removed=counts.shape[0] - filtered.shape[0],
    )

"""Unit tests for rnaseq.statistics — DEG extraction logic."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rnaseq.config import PipelineConfig
from rnaseq.statistics import extract_significant_degs, sample_correlation_matrix, summary_statistics


def _fake_config(alpha=0.05, lfc_threshold=1.0):
    return PipelineConfig(
        counts_path="x", metadata_path="y",
        alpha=alpha, lfc_threshold=lfc_threshold,
    )


def _fake_results_df():
    return pd.DataFrame({
        "baseMean": [100, 50, 200, 10, 300],
        "log2FoldChange": [2.5, -3.0, 0.2, 1.5, -0.5],
        "pvalue": [0.0001, 0.001, 0.4, 0.15, 0.02],
        "padj": [0.001, 0.01, 0.5, 0.2, 0.03],
    }, index=["gA", "gB", "gC", "gD", "gE"])


def test_extract_significant_degs_filters_correctly():
    df = _fake_results_df()
    config = _fake_config(alpha=0.05, lfc_threshold=1.0)
    summary = extract_significant_degs(df, config)

    # gA (padj=0.001, lfc=2.5) -> sig up
    # gB (padj=0.01, lfc=-3.0) -> sig down
    # gC (padj=0.5) -> not sig (padj too high)
    # gD (padj=0.2) -> not sig (padj too high)
    # gE (padj=0.03, lfc=-0.5) -> not sig (lfc below threshold)
    assert set(summary.significant.index) == {"gA", "gB"}
    assert summary.n_up == 1
    assert summary.n_down == 1
    assert summary.n_significant == 2


def test_extract_significant_degs_handles_nan_padj():
    df = _fake_results_df()
    df.loc["gA", "padj"] = float("nan")
    config = _fake_config()
    summary = extract_significant_degs(df, config)
    assert "gA" not in summary.significant.index


def test_sample_correlation_matrix_diagonal_is_one():
    counts = pd.DataFrame({
        "s1": [10, 20, 30, 5],
        "s2": [12, 18, 33, 4],
        "s3": [100, 5, 2, 90],
    })
    corr = sample_correlation_matrix(counts)
    assert corr.shape == (3, 3)
    for sample in corr.columns:
        assert abs(corr.loc[sample, sample] - 1.0) < 1e-9


def test_summary_statistics_dict_shape():
    df = _fake_results_df()
    config = _fake_config()
    summary = extract_significant_degs(df, config)
    stats = summary_statistics(summary, preprocessing_n_removed=3, total_genes=8)
    assert stats["total_genes_input"] == 8
    assert stats["genes_removed_low_count"] == 3
    assert stats["significant_degs"] == summary.n_significant

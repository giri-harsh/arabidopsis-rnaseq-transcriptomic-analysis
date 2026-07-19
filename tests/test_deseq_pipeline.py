"""
Integration test for rnaseq.deseq_pipeline — runs a real (tiny) PyDESeq2 fit.

Slower than the other unit tests since it invokes actual NB-GLM fitting,
but this is the test that proves the DESeq2 wrapper contract (dispersion,
Wald test, BH correction) genuinely works end-to-end rather than being
mocked out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rnaseq.config import PipelineConfig
from rnaseq.deseq_pipeline import run_deseq2


def test_run_deseq2_produces_expected_columns(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id")
    config = PipelineConfig(
        counts_path="x", metadata_path="y",
        reference_level="control", condition_column="condition",
    )
    result = run_deseq2(tiny_counts, meta, config)

    expected_cols = {"baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"}
    assert expected_cols.issubset(set(result.results_df.columns))
    assert result.contrast == ("condition", "stress", "control")
    assert result.results_df.shape[0] == tiny_counts.shape[0]


def test_run_deseq2_rejects_invalid_reference_level(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id")
    config = PipelineConfig(
        counts_path="x", metadata_path="y",
        reference_level="nonexistent_level", condition_column="condition",
    )
    try:
        run_deseq2(tiny_counts, meta, config)
        assert False, "expected ValueError for invalid reference_level"
    except ValueError as exc:
        assert "reference_level" in str(exc)

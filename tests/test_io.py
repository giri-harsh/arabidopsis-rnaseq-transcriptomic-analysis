"""Unit tests for rnaseq.io — validation logic must actually fire on bad input."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rnaseq.io import (
    align_counts_and_metadata,
    load_counts,
    load_metadata,
    DataValidationError,
)


def test_align_valid_data(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id")
    ds = align_counts_and_metadata(tiny_counts, meta, condition_column="condition")
    assert ds.n_genes == 6
    assert ds.n_samples == 6
    assert list(ds.counts.columns) == list(meta.index)


def test_align_rejects_sample_mismatch(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id")
    bad_meta = meta.rename(index={"s3": "s99"})
    with pytest.raises(DataValidationError):
        align_counts_and_metadata(tiny_counts, bad_meta, condition_column="condition")


def test_align_rejects_missing_condition_column(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id")
    with pytest.raises(DataValidationError):
        align_counts_and_metadata(tiny_counts, meta, condition_column="nonexistent")


def test_align_rejects_single_condition_level(tiny_counts, tiny_metadata):
    meta = tiny_metadata.set_index("sample_id").copy()
    meta["condition"] = "control"  # collapse to one level
    with pytest.raises(DataValidationError):
        align_counts_and_metadata(tiny_counts, meta, condition_column="condition")


def test_load_counts_rejects_negative_values(tmp_path):
    df = pd.DataFrame({"gene_id": ["g1", "g2"], "s1": [5, -3], "s2": [2, 4]})
    path = tmp_path / "bad_counts.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_counts(path)


def test_load_counts_rejects_non_integer_majority(tmp_path):
    # Simulate someone accidentally passing normalized (TPM-like) values.
    df = pd.DataFrame({
        "gene_id": ["g1", "g2", "g3", "g4"],
        "s1": [5.234, 2.771, 9.512, 1.003],
        "s2": [4.812, 3.331, 8.220, 1.501],
    })
    path = tmp_path / "tpm_counts.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_counts(path)


def test_load_counts_rejects_duplicate_gene_ids(tmp_path):
    df = pd.DataFrame({"gene_id": ["g1", "g1"], "s1": [5, 3], "s2": [2, 4]})
    path = tmp_path / "dup_counts.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_counts(path)


def test_load_metadata_rejects_missing_sample_id_column(tmp_path):
    df = pd.DataFrame({"condition": ["control", "stress"]})
    path = tmp_path / "bad_meta.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_metadata(path, sample_id_column="sample_id")


def test_load_counts_happy_path(tmp_path):
    df = pd.DataFrame({"gene_id": ["g1", "g2"], "s1": [5, 3], "s2": [2, 4]})
    path = tmp_path / "counts.csv"
    df.to_csv(path, index=False)
    loaded = load_counts(path)
    assert loaded.shape == (2, 2)
    assert loaded.loc["g1", "s1"] == 5

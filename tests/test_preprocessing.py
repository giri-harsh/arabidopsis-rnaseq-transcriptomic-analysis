"""Unit tests for rnaseq.preprocessing — filtering and normalization math."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rnaseq.preprocessing import (
    compute_library_sizes,
    filter_low_count_genes,
    compute_size_factors,
    normalize_counts,
)


def test_compute_library_sizes(tiny_counts):
    sizes = compute_library_sizes(tiny_counts)
    assert sizes["c1"] == tiny_counts["c1"].sum()
    assert len(sizes) == tiny_counts.shape[1]


def test_filter_low_count_genes_removes_zero_gene(tiny_counts):
    # gene index 5 ("g5") is all zero/near-zero in the fixture
    filtered = filter_low_count_genes(tiny_counts, min_total_count=10, min_samples_expressed=2)
    assert "g5" not in filtered.index
    assert filtered.shape[0] < tiny_counts.shape[0]


def test_filter_low_count_genes_keeps_expressed_genes(tiny_counts):
    filtered = filter_low_count_genes(tiny_counts, min_total_count=10, min_samples_expressed=2)
    assert "g0" in filtered.index  # highly expressed gene must survive


def test_compute_size_factors_returns_one_per_sample(tiny_counts):
    factors = compute_size_factors(tiny_counts)
    assert len(factors) == tiny_counts.shape[1]
    assert (factors > 0).all()


def test_normalize_counts_divides_correctly(tiny_counts):
    factors = compute_size_factors(tiny_counts)
    normalized = normalize_counts(tiny_counts, factors)
    expected = tiny_counts["c1"].iloc[0] / factors["c1"]
    assert np.isclose(normalized["c1"].iloc[0], expected)


def test_normalize_counts_preserves_shape(tiny_counts):
    factors = compute_size_factors(tiny_counts)
    normalized = normalize_counts(tiny_counts, factors)
    assert normalized.shape == tiny_counts.shape

"""Shared pytest fixtures: tiny synthetic count matrix + metadata for fast tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_counts() -> pd.DataFrame:
    """6 genes x 6 samples raw counts, 3 control / 3 stress, with 2 clear DEGs."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(6)]
    samples = ["c1", "c2", "c3", "s1", "s2", "s3"]

    data = {
        "c1": [100, 20, 5, 300, 8, 0],
        "c2": [110, 18, 6, 280, 9, 1],
        "c3": [95, 22, 4, 320, 7, 0],
        "s1": [400, 21, 100, 90, 8, 0],
        "s2": [420, 19, 110, 100, 9, 1],
        "s3": [390, 20, 95, 95, 10, 0],
    }
    df = pd.DataFrame(data, index=genes)
    df.index.name = "gene_id"
    return df


@pytest.fixture
def tiny_metadata() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["c1", "c2", "c3", "s1", "s2", "s3"],
        "condition": ["control", "control", "control", "stress", "stress", "stress"],
    })

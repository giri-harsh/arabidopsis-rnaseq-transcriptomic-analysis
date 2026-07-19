"""
dimensionality.py
==================

Optional: t-SNE alongside PCA, for comparison on the same top-variable-gene
matrix. Not central to DE analysis — gated by config.run_dim_reduction_comparison
since t-SNE has no principled component count/variance-explained interpretation
and can mislead if over-relied upon with small sample sizes (n=6-8 typical here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def run_tsne(
    normalized_counts: pd.DataFrame,
    n_top_variable_genes: int = 500,
    perplexity: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    t-SNE on top-variable-gene log2 expression.

    perplexity defaults low (3.0) because typical RNA-seq designs here have
    few samples per condition (n=3-5); sklearn requires perplexity < n_samples.
    Caller should lower this further for very small sample sizes.
    """
    log_counts = np.log2(normalized_counts + 1)
    gene_variance = log_counts.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(n_top_variable_genes).index
    matrix = log_counts.loc[top_genes].T

    n_samples = matrix.shape[0]
    safe_perplexity = min(perplexity, max(1.0, n_samples - 1))

    tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=random_state, init="pca")
    coords = tsne.fit_transform(matrix.values)

    return pd.DataFrame(coords, index=matrix.index, columns=["TSNE1", "TSNE2"])

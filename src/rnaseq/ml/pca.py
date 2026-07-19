"""
pca.py
======

PCA on log-transformed normalized counts. Standard first-pass check: do
samples separate by condition along the top components? Uses sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    coords: pd.DataFrame  # sample x PC
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame  # gene x PC (top-variance genes only, for interpretability)


def run_pca(
    normalized_counts: pd.DataFrame,
    n_components: int = 2,
    n_top_variable_genes: int = 500,
    random_state: int = 42,
) -> PCAResult:
    """
    normalized_counts: gene x sample.
    Restricts to top-N most variable genes (log2 scale) before PCA — standard
    practice to avoid low-count noise dominating the decomposition.
    """
    log_counts = np.log2(normalized_counts + 1)

    gene_variance = log_counts.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(n_top_variable_genes).index
    matrix = log_counts.loc[top_genes].T  # sample x gene

    centered = matrix - matrix.mean(axis=0)

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(centered)

    coords = pd.DataFrame(
        scores,
        index=matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=top_genes,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return PCAResult(
        coords=coords,
        explained_variance_ratio=pca.explained_variance_ratio_,
        loadings=loadings,
    )

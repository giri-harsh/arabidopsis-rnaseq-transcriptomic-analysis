"""
clustering.py
=============

Hierarchical clustering + KMeans on samples (using top-variable-gene
expression), complementing PCA. Answers: do samples group by condition
using an independent unsupervised method?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass
class ClusteringResult:
    hierarchical_labels: pd.Series  # sample -> cluster id
    kmeans_labels: pd.Series
    kmeans_silhouette: float
    linkage_matrix: np.ndarray


def _top_variable_matrix(normalized_counts: pd.DataFrame, n_top_variable_genes: int) -> pd.DataFrame:
    log_counts = np.log2(normalized_counts + 1)
    gene_variance = log_counts.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(n_top_variable_genes).index
    return log_counts.loc[top_genes].T  # sample x gene


def run_clustering(
    normalized_counts: pd.DataFrame,
    k: int = 2,
    n_top_variable_genes: int = 500,
    random_state: int = 42,
) -> ClusteringResult:
    """
    Hierarchical (Ward linkage, Euclidean distance) + KMeans(k) on the same
    top-variable-gene sample x gene matrix used for PCA.
    """
    matrix = _top_variable_matrix(normalized_counts, n_top_variable_genes)

    dist = pdist(matrix.values, metric="euclidean")
    Z = linkage(dist, method="ward")
    hier_labels = fcluster(Z, t=k, criterion="maxclust")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(matrix.values)

    sil = silhouette_score(matrix.values, kmeans_labels) if k > 1 and k < len(matrix) else float("nan")

    return ClusteringResult(
        hierarchical_labels=pd.Series(hier_labels, index=matrix.index, name="hierarchical_cluster"),
        kmeans_labels=pd.Series(kmeans_labels, index=matrix.index, name="kmeans_cluster"),
        kmeans_silhouette=float(sil),
        linkage_matrix=Z,
    )


def cluster_condition_agreement(cluster_labels: pd.Series, condition_labels: pd.Series) -> float:
    """
    Adjusted Rand Index between unsupervised cluster assignment and true
    condition labels. 1.0 = perfect agreement, ~0 = random.
    """
    aligned_conditions = condition_labels.loc[cluster_labels.index]
    return float(adjusted_rand_score(aligned_conditions, cluster_labels))

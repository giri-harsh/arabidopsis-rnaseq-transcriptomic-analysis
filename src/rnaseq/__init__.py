"""
rnaseq
======

A reproducible RNA-seq differential gene expression analysis pipeline for
*Arabidopsis thaliana* abiotic stress datasets.

The package is organized around the standard RNA-seq DEG workflow:

    io            -> load & validate raw counts / metadata
    preprocessing -> filtering, library size, normalization, QC
    deseq_pipeline-> PyDESeq2 wrapper (dispersion estimation, Wald test, BH FDR)
    statistics    -> summary statistics, correlation, DEG table construction
    ml            -> PCA, hierarchical clustering, KMeans, dim-reduction comparison
    viz           -> publication-quality figures (volcano, MA, heatmap, PCA, QC)
    report        -> Markdown / JSON / CSV report generation

See README.md at the repository root for the full scientific background and
usage instructions.
"""

__version__ = "0.1.0"

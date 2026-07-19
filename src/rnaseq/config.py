"""
config.py
=========

Centralized, type-safe configuration for a single pipeline run.

Why a config module at all?
----------------------------
RNA-seq pipelines have many tunable parameters (filtering thresholds,
significance cutoffs, fold-change cutoffs, clustering parameters, etc.).
Scattering these as magic numbers throughout the codebase makes the
pipeline hard to reproduce and hard to audit. Instead, every run is
described by a single, serializable ``PipelineConfig`` object that is:

  * constructed once (from CLI args or defaults),
  * passed explicitly to every module that needs it,
  * written out to ``outputs/reports/run_config.json`` for provenance,
    so that any output figure or table can be traced back to the exact
    parameters that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Immutable-by-convention configuration object for a full pipeline run.

    Attributes
    ----------
    counts_path:
        Path to the raw gene x sample count matrix (CSV/TSV). Genes are
        expected as rows, samples as columns.
    metadata_path:
        Path to the sample metadata table. Must contain a sample identifier
        column matching the count matrix column names, and a condition
        column used as the design factor for differential expression.
    output_dir:
        Root directory under which figures/, tables/, reports/, logs/ are
        created for this run.
    sample_id_column:
        Name of the column in metadata that holds sample identifiers.
    condition_column:
        Name of the column in metadata that holds the experimental condition
        (e.g. "control" vs "stress"). Used as the DESeq2 design formula
        factor: ``~condition``.
    reference_level:
        The condition value treated as the baseline/reference in the Wald
        test contrast (e.g. "control"). The non-reference level is the
        "treatment" level being tested against it.
    min_total_count:
        Minimum summed raw count across all samples for a gene to be kept
        during preprocessing. Genes below this are considered too lowly
        expressed to yield reliable dispersion estimates.
    min_samples_expressed:
        Minimum number of samples in which a gene must have a nonzero count
        to be retained.
    alpha:
        Significance threshold (adjusted p-value / FDR cutoff) used to call
        a gene significantly differentially expressed.
    lfc_threshold:
        Absolute log2 fold-change threshold combined with ``alpha`` to
        define the "significant DEG" set used in downstream volcano/MA
        plots and gene lists.
    n_pca_components:
        Number of principal components to compute in the PCA module.
    kmeans_k:
        Number of clusters for the optional KMeans sample-clustering
        analysis.
    random_state:
        Seed used everywhere randomness is involved (KMeans init, t-SNE/UMAP)
        so that results are reproducible.
    top_n_heatmap_genes:
        Number of top significant genes (by adjusted p-value) to include in
        the clustered heatmap.
    run_dim_reduction_comparison:
        Whether to additionally compute t-SNE/UMAP alongside PCA for
        comparison. Optional because it is not central to the DE analysis
        and adds extra dependencies/runtime.
    figure_dpi:
        DPI used when exporting PNG figures.
    """

    counts_path: Path
    metadata_path: Path
    output_dir: Path = Path("outputs")

    sample_id_column: str = "sample_id"
    condition_column: str = "condition"
    reference_level: str = "control"

    min_total_count: int = 10
    min_samples_expressed: int = 2

    alpha: float = 0.05
    lfc_threshold: float = 1.0

    n_pca_components: int = 2
    kmeans_k: int = 2
    random_state: int = 42
    top_n_heatmap_genes: int = 50
    run_dim_reduction_comparison: bool = False

    figure_dpi: int = 300

    # Populated automatically in __post_init__; not meant to be set by hand.
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.counts_path = Path(self.counts_path)
        self.metadata_path = Path(self.metadata_path)
        self.output_dir = Path(self.output_dir)

        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        self.logs_dir = self.output_dir / "logs"

    def create_output_dirs(self) -> None:
        """Create the full output directory tree (idempotent)."""
        for d in (
            self.output_dir,
            self.figures_dir,
            self.figures_dir / "png",
            self.figures_dir / "pdf",
            self.tables_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize the config to a JSON-friendly dictionary (paths as str)."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Persist this configuration as JSON for provenance/reproducibility.

        Parameters
        ----------
        path:
            Destination file. Defaults to ``reports_dir / "run_config.json"``.

        Returns
        -------
        Path to the written file.
        """
        destination = path or (self.reports_dir / "run_config.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return destination

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """
        Build a ``PipelineConfig`` from an ``argparse.Namespace``.

        Only attributes present as CLI flags are overridden; all other
        fields keep their dataclass defaults.
        """
        kwargs = {
            "counts_path": args.counts,
            "metadata_path": args.metadata,
            "output_dir": args.output,
        }
        for opt_field in (
            "sample_id_column",
            "condition_column",
            "reference_level",
            "min_total_count",
            "min_samples_expressed",
            "alpha",
            "lfc_threshold",
            "n_pca_components",
            "kmeans_k",
            "random_state",
            "top_n_heatmap_genes",
            "run_dim_reduction_comparison",
            "figure_dpi",
        ):
            if hasattr(args, opt_field) and getattr(args, opt_field) is not None:
                kwargs[opt_field] = getattr(args, opt_field)
        return cls(**kwargs)

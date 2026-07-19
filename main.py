#!/usr/bin/env python3
"""
main.py
=======

CLI entry point: runs the full RNA-seq DGE pipeline end-to-end.

    python main.py --counts data/example_dataset/counts.csv \\
                    --metadata data/example_dataset/metadata.csv \\
                    --output outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rnaseq.config import PipelineConfig
from rnaseq.logging_utils import setup_logging, get_logger
from rnaseq.io import load_dataset, DataValidationError
from rnaseq.preprocessing import run_preprocessing
from rnaseq.deseq_pipeline import run_deseq2
from rnaseq.statistics import extract_significant_degs, sample_correlation_matrix, summary_statistics
from rnaseq.ml.pca import run_pca
from rnaseq.ml.clustering import run_clustering, cluster_condition_agreement
from rnaseq.ml.dimensionality import run_tsne
from rnaseq.viz.volcano import plot_volcano
from rnaseq.viz.ma_plot import plot_ma
from rnaseq.viz.heatmap import plot_deg_heatmap
from rnaseq.viz.pca_plot import plot_pca
from rnaseq.viz.qc_plots import (
    plot_library_sizes,
    plot_normalization_boxplots,
    plot_expression_distribution,
    plot_sample_correlation,
)
from rnaseq.report import export_tables, generate_markdown_report

FIGURE_NAMES = [
    "volcano_plot", "ma_plot", "deg_heatmap", "pca_plot",
    "library_sizes", "normalization_boxplots",
    "expression_distribution", "sample_correlation",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arabidopsis RNA-seq differential gene expression pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--counts", required=True, help="Path to gene x sample raw count matrix (CSV/TSV).")
    parser.add_argument("--metadata", required=True, help="Path to sample metadata table (CSV/TSV).")
    parser.add_argument("--output", default="outputs", help="Output directory root.")
    parser.add_argument("--sample-id-column", dest="sample_id_column", default="sample_id")
    parser.add_argument("--condition-column", dest="condition_column", default="condition")
    parser.add_argument("--reference-level", dest="reference_level", default="control")
    parser.add_argument("--min-total-count", dest="min_total_count", type=int, default=10)
    parser.add_argument("--min-samples-expressed", dest="min_samples_expressed", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold.")
    parser.add_argument("--lfc-threshold", dest="lfc_threshold", type=float, default=1.0)
    parser.add_argument("--n-pca-components", dest="n_pca_components", type=int, default=2)
    parser.add_argument("--kmeans-k", dest="kmeans_k", type=int, default=2)
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    parser.add_argument("--top-n-heatmap-genes", dest="top_n_heatmap_genes", type=int, default=50)
    parser.add_argument("--run-dim-reduction-comparison", dest="run_dim_reduction_comparison", action="store_true")
    parser.add_argument("--figure-dpi", dest="figure_dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = PipelineConfig.from_args(args)
    config.create_output_dirs()

    logger = setup_logging(logs_dir=config.logs_dir)
    logger.info("=== Arabidopsis RNA-seq DGE pipeline starting ===")
    config.save()

    try:
        dataset = load_dataset(
            counts_path=config.counts_path,
            metadata_path=config.metadata_path,
            sample_id_column=config.sample_id_column,
            condition_column=config.condition_column,
        )
    except (DataValidationError, FileNotFoundError) as exc:
        logger.error("Input validation failed: %s", exc)
        return 1

    preprocessing = run_preprocessing(dataset.counts, config)

    try:
        deseq_result = run_deseq2(preprocessing.filtered_counts, dataset.metadata, config)
    except Exception as exc:  # noqa: BLE001 - surface any PyDESeq2 fitting failure clearly
        logger.error("DESeq2 fitting failed: %s", exc)
        return 1

    deg_summary = extract_significant_degs(deseq_result.results_df, config)
    correlation_matrix = sample_correlation_matrix(preprocessing.normalized_counts)

    pca_result = run_pca(preprocessing.normalized_counts, n_components=config.n_pca_components,
                          random_state=config.random_state)
    clustering_result = run_clustering(preprocessing.normalized_counts, k=config.kmeans_k,
                                        random_state=config.random_state)
    ari = cluster_condition_agreement(clustering_result.kmeans_labels, dataset.conditions)

    if config.run_dim_reduction_comparison:
        try:
            run_tsne(preprocessing.normalized_counts, random_state=config.random_state)
            logger.info("t-SNE comparison computed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("t-SNE comparison skipped due to error: %s", exc)

    logger.info("Generating figures...")
    plot_volcano(deg_summary.full_table, config.alpha, config.lfc_threshold, config.figures_dir, dpi=config.figure_dpi)
    plot_ma(deg_summary.full_table, config.alpha, config.figures_dir, dpi=config.figure_dpi)
    try:
        plot_deg_heatmap(
            preprocessing.normalized_counts, deg_summary.significant, dataset.metadata,
            config.condition_column, config.top_n_heatmap_genes, config.figures_dir, dpi=config.figure_dpi,
        )
    except ValueError as exc:
        logger.warning("Skipped heatmap: %s", exc)
    plot_pca(pca_result.coords, pca_result.explained_variance_ratio, dataset.metadata,
              config.condition_column, config.figures_dir, dpi=config.figure_dpi)
    plot_library_sizes(preprocessing.library_sizes, dataset.metadata, config.condition_column,
                        config.figures_dir, dpi=config.figure_dpi)
    plot_normalization_boxplots(preprocessing.filtered_counts, preprocessing.normalized_counts,
                                 config.figures_dir, dpi=config.figure_dpi)
    plot_expression_distribution(preprocessing.normalized_counts, config.figures_dir, dpi=config.figure_dpi)
    plot_sample_correlation(correlation_matrix, config.figures_dir, dpi=config.figure_dpi)

    logger.info("Exporting tables and report...")
    table_paths = export_tables(deg_summary, preprocessing.normalized_counts, correlation_matrix, config)
    report_path = generate_markdown_report(
        dataset_name=str(config.counts_path),
        config=config,
        n_genes_input=dataset.n_genes,
        n_samples=dataset.n_samples,
        preprocessing_n_removed=preprocessing.n_genes_removed,
        deg_summary=deg_summary,
        pca_explained_variance=pca_result.explained_variance_ratio,
        clustering_ari=ari,
        contrast=deseq_result.contrast,
        table_paths=table_paths,
        figure_names=FIGURE_NAMES,
    )

    summary = summary_statistics(deg_summary, preprocessing.n_genes_removed, dataset.n_genes)
    logger.info("Run complete. Summary: %s", summary)
    logger.info("Report written to: %s", report_path)
    print(f"\nPipeline complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
report.py
=========

Final reporting stage: writes all machine-readable outputs (CSV/TSV/JSON)
+ a human-readable analysis_report.md summarizing the full run.

Nothing here computes new statistics — it only serializes results already
produced by preprocessing / deseq_pipeline / statistics / ml modules, so
the report can never say something the pipeline didn't actually do.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .statistics import DEGSummary
from .logging_utils import get_logger

logger = get_logger(__name__)


def export_tables(
    deg_summary: DEGSummary,
    normalized_counts: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    """
    Write all DEG tables + supporting matrices as CSV/TSV, and a summary
    JSON. Returns dict of {name: path} for use in the markdown report.
    """
    tables_dir = config.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    full_csv = tables_dir / "deseq2_full_results.csv"
    deg_summary.full_table.to_csv(full_csv)
    paths["full_results_csv"] = full_csv

    sig_csv = tables_dir / "significant_degs.csv"
    deg_summary.significant.to_csv(sig_csv)
    paths["significant_degs_csv"] = sig_csv

    sig_tsv = tables_dir / "significant_degs.tsv"
    deg_summary.significant.to_csv(sig_tsv, sep="\t")
    paths["significant_degs_tsv"] = sig_tsv

    up_csv = tables_dir / "upregulated_genes.csv"
    deg_summary.upregulated.to_csv(up_csv)
    paths["upregulated_csv"] = up_csv

    down_csv = tables_dir / "downregulated_genes.csv"
    deg_summary.downregulated.to_csv(down_csv)
    paths["downregulated_csv"] = down_csv

    norm_csv = tables_dir / "normalized_counts.csv"
    normalized_counts.to_csv(norm_csv)
    paths["normalized_counts_csv"] = norm_csv

    corr_csv = tables_dir / "sample_correlation_matrix.csv"
    correlation_matrix.to_csv(corr_csv)
    paths["sample_correlation_csv"] = corr_csv

    sig_gene_list_json = tables_dir / "significant_gene_ids.json"
    with open(sig_gene_list_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_significant": deg_summary.n_significant,
                "gene_ids": deg_summary.significant.index.tolist(),
            },
            fh,
            indent=2,
        )
    paths["significant_gene_ids_json"] = sig_gene_list_json

    logger.info("Exported %d table files to %s", len(paths), tables_dir)
    return paths


def _fmt(x, digits: int = 3) -> str:
    try:
        return f"{x:.{digits}g}"
    except (TypeError, ValueError):
        return str(x)


def generate_markdown_report(
    dataset_name: str,
    config: PipelineConfig,
    n_genes_input: int,
    n_samples: int,
    preprocessing_n_removed: int,
    deg_summary: DEGSummary,
    pca_explained_variance,
    clustering_ari: float,
    contrast: tuple,
    table_paths: dict,
    figure_names: list[str],
) -> Path:
    """
    Assemble analysis_report.md from already-computed results.

    Every number in the report is pulled directly from pipeline objects —
    no numbers are invented or estimated here.
    """
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "analysis_report.md"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    factor, test_level, ref_level = contrast

    top_degs = deg_summary.significant.head(15)
    top_deg_rows = "\n".join(
        f"| {gid} | {row['baseMean']:.1f} | {row['log2FoldChange']:.2f} | {row['padj']:.2e} |"
        for gid, row in top_degs.iterrows()
    )

    md = f"""# Transcriptomic Analysis Report

**Dataset:** {dataset_name}
**Generated:** {timestamp}
**Design:** `~{factor}` — {test_level} vs {ref_level} (reference)

---

## 1. Dataset Overview

- Input genes: **{n_genes_input}**
- Samples: **{n_samples}**
- Genes removed during low-count filtering: **{preprocessing_n_removed}**
  (min total count = {config.min_total_count}, min samples expressed = {config.min_samples_expressed})
- Genes tested for differential expression: **{deg_summary.n_tested}**

## 2. Pipeline Summary

1. Raw counts + metadata loaded and validated (sample alignment, integer
   count check, missing value check).
2. Low-count genes filtered.
3. Library sizes computed; size factors estimated via median-of-ratios
   normalization (DESeq2 methodology).
4. PyDESeq2 fit: gene-wise dispersion estimation → dispersion trend curve →
   MAP shrinkage → negative binomial GLM fit.
5. Wald test performed per gene for the contrast `{factor}: {test_level} vs {ref_level}`.
6. Benjamini-Hochberg FDR correction applied to raw p-values (`padj` column).
7. Significant DEGs extracted at padj < {config.alpha} and
   |log2FoldChange| >= {config.lfc_threshold}.
8. PCA, hierarchical clustering, and KMeans clustering performed on the
   top-variable-gene expression matrix as an unsupervised cross-check.
9. Publication-quality figures generated (PCA, volcano, MA, heatmap, QC plots).

## 3. Statistical Methods

- **Normalization:** median-of-ratios size factors (as in DESeq2), computed
  independently for QC visualization; PyDESeq2 additionally computes its own
  internal size factors for the GLM fit.
- **Dispersion estimation:** per-gene dispersion estimated by PyDESeq2 using
  the DESeq2 gene-wise → trend → MAP shrinkage procedure, which stabilizes
  variance estimates for genes with few replicates.
- **Hypothesis testing:** Wald test on the fitted log2 fold change for the
  `{factor}` coefficient.
- **Multiple testing correction:** Benjamini-Hochberg procedure controlling
  the false discovery rate at alpha = {config.alpha}.

## 4. Differential Expression Results

| Metric | Value |
|---|---|
| Significant DEGs (padj < {config.alpha}, \\|log2FC\\| >= {config.lfc_threshold}) | {deg_summary.n_significant} |
| Upregulated in {test_level} | {deg_summary.n_up} |
| Downregulated in {test_level} | {deg_summary.n_down} |

### Top 15 Significant Genes (by adjusted p-value)

| Gene ID | Base Mean | log2FC | padj |
|---|---|---|---|
{top_deg_rows}

Full results: `{table_paths.get('full_results_csv', '')}`
Significant DEG table: `{table_paths.get('significant_degs_csv', '')}`

## 5. Machine Learning / Unsupervised Analysis

- **PCA:** PC1 explains {pca_explained_variance[0]*100:.1f}% of variance,
  PC2 explains {pca_explained_variance[1]*100:.1f}% (top {config.n_pca_components}
  components computed on the {config.n_pca_components}-component projection
  of the top-variable-gene matrix).
- **Clustering agreement:** Adjusted Rand Index between KMeans (k={config.kmeans_k})
  cluster assignment and true condition labels = **{clustering_ari:.3f}**
  (1.0 = perfect agreement with known condition groups, ~0 = no better than
  random assignment).

## 6. Figures

Saved under `outputs/figures/{{png,pdf}}/`:

{chr(10).join(f"- {name}" for name in figure_names)}

## 7. Biological Interpretation

The significant DEG set represents genes whose expression changed between
{ref_level} and {test_level} conditions beyond what is expected from
normal sample-to-sample variability, at the stated FDR threshold. Genes
with large positive log2 fold change are candidates for stress-induced
upregulation (e.g. stress-response transcription factors, osmotic/oxidative
stress pathway genes in a real Arabidopsis abiotic stress context);
genes with large negative log2 fold change are candidates for
stress-induced repression (e.g. genes tied to normal growth/metabolic
processes that are downregulated under stress). Interpreting specific
gene identities biologically requires cross-referencing the gene ID list
against a functional annotation database (e.g. TAIR, Ensembl Plants) —
this pipeline produces the statistically validated candidate list; it
does not perform gene ontology or pathway enrichment itself (see
Limitations).

## 8. Limitations

- This report's numeric example run may use the bundled **synthetic**
  example dataset (see `data/example_dataset/README.md`) unless a real
  count matrix was supplied via `--counts`/`--metadata`; synthetic data
  validates pipeline correctness but carries no biological meaning.
- No gene ontology (GO) or pathway enrichment analysis is performed;
  DEG lists are statistical outputs only and require external biological
  annotation to interpret function.
- Dispersion/Wald-test power is limited by sample size; small replicate
  counts (as is common in real public datasets) widen confidence intervals
  on fold-change estimates.
- t-SNE (if enabled) has no principled variance-explained interpretation
  and should be read qualitatively, not quantitatively.
- Batch effects, if present in the source experiment and not encoded as a
  metadata covariate, are not modeled by the current `~condition` design
  formula.

## 9. Future Work

- Add GO-term / KEGG pathway enrichment on the significant DEG list.
- Support multi-factor designs (e.g. `~batch + condition`) and
  interaction-term contrasts.
- Add automatic outlier-sample detection/flagging prior to DESeq2 fitting.
- Extend to time-course RNA-seq designs (likelihood ratio test workflows).
- Integrate a real GEO/SRA Arabidopsis abiotic-stress dataset by default
  (see `docs/DATASET.md`) with an automated download script.

---

*Report generated automatically by `rnaseq.report`. All statistics above
are computed directly from the pipeline run; no values in this report are
manually edited or estimated.*
"""

    report_path.write_text(md, encoding="utf-8")
    logger.info("Wrote analysis report to %s", report_path)
    return report_path

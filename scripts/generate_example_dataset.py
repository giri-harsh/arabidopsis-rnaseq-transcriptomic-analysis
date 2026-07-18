#!/usr/bin/env python3
"""
generate_example_dataset.py
============================

Generates a small, fully SYNTHETIC gene x sample RNA-seq count matrix and
matching metadata, used as the repository's bundled "example_dataset".

This is **not** real biological data. It exists so that:

  1. ``python main.py`` and the test suite can run immediately after
     cloning, with no external download and no licensing concerns.
  2. The simulated design mirrors a real two-condition Arabidopsis
     abiotic-stress experiment (control vs. stress, n replicates per
     group), including a known, seeded set of "true" differentially
     expressed genes -- which makes it useful for sanity-checking that
     the pipeline recovers a reasonable DEG set.

Simulation model
-----------------
For each gene, baseline mean expression is drawn from a log-normal
distribution (mimicking the highly skewed expression-level distribution
seen in real RNA-seq data). Counts are then sampled from a Negative
Binomial distribution parameterized by that mean and a per-gene dispersion,
which is how DESeq2 itself models RNA-seq counts -- so the simulated data
is compatible with the statistical assumptions the downstream pipeline
relies on.

A subset of genes ("true DEGs") have their mean expression multiplied by a
fold-change in the stress condition, in both directions (up- and
down-regulated), so that differential expression analysis has real signal
to recover.

Usage
-----
    python scripts/generate_example_dataset.py \\
        --n-genes 2000 --n-control 4 --n-stress 4 --out-dir data/example_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_counts(
    n_genes: int,
    n_control: int,
    n_stress: int,
    frac_de: float = 0.10,
    mean_log_expr: float = 5.0,
    sd_log_expr: float = 2.0,
    dispersion_shape: float = 2.0,
    dispersion_scale: float = 0.15,
    fold_change_range: tuple[float, float] = (2.0, 6.0),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Simulate a gene x sample negative-binomial RNA-seq count matrix.

    Parameters
    ----------
    n_genes:
        Number of genes to simulate.
    n_control, n_stress:
        Number of replicate samples per condition.
    frac_de:
        Fraction of genes seeded as "true" differentially expressed genes.
    mean_log_expr, sd_log_expr:
        Parameters of the log-normal distribution used to draw each gene's
        baseline mean count (controls the realistic skew of expression
        levels across genes).
    dispersion_shape, dispersion_scale:
        Gamma distribution parameters used to draw each gene's NB
        dispersion (variance = mean + dispersion * mean^2).
    fold_change_range:
        Range of absolute fold-changes applied to true DEGs, split roughly
        evenly between up- and down-regulated genes.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    counts:
        DataFrame (genes x samples) of simulated raw integer counts.
    metadata:
        DataFrame (samples x fields) with columns 'sample_id', 'condition',
        'replicate'.
    true_deg_ids:
        List of gene ids seeded as true differentially expressed genes
        (useful for validating pipeline recall, not required for the
        pipeline itself).
    """
    rng = np.random.default_rng(random_state)

    n_samples = n_control + n_stress
    gene_ids = [f"AT_SIM_{i:05d}" for i in range(n_genes)]

    sample_ids = [f"control_{i+1}" for i in range(n_control)] + [
        f"stress_{i+1}" for i in range(n_stress)
    ]
    conditions = ["control"] * n_control + ["stress"] * n_stress

    baseline_means = rng.lognormal(mean=mean_log_expr, sigma=sd_log_expr, size=n_genes)
    baseline_means = np.clip(baseline_means, 1.0, None)

    dispersions = rng.gamma(shape=dispersion_shape, scale=dispersion_scale, size=n_genes)

    n_de = int(round(n_genes * frac_de))
    de_indices = rng.choice(n_genes, size=n_de, replace=False)
    up_indices = de_indices[: n_de // 2]
    down_indices = de_indices[n_de // 2 :]

    fold_changes = np.ones(n_genes)
    fold_changes[up_indices] = rng.uniform(*fold_change_range, size=len(up_indices))
    fold_changes[down_indices] = 1.0 / rng.uniform(*fold_change_range, size=len(down_indices))

    counts = np.zeros((n_genes, n_samples), dtype=int)

    for j, condition in enumerate(conditions):
        # library-size (sequencing depth) factor per sample: real samples
        # never have identical depth, so we perturb around 1.0.
        size_factor = rng.normal(loc=1.0, scale=0.12)
        size_factor = max(size_factor, 0.5)

        gene_means = baseline_means.copy()
        if condition == "stress":
            gene_means = gene_means * fold_changes
        gene_means = gene_means * size_factor

        # NB parameterization: variance = mu + alpha * mu^2
        # => n = 1 / alpha, p = n / (n + mu)
        alpha = dispersions
        n_param = 1.0 / alpha
        p_param = n_param / (n_param + gene_means)

        counts[:, j] = rng.negative_binomial(n_param, p_param)

    counts_df = pd.DataFrame(counts, index=gene_ids, columns=sample_ids)
    counts_df.index.name = "gene_id"

    metadata_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": conditions,
            "replicate": [i + 1 for i in range(n_control)]
            + [i + 1 for i in range(n_stress)],
        }
    )

    true_deg_ids = [gene_ids[i] for i in de_indices]
    return counts_df, metadata_df, true_deg_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-control", type=int, default=4)
    parser.add_argument("--n-stress", type=int, default=4)
    parser.add_argument("--frac-de", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/example_dataset"),
        help="Directory to write counts.csv, metadata.csv, true_degs.json into.",
    )
    args = parser.parse_args()

    counts_df, metadata_df, true_deg_ids = simulate_counts(
        n_genes=args.n_genes,
        n_control=args.n_control,
        n_stress=args.n_stress,
        frac_de=args.frac_de,
        random_state=args.random_state,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts_path = args.out_dir / "counts.csv"
    metadata_path = args.out_dir / "metadata.csv"
    true_degs_path = args.out_dir / "true_degs.json"
    readme_path = args.out_dir / "README.md"

    counts_df.to_csv(counts_path)
    metadata_df.to_csv(metadata_path, index=False)
    with open(true_degs_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "description": "Gene ids seeded as true differentially expressed "
                "genes in this SYNTHETIC dataset. Provided only for pipeline "
                "sanity-checking, not biological reference.",
                "n_true_degs": len(true_deg_ids),
                "true_deg_ids": true_deg_ids,
            },
            fh,
            indent=2,
        )

    readme_path.write_text(
        "# Example Dataset (SYNTHETIC)\n\n"
        "This dataset is generated by `scripts/generate_example_dataset.py` "
        "using a Negative-Binomial simulation model and is **not** real "
        "Arabidopsis thaliana sequencing data.\n\n"
        "It exists so the pipeline can be run and tested immediately after "
        "cloning the repository, without downloading external data.\n\n"
        f"- Genes: {args.n_genes}\n"
        f"- Samples: {args.n_control} control, {args.n_stress} stress\n"
        f"- Seeded true DEGs: {len(true_deg_ids)} "
        f"({args.frac_de:.0%} of genes, split up/down-regulated)\n"
        f"- Random seed: {args.random_state}\n\n"
        "For the real dataset used in the published analysis, see "
        "`docs/DATASET.md` at the repository root.\n",
        encoding="utf-8",
    )

    print(f"Wrote synthetic counts matrix:   {counts_path}")
    print(f"Wrote synthetic metadata:        {metadata_path}")
    print(f"Wrote true DEG reference list:   {true_degs_path}")
    print(f"Wrote dataset README:            {readme_path}")


if __name__ == "__main__":
    main()

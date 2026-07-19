"""
io.py
=====

Loading and validation of raw RNA-seq inputs: the gene x sample count
matrix and the sample metadata table.

Why validation is its own step
-------------------------------
DESeq2-style negative binomial modeling is sensitive to malformed inputs:
mismatched sample identifiers between counts and metadata, non-integer
counts (e.g. accidentally loading TPM/FPKM instead of raw counts), or
missing values will either crash PyDESeq2 with an opaque error or silently
produce nonsense dispersion estimates. Validating explicitly, with
human-readable error messages, is what separates a research-grade pipeline
from a notebook that "worked once."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when raw inputs fail structural or biological sanity checks."""


@dataclass
class RNASeqDataset:
    """
    Container bundling a validated count matrix with its sample metadata.

    Attributes
    ----------
    counts:
        Gene x sample integer count matrix (genes as index, samples as
        columns), already restricted/ordered to match ``metadata``.
    metadata:
        Sample metadata, indexed by sample id, in the same order as
        ``counts.columns``.
    condition_column:
        Name of the metadata column used as the DESeq2 design factor.
    """

    counts: pd.DataFrame
    metadata: pd.DataFrame
    condition_column: str

    @property
    def n_genes(self) -> int:
        return self.counts.shape[0]

    @property
    def n_samples(self) -> int:
        return self.counts.shape[1]

    @property
    def conditions(self) -> pd.Series:
        return self.metadata[self.condition_column]


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV or TSV file, inferring the delimiter from the extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".tsv", ".txt"):
        sep = "\t"
    else:
        sep = ","

    logger.info("Reading table from %s (sep=%r)", path, sep)
    return pd.read_csv(path, sep=sep)


def load_counts(path: Union[str, Path], gene_id_column: str = "gene_id") -> pd.DataFrame:
    """
    Load a raw gene x sample count matrix.

    Parameters
    ----------
    path:
        Path to a CSV/TSV file. The first column (or the column named
        ``gene_id_column`` if present) is treated as the gene identifier
        and set as the DataFrame index. All remaining columns must be
        numeric raw counts, one per sample.
    gene_id_column:
        Expected name of the gene identifier column. If not found, the
        first column of the file is used instead (with a warning).

    Returns
    -------
    DataFrame of shape (n_genes, n_samples), integer dtype, indexed by
    gene id.

    Raises
    ------
    DataValidationError
        If counts contain negative values, non-numeric entries, or are
        clearly not raw integer counts (e.g. contain many non-integer
        fractional values, suggesting normalized/TPM data was supplied).
    """
    df = _read_table(path)

    if gene_id_column in df.columns:
        id_col = gene_id_column
    else:
        id_col = df.columns[0]
        logger.warning(
            "Column '%s' not found in counts file; using first column '%s' as gene id.",
            gene_id_column,
            id_col,
        )

    df = df.set_index(id_col)

    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        raise DataValidationError(
            f"Count matrix contains {n_dupes} duplicated gene identifiers. "
            "Aggregate or deduplicate before running the pipeline."
        )

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise DataValidationError(
            f"Count matrix has non-numeric sample columns: {non_numeric}. "
            "Ensure the gene id column was correctly detected."
        )

    if (df.values < 0).any():
        raise DataValidationError(
            "Count matrix contains negative values, which is invalid for raw "
            "RNA-seq counts."
        )

    if df.isna().any().any():
        n_missing = int(df.isna().sum().sum())
        raise DataValidationError(
            f"Count matrix contains {n_missing} missing values. "
            "Impute or remove affected genes/samples before running the pipeline."
        )

    non_integer_fraction = float(np.mean(~np.isclose(df.values, np.round(df.values))))
    if non_integer_fraction > 0.05:
        raise DataValidationError(
            f"{non_integer_fraction:.1%} of count values are non-integer. "
            "PyDESeq2 expects raw integer counts, not normalized values "
            "(e.g. TPM/FPKM/CPM). Provide the raw count matrix instead."
        )

    df = df.round().astype(int)

    logger.info(
        "Loaded count matrix: %d genes x %d samples from %s",
        df.shape[0],
        df.shape[1],
        path,
    )
    return df


def load_metadata(
    path: Union[str, Path],
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    """
    Load sample metadata.

    Parameters
    ----------
    path:
        Path to a CSV/TSV file describing each sample (at minimum a sample
        identifier and an experimental condition column).
    sample_id_column:
        Column to use as the metadata index and to match against count
        matrix column names.

    Returns
    -------
    DataFrame indexed by sample id.

    Raises
    ------
    DataValidationError
        If the sample id column is missing or contains duplicates.
    """
    df = _read_table(path)

    if sample_id_column not in df.columns:
        raise DataValidationError(
            f"Metadata file is missing required sample id column "
            f"'{sample_id_column}'. Found columns: {list(df.columns)}"
        )

    if df[sample_id_column].duplicated().any():
        raise DataValidationError(
            f"Metadata column '{sample_id_column}' contains duplicate sample ids."
        )

    df = df.set_index(sample_id_column)
    logger.info("Loaded metadata: %d samples x %d fields from %s", *df.shape, path)
    return df


def align_counts_and_metadata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_column: str = "condition",
) -> RNASeqDataset:
    """
    Validate that counts and metadata describe the same samples, then
    reorder both to a consistent sample order.

    Parameters
    ----------
    counts:
        Output of :func:`load_counts`.
    metadata:
        Output of :func:`load_metadata`.
    condition_column:
        Column in ``metadata`` to use as the DESeq2 design factor.

    Returns
    -------
    A validated, aligned :class:`RNASeqDataset`.

    Raises
    ------
    DataValidationError
        If sample sets differ between counts and metadata, if the
        condition column is missing, or if fewer than two condition
        levels are present (differential expression requires at least
        two groups to contrast).
    """
    if condition_column not in metadata.columns:
        raise DataValidationError(
            f"Metadata is missing the condition column '{condition_column}'. "
            f"Available columns: {list(metadata.columns)}"
        )

    count_samples = set(counts.columns)
    meta_samples = set(metadata.index)

    missing_in_metadata = count_samples - meta_samples
    missing_in_counts = meta_samples - count_samples

    if missing_in_metadata or missing_in_counts:
        raise DataValidationError(
            "Sample mismatch between counts and metadata.\n"
            f"  In counts but not metadata: {sorted(missing_in_metadata)}\n"
            f"  In metadata but not counts: {sorted(missing_in_counts)}"
        )

    ordered_samples = list(metadata.index)
    aligned_counts = counts[ordered_samples]
    aligned_metadata = metadata.loc[ordered_samples]

    n_levels = aligned_metadata[condition_column].nunique()
    if n_levels < 2:
        raise DataValidationError(
            f"Condition column '{condition_column}' has only {n_levels} unique "
            "level(s); differential expression requires at least 2 groups."
        )

    logger.info(
        "Aligned dataset: %d genes x %d samples, condition levels: %s",
        aligned_counts.shape[0],
        aligned_counts.shape[1],
        sorted(aligned_metadata[condition_column].unique().tolist()),
    )

    return RNASeqDataset(
        counts=aligned_counts,
        metadata=aligned_metadata,
        condition_column=condition_column,
    )


def load_dataset(
    counts_path: Union[str, Path],
    metadata_path: Union[str, Path],
    sample_id_column: str = "sample_id",
    condition_column: str = "condition",
    gene_id_column: str = "gene_id",
) -> RNASeqDataset:
    """
    Convenience wrapper: load counts + metadata and return a validated,
    aligned :class:`RNASeqDataset` in one call.
    """
    counts = load_counts(counts_path, gene_id_column=gene_id_column)
    metadata = load_metadata(metadata_path, sample_id_column=sample_id_column)
    return align_counts_and_metadata(counts, metadata, condition_column=condition_column)

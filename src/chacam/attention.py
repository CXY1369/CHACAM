"""Fig 2 cosine attention and bidirectional normalizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AttentionResult:
    S: pd.DataFrame
    P_col: pd.DataFrame
    P_row: pd.DataFrame
    L: pd.DataFrame
    used_genes: list[str]
    zero_target_cells: list[str]
    zero_reference_identities: list[str]


def _frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    numeric = value.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite values")
    return numeric


def l1_normalize(similarity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return column-normalized, row-normalized, and product matrices."""

    similarity = _frame(similarity, "similarity")
    if (similarity.to_numpy() < 0).any():
        raise ValueError("l1 normalization requires nonnegative similarity; use softmax for negative values")
    values = similarity.to_numpy(dtype=float)
    column_sums = values.sum(axis=0, keepdims=True)
    row_sums = values.sum(axis=1, keepdims=True)
    p_col_values = np.divide(values, column_sums, out=np.zeros_like(values), where=column_sums != 0)
    p_row_values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    p_col = pd.DataFrame(p_col_values, index=similarity.index, columns=similarity.columns)
    p_row = pd.DataFrame(p_row_values, index=similarity.index, columns=similarity.columns)
    product = p_col * p_row
    return p_col, p_row, product


def softmax_normalize(similarity: pd.DataFrame, temperature: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stable softmaxes over cells and identities and their product."""

    similarity = _frame(similarity, "similarity")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    scaled = similarity.to_numpy(dtype=float) / float(temperature)
    col_shifted = scaled - scaled.max(axis=0, keepdims=True)
    row_shifted = scaled - scaled.max(axis=1, keepdims=True)
    col_exp = np.maximum(np.exp(col_shifted), np.finfo(float).tiny)
    row_exp = np.maximum(np.exp(row_shifted), np.finfo(float).tiny)
    p_col_values = col_exp / col_exp.sum(axis=0, keepdims=True)
    p_row_values = row_exp / row_exp.sum(axis=1, keepdims=True)
    p_col = pd.DataFrame(p_col_values, index=similarity.index, columns=similarity.columns)
    p_row = pd.DataFrame(p_row_values, index=similarity.index, columns=similarity.columns)
    product = p_col * p_row
    return p_col, p_row, product


def _cosine_similarity(target: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    target = _frame(target, "target")
    reference = _frame(reference, "reference")
    genes = [gene for gene in target.columns if gene in reference.columns]
    if not genes:
        raise ValueError("target and reference have no overlapping genes")
    target_values = target.loc[:, genes].to_numpy(dtype=float)
    reference_values = reference.loc[:, genes].to_numpy(dtype=float)
    target_norms = np.linalg.norm(target_values, axis=1)
    reference_norms = np.linalg.norm(reference_values, axis=1)
    zero_targets = [str(target.index[index]) for index, norm in enumerate(target_norms) if norm == 0]
    zero_references = [str(reference.index[index]) for index, norm in enumerate(reference_norms) if norm == 0]
    target_unit = np.divide(target_values, target_norms[:, None], out=np.zeros_like(target_values), where=target_norms[:, None] != 0)
    reference_unit = np.divide(reference_values, reference_norms[:, None], out=np.zeros_like(reference_values), where=reference_norms[:, None] != 0)
    similarity = pd.DataFrame(target_unit @ reference_unit.T, index=target.index, columns=reference.index)
    return similarity, genes, zero_targets, zero_references


def attention(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    normalization: str,
    temperature: float | None = None,
) -> AttentionResult:
    """Compute cosine similarity, both normalizations, and L."""

    similarity, genes, zero_targets, zero_references = _cosine_similarity(target, reference)
    if normalization == "l1":
        p_col, p_row, product = l1_normalize(similarity)
    elif normalization == "softmax":
        p_col, p_row, product = softmax_normalize(similarity, 1.0 if temperature is None else temperature)
    else:
        raise ValueError("normalization must be l1 or softmax")
    return AttentionResult(
        S=similarity,
        P_col=p_col,
        P_row=p_row,
        L=product,
        used_genes=genes,
        zero_target_cells=zero_targets,
        zero_reference_identities=zero_references,
    )


normalize_l1 = l1_normalize
normalize_softmax = softmax_normalize

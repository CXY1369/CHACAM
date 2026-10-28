"""Cell-pair communication scoring and M2 L-R selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _all_columns_numeric_dtype(expression: pd.DataFrame) -> bool:
    return all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes)


def _numeric_expression(expression: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame")
    if _all_columns_numeric_dtype(expression):
        numeric = expression.copy()
    else:
        numeric = expression.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if numeric.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("expression contains non-finite values")
    index_names = [str(value) for value in numeric.index]
    column_names = [str(value) for value in numeric.columns]
    duplicate_index = pd.Index(index_names)[pd.Index(index_names).duplicated()].unique().tolist()
    duplicate_columns = pd.Index(column_names)[pd.Index(column_names).duplicated()].unique().tolist()
    if duplicate_index:
        raise ValueError(f"expression index has duplicate names after string conversion: {duplicate_index}")
    if duplicate_columns:
        raise ValueError(f"expression columns have duplicate names after string conversion: {duplicate_columns}")
    numeric.index = index_names
    numeric.columns = column_names
    return numeric


def _transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        if (values < 0).any():
            raise ValueError("log1p transform requires nonnegative expression values")
        return np.log1p(values)
    raise ValueError("transform must be none or log1p")


def _prepare_score_pairs(
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    transform: str,
    direction: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    expression = _numeric_expression(expression)
    if not isinstance(lr_pairs, pd.DataFrame):
        raise TypeError("lr_pairs must be a pandas DataFrame")
    required = {"ligand", "receptor", "weight"}
    missing = required.difference(lr_pairs.columns)
    if missing:
        raise ValueError(f"lr_pairs is missing required columns: {', '.join(sorted(missing))}")
    if direction not in {"both", "ref_to_cand", "cand_to_ref"}:
        raise ValueError("direction must be both, ref_to_cand, or cand_to_ref")
    if transform not in {"none", "log1p"}:
        raise ValueError("transform must be none or log1p")
    reserved_output_columns = {"reference_cell", "bucket_cell", "contribution"}
    conflicts = sorted(reserved_output_columns.intersection(lr_pairs.columns))
    if conflicts:
        raise ValueError(f"lr_pairs contains reserved output columns: {', '.join(conflicts)}")

    gene_names = set(expression.columns.astype(str))
    lr_rows = lr_pairs.to_dict(orient="records")
    for lr_row in lr_rows:
        ligand = str(lr_row["ligand"])
        receptor = str(lr_row["receptor"])
        if ligand not in gene_names or receptor not in gene_names:
            raise ValueError(f"L-R genes are absent from expression: {ligand}, {receptor}")
        weight = pd.to_numeric(lr_row["weight"], errors="coerce")
        if pd.isna(weight) or not np.isfinite(float(weight)):
            raise ValueError("lr_pairs weight values must be finite numbers")
    return expression, lr_rows


def _require_expression_cells(expression: pd.DataFrame, cell_ids: Sequence[str]) -> None:
    expression_ids = set(expression.index.astype(str))
    missing_cells = [cell_id for cell_id in cell_ids if cell_id not in expression_ids]
    if missing_cells:
        raise ValueError(f"cell IDs are absent from expression: {', '.join(dict.fromkeys(missing_cells))}")


def score_pairs(
    expression: pd.DataFrame,
    cell_pairs: Sequence[tuple[str, str]],
    lr_pairs: pd.DataFrame,
    transform: str,
    direction: str,
) -> pd.DataFrame:
    """Return one weighted contribution row for every requested pair and L-R pair."""

    expression, lr_rows = _prepare_score_pairs(expression, lr_pairs, transform, direction)
    normalized_pairs: list[tuple[str, str]] = []
    for reference_cell_raw, bucket_cell_raw in cell_pairs:
        reference_cell = str(reference_cell_raw)
        bucket_cell = str(bucket_cell_raw)
        _require_expression_cells(expression, (reference_cell, bucket_cell))
        normalized_pairs.append((reference_cell, bucket_cell))
    return _score_pairs_vectorized(expression, normalized_pairs, lr_pairs, lr_rows, transform, direction)


def _score_pairs_vectorized(
    expression: pd.DataFrame,
    cell_pairs: Sequence[tuple[str, str]],
    lr_pairs: pd.DataFrame,
    lr_rows: list[dict[str, object]],
    transform: str,
    direction: str,
) -> pd.DataFrame:
    columns = list(dict.fromkeys(["reference_cell", "bucket_cell", *lr_pairs.columns.tolist(), "contribution"]))
    if not cell_pairs or not lr_rows:
        return pd.DataFrame(columns=columns)

    ligands = [str(row["ligand"]) for row in lr_rows]
    receptors = [str(row["receptor"]) for row in lr_rows]
    weights = np.asarray([float(row["weight"]) for row in lr_rows], dtype=float)
    reference_cells = [pair[0] for pair in cell_pairs]
    bucket_cells = [pair[1] for pair in cell_pairs]
    lig_ref = _transform_values(expression.loc[reference_cells, ligands].to_numpy(dtype=float), transform)
    rec_bucket = _transform_values(expression.loc[bucket_cells, receptors].to_numpy(dtype=float), transform)
    lig_bucket = _transform_values(expression.loc[bucket_cells, ligands].to_numpy(dtype=float), transform)
    rec_ref = _transform_values(expression.loc[reference_cells, receptors].to_numpy(dtype=float), transform)
    ref_to_cand = (weights * lig_ref) * rec_bucket
    cand_to_ref = (weights * lig_bucket) * rec_ref
    if direction == "ref_to_cand":
        contributions = ref_to_cand
    elif direction == "cand_to_ref":
        contributions = cand_to_ref
    else:
        contributions = ref_to_cand + cand_to_ref

    rows: list[dict[str, object]] = []
    for pair_index, (reference_cell, bucket_cell) in enumerate(cell_pairs):
        for lr_index, lr_row in enumerate(lr_rows):
            row = {"reference_cell": reference_cell, "bucket_cell": bucket_cell}
            row.update(lr_row)
            row["contribution"] = float(contributions[pair_index, lr_index])
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def select_lr(lr_pairs: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Select the supplied L-R table for M2, or reserve training selection for M3."""

    if not isinstance(lr_pairs, pd.DataFrame):
        raise TypeError("lr_pairs must be a pandas DataFrame")
    conflicts = sorted({"selection", "effect"}.intersection(lr_pairs.columns))
    if conflicts:
        raise ValueError(f"lr_pairs contains reserved output columns: {', '.join(conflicts)}")
    if mode == "given":
        selected = lr_pairs.copy()
        selected["selection"] = "given"
        selected["effect"] = np.nan
        return selected
    if mode == "train_topk":
        raise NotImplementedError("train_topk is implemented in M3")
    raise ValueError("mode must be given or train_topk")

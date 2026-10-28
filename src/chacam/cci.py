"""Cell-pair communication scoring and L-R selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import _validate_contacts


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


def _require_expression_cells(
    expression: pd.DataFrame,
    cell_ids: Sequence[str],
    expression_ids: set[str] | None = None,
) -> None:
    if expression_ids is None:
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
    expression_ids = set(expression.index.astype(str))
    normalized_pairs: list[tuple[str, str]] = []
    for reference_cell_raw, bucket_cell_raw in cell_pairs:
        reference_cell = str(reference_cell_raw)
        bucket_cell = str(bucket_cell_raw)
        _require_expression_cells(expression, (reference_cell, bucket_cell), expression_ids)
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
        raise ValueError("train_topk requires training data; call train_topk")
    raise ValueError("mode must be given or train_topk")


@dataclass
class TopKSelection:
    table: pd.DataFrame
    n_contact_pairs: int
    n_noncontact_pairs: int


def _empty_topk_table(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    table = lr_pairs.iloc[0:0].copy()
    table["selection"] = pd.Series(dtype=object)
    table["effect"] = pd.Series(dtype=float)
    return table


def _contact_grid(contacts: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    pair_values: dict[tuple[str, str], float] = {}
    for _, row in contacts.iterrows():
        key = tuple(sorted((str(row["identity_a"]), str(row["identity_b"]))))
        pair_values[key] = float(row["contact"])
    identities = sorted({name for pair in pair_values for name in pair})
    index = {name: position for position, name in enumerate(identities)}
    grid = np.full((len(identities), len(identities)), np.nan)
    for (first, second), value in pair_values.items():
        i = index[first]
        j = index[second]
        grid[i, j] = grid[j, i] = value
    return index, grid


def train_topk(
    expression: pd.DataFrame,
    cells: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    contacts: pd.DataFrame,
    group_names: Sequence[str],
    transform: str,
    top_k: int,
) -> TopKSelection:
    """Select L-R pairs by contact minus non-contact mean scores on training cells."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if transform not in {"none", "log1p"}:
        raise ValueError("transform must be none or log1p")
    if not isinstance(expression, pd.DataFrame) or not isinstance(cells, pd.DataFrame):
        raise TypeError("expression and cells must be pandas DataFrames")
    if not isinstance(lr_pairs, pd.DataFrame) or not isinstance(contacts, pd.DataFrame):
        raise TypeError("lr_pairs and contacts must be pandas DataFrames")
    expression = expression.copy(deep=False)
    expression.index = [str(value) for value in expression.index]
    expression.columns = [str(value) for value in expression.columns]
    duplicate_expression_index = expression.index[expression.index.duplicated()].unique().tolist()
    duplicate_expression_columns = expression.columns[expression.columns.duplicated()].unique().tolist()
    if duplicate_expression_index:
        raise ValueError(
            f"expression index has duplicate names after string conversion: {duplicate_expression_index}"
        )
    if duplicate_expression_columns:
        raise ValueError(
            f"expression columns have duplicate names after string conversion: {duplicate_expression_columns}"
        )
    cells = cells.copy(deep=False)
    cells.index = [str(value) for value in cells.index]
    duplicate_cell_ids = cells.index[cells.index.duplicated()].unique().tolist()
    if duplicate_cell_ids:
        raise ValueError(f"cells index has duplicate names after string conversion: {duplicate_cell_ids}")
    missing_cell_columns = {"sample_id", "label"}.difference(cells.columns)
    if missing_cell_columns:
        raise ValueError(f"cells is missing required columns: {', '.join(sorted(missing_cell_columns))}")
    expression_ids = set(expression.index)
    missing_cells = [cell_id for cell_id in cells.index if cell_id not in expression_ids]
    if missing_cells:
        raise ValueError(f"cell IDs are absent from expression: {', '.join(missing_cells[:5])}")
    reserved = sorted({"selection", "effect"}.intersection(lr_pairs.columns))
    if reserved:
        raise ValueError(f"lr_pairs contains reserved output columns: {', '.join(reserved)}")
    required = {"ligand", "receptor"}
    missing_lr = required.difference(lr_pairs.columns)
    if missing_lr:
        raise ValueError(f"lr_pairs is missing required columns: {', '.join(sorted(missing_lr))}")
    if "weight" not in lr_pairs.columns:
        lr_pairs = lr_pairs.copy()
        lr_pairs["weight"] = 1.0
    if not {"identity_a", "identity_b", "contact"}.issubset(contacts.columns):
        raise ValueError("contacts must contain identity_a, identity_b, and contact columns")
    lr_table = lr_pairs.copy()
    ligands = lr_table["ligand"].astype(str).tolist()
    receptors = lr_table["receptor"].astype(str).tolist()
    lr_table["ligand"] = ligands
    lr_table["receptor"] = receptors
    gene_names = set(expression.columns)
    for ligand, receptor in zip(ligands, receptors):
        if ligand not in gene_names or receptor not in gene_names:
            raise ValueError(f"L-R genes are absent from expression: {ligand}, {receptor}")

    weights = pd.to_numeric(lr_table["weight"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(weights).any() or not np.isfinite(weights).all():
        raise ValueError("lr_pairs weight values must be finite numbers")
    group_name_set = {str(name) for name in group_names}
    contacts = _validate_contacts(contacts, ())
    identity_index, grid = _contact_grid(contacts)
    used_genes = list(dict.fromkeys([*ligands, *receptors]))
    used_labels = cells["label"].astype(str)
    used_cell_ids = cells.index[~used_labels.isin(group_name_set)].tolist()
    numeric_used = expression.loc[used_cell_ids, used_genes].apply(pd.to_numeric, errors="coerce")
    used_values = numeric_used.to_numpy(dtype=float)
    if numeric_used.isna().any().any() or not np.isfinite(used_values).all():
        raise ValueError("expression contains non-finite values")
    n_identities = len(identity_index)
    n_lr = len(lr_table)
    contact_sum = np.zeros(n_lr, dtype=float)
    noncontact_sum = np.zeros(n_lr, dtype=float)
    n_contact_pairs = 0
    n_noncontact_pairs = 0

    for _sample_id, sample_cells in cells.groupby(cells["sample_id"].astype(str), sort=False):
        labels = sample_cells["label"].astype(str)
        fine_cells = sample_cells.index[ ~labels.isin(group_name_set)].astype(str)
        if len(fine_cells) < 2:
            continue
        fine_labels = labels.loc[fine_cells].astype(str).to_numpy()
        mapped = np.array([identity_index.get(label, n_identities) for label in fine_labels])
        grid_ext = np.full((n_identities + 1, n_identities + 1), np.nan)
        if n_identities:
            grid_ext[:n_identities, :n_identities] = grid
        values = grid_ext[mapped[:, None], mapped[None, :]]
        contact_mask = np.isfinite(values) & (values > 0)
        noncontact_mask = np.isfinite(values) & (values == 0)
        np.fill_diagonal(contact_mask, False)
        np.fill_diagonal(noncontact_mask, False)
        ligand_values = _transform_values(numeric_used.loc[fine_cells, ligands].to_numpy(dtype=float), transform)
        receptor_values = _transform_values(numeric_used.loc[fine_cells, receptors].to_numpy(dtype=float), transform)

        def class_sum(mask: np.ndarray) -> np.ndarray:
            mixed = ligand_values * (mask.astype(float) @ receptor_values)
            return weights * mixed.sum(axis=0)

        n_contact = int(contact_mask.sum() // 2)
        n_noncontact = int(noncontact_mask.sum() // 2)
        if n_contact:
            contact_sum += class_sum(contact_mask)
            n_contact_pairs += n_contact
        if n_noncontact:
            noncontact_sum += class_sum(noncontact_mask)
            n_noncontact_pairs += n_noncontact

    if n_contact_pairs == 0 or n_noncontact_pairs == 0:
        return TopKSelection(
            table=_empty_topk_table(lr_table),
            n_contact_pairs=n_contact_pairs,
            n_noncontact_pairs=n_noncontact_pairs,
        )
    effect = contact_sum / n_contact_pairs - noncontact_sum / n_noncontact_pairs
    table = lr_table.copy()
    table["selection"] = "train_topk"
    table["effect"] = effect
    table = table.loc[table["effect"] > 0]
    table = table.sort_values("effect", ascending=False, kind="mergesort")
    table = table.head(top_k).reset_index(drop=True)
    return TopKSelection(table=table, n_contact_pairs=n_contact_pairs, n_noncontact_pairs=n_noncontact_pairs)

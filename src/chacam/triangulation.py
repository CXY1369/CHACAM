"""Binary CCI triangulation for ambiguity buckets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cci import _prepare_score_pairs, _require_expression_cells, _score_pairs_vectorized
from .schema import AmbiguityGroup


@dataclass
class TriangulationResult:
    labels: pd.Series
    status: pd.Series
    reasons: pd.Series
    evidence: pd.DataFrame


def _evidence_columns(lr_table: pd.DataFrame) -> list[str]:
    base = ["sample_id", "group", "reference_identity", "reference_cell", "bucket_cell"]
    lr_columns = [
        str(column)
        for column in lr_table.columns
        if column not in {"reference_cell", "bucket_cell", "contribution"}
    ]
    return list(dict.fromkeys([*base, *lr_columns, "contribution"]))


def _empty_evidence(lr_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_evidence_columns(lr_table))


def _contact_value(contacts: pd.DataFrame, first: str, second: str) -> float:
    if not {"identity_a", "identity_b", "contact"}.issubset(contacts.columns):
        raise ValueError("contacts must contain identity_a, identity_b, and contact columns")
    first_values = contacts["identity_a"].astype(str)
    second_values = contacts["identity_b"].astype(str)
    mask = ((first_values == first) & (second_values == second)) | (
        (first_values == second) & (second_values == first)
    )
    matches = contacts.loc[mask, "contact"]
    if matches.empty:
        raise ValueError(f"missing contact record for {first} and {second}")
    return float(matches.iloc[0])


def _group_parts(group: AmbiguityGroup) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not hasattr(group, "name") or not hasattr(group, "candidates") or not hasattr(group, "references"):
        raise TypeError("group must expose name, candidates, and references")
    return str(group.name), tuple(str(value) for value in group.candidates), tuple(str(value) for value in group.references)


def triangulate(
    stage1_labels: pd.Series,
    expression: pd.DataFrame,
    group: AmbiguityGroup,
    contacts: pd.DataFrame,
    lr_table: pd.DataFrame,
    transform: str,
    direction: str,
    *,
    sample_id: str,
) -> TriangulationResult:
    """Triangulate a two-cell ambiguity bucket with unanimous reference votes."""

    if not isinstance(stage1_labels, pd.Series):
        raise TypeError("stage1_labels must be a pandas Series")
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame")
    if not isinstance(contacts, pd.DataFrame) or not isinstance(lr_table, pd.DataFrame):
        raise TypeError("contacts and lr_table must be pandas DataFrames")
    conflicts = sorted({"sample_id", "group", "reference_identity"}.intersection(lr_table.columns))
    if conflicts:
        raise ValueError(f"lr_table contains reserved evidence columns: {', '.join(conflicts)}")
    group_name, candidates, references = _group_parts(group)
    bucket_cells = [str(cell_id) for cell_id in stage1_labels.index[stage1_labels.astype(str) == group_name]]
    labels = pd.Series(group_name, index=bucket_cells, dtype=object, name="final_label")
    status = pd.Series("unresolved", index=bucket_cells, dtype=object, name="status")
    reasons = pd.Series("", index=bucket_cells, dtype=object, name="unresolved_reason")
    empty_evidence = _empty_evidence(lr_table)

    if len(bucket_cells) != 2:
        reasons[:] = "unresolved_bucket_size"
        return TriangulationResult(labels=labels, status=status, reasons=reasons, evidence=empty_evidence)
    if lr_table.empty:
        reasons[:] = "unresolved_no_lr"
        return TriangulationResult(labels=labels, status=status, reasons=reasons, evidence=empty_evidence)

    evidence_frames: list[pd.DataFrame] = []
    votes: list[dict[str, str]] = []
    tie_found = False
    stage1_values = stage1_labels.astype(str)
    prepared_expression: pd.DataFrame | None = None
    lr_rows: list[dict[str, object]] = []
    lr_genes: list[str] = []
    for reference_identity in references:
        reference_cells = [str(cell_id) for cell_id in stage1_values.index[stage1_values == reference_identity]]
        if not reference_cells:
            continue
        contact_values = {
            candidate: _contact_value(contacts, reference_identity, candidate)
            for candidate in candidates
        }
        positive_candidates = [candidate for candidate, value in contact_values.items() if value > 0]
        if len(positive_candidates) != 1:
            raise ValueError(
                f"reference {reference_identity} must contact exactly one candidate in group {group_name}"
            )
        positive_candidate = positive_candidates[0]
        negative_candidate = next(candidate for candidate in candidates if candidate != positive_candidate)
        if prepared_expression is None:
            prepared_expression, lr_rows = _prepare_score_pairs(expression, lr_table, transform, direction)
            lr_genes = list(
                dict.fromkeys([str(row["ligand"]) for row in lr_rows] + [str(row["receptor"]) for row in lr_rows])
            )
        pairs = [
            (reference_cell, bucket_cell)
            for reference_cell in reference_cells
            for bucket_cell in bucket_cells
        ]
        for reference_cell, bucket_cell in pairs:
            _require_expression_cells(prepared_expression, (reference_cell, bucket_cell))
        sliced = prepared_expression.loc[[*reference_cells, *bucket_cells], lr_genes]
        scored = _score_pairs_vectorized(sliced, pairs, lr_table, lr_rows, transform, direction).copy()
        scored["contribution"] = scored["contribution"] / len(reference_cells)
        scored.insert(0, "sample_id", sample_id)
        scored.insert(1, "group", group_name)
        scored.insert(2, "reference_identity", reference_identity)
        reference_evidence = scored
        evidence_frames.append(reference_evidence)
        totals = reference_evidence.groupby("bucket_cell", sort=False)["contribution"].sum()
        first_score = float(totals.loc[bucket_cells[0]])
        second_score = float(totals.loc[bucket_cells[1]])
        if np.isclose(first_score, second_score):
            tie_found = True
            continue
        high_cell, low_cell = (
            (bucket_cells[0], bucket_cells[1])
            if first_score > second_score
            else (bucket_cells[1], bucket_cells[0])
        )
        votes.append({high_cell: positive_candidate, low_cell: negative_candidate})

    evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else empty_evidence
    evidence = evidence.reindex(columns=_evidence_columns(lr_table))
    if not votes:
        reasons[:] = "unresolved_no_reference" if not evidence_frames else "unresolved_tie"
    elif tie_found:
        reasons[:] = "unresolved_tie"
    elif any(vote != votes[0] for vote in votes[1:]):
        reasons[:] = "unresolved_reference_conflict"
    else:
        for bucket_cell, final_label in votes[0].items():
            labels.loc[bucket_cell] = final_label
            status.loc[bucket_cell] = "resolved"
        reasons[:] = ""
    return TriangulationResult(labels=labels, status=status, reasons=reasons, evidence=evidence)

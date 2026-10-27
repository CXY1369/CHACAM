"""Capacity-aware greedy assignment for the Fig 2 L matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AssignmentResult:
    labels: pd.Series
    trace: pd.DataFrame

    @property
    def assignments(self) -> pd.Series:
        return self.labels


def _group_name_and_candidates(group: object) -> tuple[str, list[str]]:
    if hasattr(group, "name") and hasattr(group, "candidates"):
        return str(group.name), [str(candidate) for candidate in group.candidates]
    if isinstance(group, dict):
        return str(group["name"]), [str(candidate) for candidate in group["candidates"]]
    raise ValueError("ambiguity_groups entries must expose name and candidates")


def capacities(identities: Any, ambiguity_groups: Any, mode: str) -> dict[str, float | int]:
    """Construct per-column capacities for unique or many-to-one assignment."""

    identities = [str(identity) for identity in identities]
    if mode not in {"unique", "many_to_one"}:
        raise ValueError("mode must be unique or many_to_one")
    if mode == "many_to_one":
        return {identity: float("inf") for identity in identities}
    group_capacities = {}
    for group in ambiguity_groups:
        name, candidates = _group_name_and_candidates(group)
        group_capacities[name] = len(candidates)
    return {identity: group_capacities.get(identity, 1) for identity in identities}


def greedy_assign(L: pd.DataFrame, capacities: dict[str, float | int]) -> AssignmentResult:
    """Greedily assign the largest positive available L entry."""

    if not isinstance(L, pd.DataFrame):
        raise TypeError("L must be a pandas DataFrame")
    values = L.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("L contains non-finite values")
    missing_capacities = [identity for identity in L.columns if identity not in capacities]
    if missing_capacities:
        raise ValueError(f"capacities missing identities: {', '.join(map(str, missing_capacities))}")
    remaining = {identity: capacities[identity] for identity in L.columns}
    labels = pd.Series("unassigned", index=L.index, dtype=object, name="label")
    unassigned_rows = list(range(len(L.index)))
    trace_rows: list[dict[str, object]] = []
    order = 1
    matrix = values.to_numpy(dtype=float)
    while unassigned_rows:
        available_columns = [column_index for column_index, identity in enumerate(L.columns) if remaining[identity] > 0]
        if not available_columns:
            break
        available = []
        for row_index in unassigned_rows:
            for column_index in available_columns:
                value = matrix[row_index, column_index]
                if value > 0:
                    available.append((row_index, column_index, value))
        if not available:
            break
        maximum = max(item[2] for item in available)
        tied = [item for item in available if np.isclose(item[2], maximum, rtol=1e-12, atol=1e-12)]
        row_index, column_index, value = tied[0]
        cell_id = L.index[row_index]
        identity = L.columns[column_index]
        tie = len(tied) > 1
        labels.iloc[row_index] = identity
        trace_rows.append(
            {
                "order": order,
                "cell_id": cell_id,
                "identity": identity,
                "L": float(value),
                "tie": tie,
                "status": "tie" if tie else "selected",
            }
        )
        order += 1
        unassigned_rows.remove(row_index)
        if not np.isinf(remaining[identity]):
            remaining[identity] -= 1
    trace = pd.DataFrame(trace_rows, columns=["order", "cell_id", "identity", "L", "tie", "status"])
    return AssignmentResult(labels=labels, trace=trace)

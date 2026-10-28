"""Capacity-aware greedy assignment for the Fig 2 L matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schema import RESERVED_LABEL


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
    if RESERVED_LABEL in [str(column) for column in L.columns]:
        raise ValueError(f"{RESERVED_LABEL!r} is a reserved label name; rename the identity column")
    values = L.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("L contains non-finite values")
    missing_capacities = [identity for identity in L.columns if identity not in capacities]
    if missing_capacities:
        raise ValueError(f"capacities missing identities: {', '.join(map(str, missing_capacities))}")
    remaining = {identity: capacities[identity] for identity in L.columns}
    labels = pd.Series(RESERVED_LABEL, index=L.index, dtype=object, name="label")
    trace_rows: list[dict[str, object]] = []
    order = 1
    matrix = values.to_numpy(dtype=float)
    row_indices, column_indices = np.nonzero(matrix > 0)
    item_values = matrix[row_indices, column_indices]
    order_index = np.lexsort((column_indices, row_indices, -item_values))
    items = [
        (int(row_indices[index]), int(column_indices[index]), float(item_values[index]))
        for index in order_index
    ]
    row_assigned = np.zeros(len(L.index), dtype=bool)
    assigned_count = 0
    start = 0
    n_items = len(items)
    n_rows = len(L.index)
    while start < n_items and assigned_count < n_rows:
        while start < n_items:
            row_index, column_index, _value = items[start]
            identity = L.columns[column_index]
            if not row_assigned[row_index] and remaining[identity] > 0:
                break
            start += 1
        if start >= n_items:
            break
        top = items[start][2]
        tolerance = 1e-12 + 1e-12 * abs(top)
        tied: list[tuple[int, int, float]] = []
        cursor = start
        while cursor < n_items:
            row_index, column_index, value = items[cursor]
            if abs(value - top) > tolerance:
                break
            identity = L.columns[column_index]
            if not row_assigned[row_index] and remaining[identity] > 0:
                tied.append((row_index, column_index, value))
            cursor += 1
        row_index, column_index, value = min(tied, key=lambda item: (item[0], item[1]))
        identity = L.columns[column_index]
        tie = len(tied) > 1
        labels.iloc[row_index] = identity
        trace_rows.append(
            {
                "order": order,
                "cell_id": L.index[row_index],
                "identity": identity,
                "L": float(value),
                "tie": tie,
                "status": "tie" if tie else "selected",
            }
        )
        order += 1
        row_assigned[row_index] = True
        assigned_count += 1
        if not np.isinf(remaining[identity]):
            remaining[identity] -= 1
    trace = pd.DataFrame(trace_rows, columns=["order", "cell_id", "identity", "L", "tie", "status"])
    return AssignmentResult(labels=labels, trace=trace)

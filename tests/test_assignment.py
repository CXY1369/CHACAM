import math

import numpy as np
import pandas as pd
import pytest

from chacam.assignment import AssignmentResult, capacities, greedy_assign
from chacam.schema import RESERVED_LABEL, AmbiguityGroup


GROUP = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1", "R2"))


def test_unique_capacities_give_ambiguity_group_candidate_count():
    result = capacities(["Ax", "R1", "R2"], [GROUP], mode="unique")
    assert result == {"Ax": 2, "R1": 1, "R2": 1}


def test_many_to_one_capacities_are_infinite():
    result = capacities(["Ax", "R1"], [GROUP], mode="many_to_one")
    assert all(math.isinf(value) for value in result.values())


def test_fig2_greedy_assignment_uses_expected_global_order():
    l = pd.DataFrame(
        [
            [0.0077, 0.32, 0.0014, 0.12],
            [0.045, 0.0056, 0.091, 0.18],
            [2.8e-07, 0.0044, 0.28, 0.092],
            [0.33, 0, 0.055, 0.022],
        ],
        index=["c1", "c2", "c3", "c4"],
        columns=["P2", "EMS", "ABp", "ABa"],
    )
    result = greedy_assign(l, {identity: 1 for identity in l.columns})
    assert result.labels.to_dict() == {"c1": "EMS", "c2": "ABa", "c3": "ABp", "c4": "P2"}
    assert result.trace["cell_id"].tolist() == ["c4", "c1", "c3", "c2"]


def test_ambiguity_capacity_prevents_deadlock_and_accepts_two_cells():
    l = pd.DataFrame(
        [[0.9, 0.8, 0.7], [0.85, 0.6, 0.5], [0.4, 0.95, 0.3], [0.3, 0.2, 0.96]],
        index=["c1", "c2", "c3", "c4"], columns=["Ax", "R1", "R2"],
    )
    result = greedy_assign(l, capacities(l.columns, [GROUP], "unique"))
    assert (result.labels == "Ax").sum() == 2
    assert (result.labels == "R1").sum() == 1
    assert (result.labels == "R2").sum() == 1


def test_many_to_one_matches_rowwise_argmax():
    l = pd.DataFrame([[0.1, 0.8], [0.7, 0.2], [0.6, 0.9]], index=["c1", "c2", "c3"], columns=["a", "b"])
    result = greedy_assign(l, capacities(l.columns, [], "many_to_one"))
    expected = l.idxmax(axis=1).astype(object)
    pd.testing.assert_series_equal(result.labels, expected.rename(result.labels.name), check_names=False)


def test_zero_values_and_insufficient_capacity_are_unassigned():
    l = pd.DataFrame([[0.0, 0.0], [0.8, 0.1], [0.7, 0.2]], index=["zero", "c1", "c2"], columns=["a", "b"])
    result = greedy_assign(l, {"a": 1, "b": 1})
    assert result.labels.loc["zero"] == "unassigned"
    assert (result.labels == "unassigned").sum() == 1


def test_positive_cells_beyond_exhausted_unique_capacity_are_unassigned():
    l = pd.DataFrame([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]], index=["c1", "c2", "c3"], columns=["a", "b"])
    result = greedy_assign(l, {"a": 1, "b": 1})
    assert (result.labels != "unassigned").sum() == 2
    assert set(result.labels[result.labels != "unassigned"]) == {"a", "b"}
    assert result.labels.loc["c3"] == "unassigned"


def test_exact_ties_are_deterministic_and_marked_in_trace():
    l = pd.DataFrame([[0.5, 0.5], [0.5, 0.1]], index=["c1", "c2"], columns=["a", "b"])
    first = greedy_assign(l, {"a": 1, "b": 1})
    second = greedy_assign(l, {"a": 1, "b": 1})
    pd.testing.assert_series_equal(first.labels, second.labels)
    assert first.trace.iloc[0]["cell_id"] == "c1"
    assert bool(first.trace.iloc[0]["tie"])
    assert first.trace.iloc[0]["status"] == "tie"


def test_greedy_assign_rejects_reserved_unassigned_identity_column():
    l = pd.DataFrame([[1.0]], index=["c1"], columns=["unassigned"])

    with pytest.raises(ValueError, match="unassigned"):
        greedy_assign(l, {"unassigned": 1})


def test_reserved_name_check_is_case_sensitive_for_assignment():
    l = pd.DataFrame([[1.0]], index=["c1"], columns=["Unassigned"])

    result = greedy_assign(l, {"Unassigned": 1})

    assert result.labels.loc["c1"] == "Unassigned"


def _reference_greedy_assign(L: pd.DataFrame, capacities: dict[str, float | int]) -> AssignmentResult:
    """Exact copy of the a3fc6e7 greedy_assign implementation."""

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


def _assert_same_assignment(left: AssignmentResult, right: AssignmentResult) -> None:
    assert left.labels.equals(right.labels)
    assert left.trace.equals(right.trace)


def test_greedy_assign_near_tie_keeps_later_available_item_in_window():
    l = pd.DataFrame(
        [[1.0 - 5e-13, 0.0], [0.0, 1.0]],
        index=["r0", "r1"],
        columns=["A", "B"],
    )
    result = greedy_assign(l, {"A": 1, "B": 1})
    expected = _reference_greedy_assign(l, {"A": 1, "B": 1})

    _assert_same_assignment(result, expected)
    assert result.labels.to_dict() == {"r0": "A", "r1": "B"}
    assert result.trace["cell_id"].tolist() == ["r0", "r1"]
    assert result.trace["identity"].tolist() == ["A", "B"]
    assert result.trace["L"].tolist() == [1.0 - 5e-13, 1.0]
    assert result.trace["tie"].tolist() == [True, False]
    assert result.trace["status"].tolist() == ["tie", "selected"]
    assert result.trace["order"].tolist() == [1, 2]


def test_greedy_assign_two_by_two_diagonal_trace_is_exact():
    l = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["r0", "r1"], columns=["A", "B"])
    result = greedy_assign(l, {"A": 1, "B": 1})
    expected = _reference_greedy_assign(l, {"A": 1, "B": 1})

    _assert_same_assignment(result, expected)
    assert result.labels.to_dict() == {"r0": "A", "r1": "B"}
    assert result.trace["cell_id"].tolist() == ["r0", "r1"]
    assert result.trace["identity"].tolist() == ["A", "B"]
    assert result.trace["L"].tolist() == [1.0, 1.0]
    assert result.trace["tie"].tolist() == [True, False]
    assert result.trace["status"].tolist() == ["tie", "selected"]


def test_greedy_assign_uses_position_not_alphabetical_names():
    l = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0 + 1e-13]],
        index=["b", "a"],
        columns=["m", "a"],
    )
    result = greedy_assign(l, {"m": 1, "a": 1})
    expected = _reference_greedy_assign(l, {"m": 1, "a": 1})

    _assert_same_assignment(result, expected)
    assert list(result.labels.index) == ["b", "a"]
    assert result.labels.to_dict() == {"b": "m", "a": "a"}
    assert result.trace["cell_id"].tolist() == ["b", "a"]
    assert result.trace["identity"].tolist() == ["m", "a"]
    assert result.trace["L"].tolist() == [1.0, 1.0 + 1e-13]
    assert bool(result.trace.iloc[0]["tie"])
    assert result.trace.iloc[0]["status"] == "tie"


def test_greedy_assign_all_equal_matrix_uses_row_then_column_position():
    names = ["d", "c", "b", "a"]
    l = pd.DataFrame(np.ones((4, 4)), index=names, columns=list(reversed(names)))
    result = greedy_assign(l, {column: 1 for column in l.columns})
    expected = _reference_greedy_assign(l, {column: 1 for column in l.columns})

    _assert_same_assignment(result, expected)
    assert result.labels.to_dict() == {"d": "a", "c": "b", "b": "c", "a": "d"}
    assert result.trace["cell_id"].tolist() == ["d", "c", "b", "a"]
    assert result.trace["identity"].tolist() == ["a", "b", "c", "d"]
    assert result.trace["L"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result.trace["tie"].tolist() == [True, True, True, False]
    assert result.trace["status"].tolist() == ["tie", "tie", "tie", "selected"]


@pytest.mark.parametrize("seed", range(200))
def test_greedy_assign_matches_reference_on_random_matrices(seed):
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(2, 13))
    n_cols = int(rng.integers(2, 13))
    matrix = rng.normal(loc=0.2, scale=0.5, size=(n_rows, n_cols))
    if seed % 5 == 0:
        matrix[:, 0] = matrix[0, 0]
    if seed % 3 == 0:
        matrix[matrix < 0] = 0.0
    if seed % 7 == 0:
        row, col = int(rng.integers(n_rows)), int(rng.integers(n_cols))
        peak = float(np.max(matrix)) if matrix.size else 1.0
        matrix[row, col] = peak + float(rng.choice([-1e-14, 1e-14]))
    row_names = [f"r{index}" for index in range(n_rows - 1, -1, -1)]
    col_names = [f"c{index}" for index in range(n_cols - 1, -1, -1)]
    l = pd.DataFrame(matrix, index=row_names, columns=col_names)
    capacity_choices = (1, 2, float("inf"))
    column_capacities = {name: capacity_choices[int(rng.integers(3))] for name in col_names}

    result = greedy_assign(l, column_capacities)
    expected = _reference_greedy_assign(l, column_capacities)
    _assert_same_assignment(result, expected)

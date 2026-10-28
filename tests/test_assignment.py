import math

import numpy as np
import pandas as pd
import pytest

from chacam.assignment import capacities, greedy_assign
from chacam.schema import AmbiguityGroup


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

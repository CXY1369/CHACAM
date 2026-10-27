import pandas as pd

from chacam.assignment import capacities, greedy_assign
from chacam.attention import attention
from chacam.reference import training_mean
from chacam.synthetic import make_synthetic


def test_synthetic_leave_one_sample_smoke_assignment():
    data = make_synthetic(seed=17, n_samples=8)
    target_sample = data.cells["sample_id"].iloc[0]
    target_cells = data.cells[data.cells["sample_id"] == target_sample].set_index("cell_id")
    training_cells = data.cells[data.cells["sample_id"] != target_sample].set_index("cell_id")
    expression = data.expression.set_index("cell_id")
    genes = list(data.marker_genes)
    profiles = training_mean(expression.loc[training_cells.index], training_cells["label"], genes, expected_labels=sorted(training_cells["label"].unique()))
    result = attention(expression.loc[target_cells.index, genes], profiles, normalization="l1")
    assigned = greedy_assign(result.L, capacities(profiles.index, [{"name": "Ax", "candidates": ["A1", "A2"], "references": ["R1", "R2"]}], "unique"))

    expected = target_cells["label"]
    nonambiguous = expected != "Ax"
    assert (assigned.labels[nonambiguous] == expected[nonambiguous]).mean() >= 0.9
    assert (assigned.labels == "Ax").sum() == 2

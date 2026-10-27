from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from chacam.schema import load_config, load_inputs
from chacam.synthetic import make_synthetic, write_synthetic


def _synthetic_config(tmp_path: Path):
    config = {
        "provenance": "framework_baseline",
        "seed": 11,
        "inputs": {
            "expression": "expression.csv",
            "cells": "cells.csv",
            "features": "features.csv",
            "lr_pairs": "lr_pairs.csv",
            "contacts": "contacts.csv",
            "expression_scale": "counts",
        },
        "reference": {"mode": "training_mean"},
        "attention": {"normalization": "l1"},
        "assignment": {"mode": "unique"},
        "cci": {"transform": "log1p", "direction": "both", "lr_selection": "given", "top_k": 2},
        "ambiguity_groups": [{"name": "Ax", "candidates": ["A1", "A2"], "references": ["R1", "R2"]}],
        "output_dir": "runs/test",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_synthetic_generation_is_reproducible_by_seed():
    first = make_synthetic(seed=13, n_samples=8)
    second = make_synthetic(seed=13, n_samples=8)
    different = make_synthetic(seed=14, n_samples=8)

    pd.testing.assert_frame_equal(first.expression, second.expression)
    pd.testing.assert_frame_equal(first.cells, second.cells)
    pd.testing.assert_frame_equal(first.truth, second.truth)
    assert not first.expression.equals(different.expression)


def test_synthetic_data_has_merged_observations_and_complete_truth():
    data = make_synthetic(seed=3, n_samples=4)
    assert set(data.cells.loc[data.cells["label"] == "Ax", "cell_id"]) == set(
        data.truth.loc[data.truth["true_label"].isin(["A1", "A2"]), "cell_id"]
    )
    assert set(data.truth["cell_id"]) == set(data.cells["cell_id"])
    assert all(data.cells.groupby("sample_id").size() == data.cells.groupby("sample_id")["label"].size())
    assert (data.expression.drop(columns="cell_id").to_numpy() >= 0).all()


def test_synthetic_write_output_passes_all_input_validation(tmp_path):
    data = make_synthetic(seed=5, n_samples=5)
    write_synthetic(data, tmp_path)
    inputs = load_inputs(load_config(_synthetic_config(tmp_path)))
    assert len(inputs.expression) == len(data.expression)
    assert len(inputs.lr_pairs) == len(data.lr_pairs)
    assert inputs.dropped_lr_pairs == []


def test_contact_signal_changes_contacted_candidate_receptor_expression_only():
    positive = make_synthetic(seed=21, n_samples=40, contact_signal=True)
    negative = make_synthetic(seed=21, n_samples=40, contact_signal=False)

    def candidate_receptor_difference(data):
        cells = data.cells.merge(data.truth, on="cell_id")
        means = data.expression.merge(cells[["cell_id", "true_label"]], on="cell_id").groupby("true_label").mean(numeric_only=True)
        return float(means.loc["A1", "receptor_A1"] - means.loc["A2", "receptor_A1"])

    assert candidate_receptor_difference(positive) > 8.0
    assert abs(candidate_receptor_difference(negative)) < 3.0


def test_synthetic_contacts_include_explicit_positive_and_negative_pairs():
    data = make_synthetic(seed=2, n_samples=3)
    assert set(data.contacts["contact"]) >= {0, 1}
    for reference, candidate in [("R1", "A1"), ("R2", "A2")]:
        row = data.contacts[
            ((data.contacts["identity_a"] == reference) & (data.contacts["identity_b"] == candidate))
            | ((data.contacts["identity_a"] == candidate) & (data.contacts["identity_b"] == reference))
        ]
        assert len(row) == 1 and row.iloc[0]["contact"] > 0

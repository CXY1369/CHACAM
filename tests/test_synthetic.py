from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
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


def _contact_lookup(data):
    return {
        frozenset((row.identity_a, row.identity_b)): float(row.contact)
        for row in data.contacts.itertuples()
    }


def _fine_identity_lr_effects(data):
    expression = data.expression.set_index("cell_id")
    cells = data.cells.set_index("cell_id")
    contacts = _contact_lookup(data)
    lr_pairs = list(data.lr_pairs.itertuples(index=False))
    scores = {f"{pair.ligand}/{pair.receptor}": {True: [], False: []} for pair in lr_pairs}
    for sample_id, sample_cells in cells.groupby("sample_id", sort=False):
        fine_cells = sample_cells[sample_cells["label"] != "Ax"]
        for first_id, second_id in combinations(fine_cells.index, 2):
            first_label = fine_cells.loc[first_id, "label"]
            second_label = fine_cells.loc[second_id, "label"]
            contact = contacts[frozenset((first_label, second_label))] > 0
            for pair in lr_pairs:
                weight = getattr(pair, "weight", 1.0)
                forward = np.log1p(expression.loc[first_id, pair.ligand]) * np.log1p(expression.loc[second_id, pair.receptor])
                reverse = np.log1p(expression.loc[second_id, pair.ligand]) * np.log1p(expression.loc[first_id, pair.receptor])
                scores[f"{pair.ligand}/{pair.receptor}"][contact].append(float(weight) * (forward + reverse))
    return {
        pair_name: np.mean(values[True]) - np.mean(values[False])
        for pair_name, values in scores.items()
    }


def test_synthetic_contact_signal_supports_d13_topk_learning():
    data = make_synthetic(seed=0, n_samples=20, contact_signal=True)
    effects = _fine_identity_lr_effects(data)
    signal_effects = [effects["lig_R1/rec_R1"], effects["lig_R2/rec_R2"]]
    decoy_effects = [effect for name, effect in effects.items() if "/rec_decoy" in name]

    assert min(signal_effects) > 0.5
    assert min(signal_effects) > max(decoy_effects)


def test_candidate_receptor_differences_depend_on_contact_signal():
    positive = make_synthetic(seed=21, n_samples=20, contact_signal=True)
    negative = make_synthetic(seed=21, n_samples=20, contact_signal=False)

    def candidate_receptor_means(data):
        cells = data.cells.merge(data.truth, on="cell_id")
        return data.expression.merge(cells[["cell_id", "true_label"]], on="cell_id").groupby("true_label").mean(numeric_only=True)

    positive_means = candidate_receptor_means(positive)
    negative_means = candidate_receptor_means(negative)
    assert positive_means.loc["A1", "rec_R1"] - positive_means.loc["A2", "rec_R1"] > 8.0
    assert positive_means.loc["A2", "rec_R2"] - positive_means.loc["A1", "rec_R2"] > 8.0
    assert abs(negative_means.loc["A1", "rec_R1"] - negative_means.loc["A2", "rec_R1"]) < 5.0
    assert abs(negative_means.loc["A2", "rec_R2"] - negative_means.loc["A1", "rec_R2"]) < 5.0


def test_contact_signal_modes_share_background_and_receptor_draw_totals():
    positive = make_synthetic(seed=21, n_samples=5, contact_signal=True)
    negative = make_synthetic(seed=21, n_samples=5, contact_signal=False)
    non_receptor_genes = [gene for gene in positive.expression.columns if gene != "cell_id" and not gene.startswith("rec_")]
    receptor_genes = [gene for gene in positive.expression.columns if gene.startswith("rec_")]
    pd.testing.assert_frame_equal(positive.expression[["cell_id", *non_receptor_genes]], negative.expression[["cell_id", *non_receptor_genes]])
    np.testing.assert_array_equal(
        positive.expression[receptor_genes].sum(axis=1).to_numpy(),
        negative.expression[receptor_genes].sum(axis=1).to_numpy(),
    )


def test_synthetic_contacts_include_explicit_positive_and_negative_pairs():
    data = make_synthetic(seed=2, n_samples=3, n_extra_identities=6)
    expected_positive = {
        frozenset(pair)
        for pair in [
            ("R1", "A1"), ("R2", "A2"),
            ("R1", "B1"), ("R1", "B2"), ("R1", "B3"),
            ("R2", "B3"), ("R2", "B4"), ("R2", "B5"),
            ("B1", "B2"), ("B3", "B4"), ("B5", "B6"),
        ]
    }
    actual = {
        frozenset((row.identity_a, row.identity_b)): float(row.contact)
        for row in data.contacts.itertuples()
    }
    fine_identities = {"R1", "R2", "B1", "B2", "B3", "B4", "B5", "B6"}
    fine_pairs = {pair: value for pair, value in actual.items() if pair.issubset(fine_identities)}
    assert len(actual) == 45
    assert len(fine_pairs) == 28
    assert {pair for pair, value in actual.items() if value > 0} == expected_positive
    assert all(value == 0 for pair, value in actual.items() if pair not in expected_positive)


def test_synthetic_requires_six_extra_identities():
    with pytest.raises(ValueError, match="six|6"):
        make_synthetic(seed=2, n_samples=3, n_extra_identities=5)


def test_synthetic_requires_at_least_one_decoy_pair():
    with pytest.raises(ValueError, match="n_decoy_pairs"):
        make_synthetic(seed=0, n_samples=2, n_decoy_pairs=0)

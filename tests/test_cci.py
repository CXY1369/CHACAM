import numpy as np
import pandas as pd
import pytest

from chacam.cci import _numeric_expression, score_pairs, select_lr, train_topk


@pytest.fixture
def cci_inputs():
    expression = pd.DataFrame(
        {
            "lig1": [1.0, 3.0, 100.0],
            "rec1": [2.0, 5.0, 100.0],
            "lig2": [4.0, 6.0, 100.0],
            "rec2": [8.0, 7.0, 100.0],
        },
        index=["ref", "bucket", "unused"],
    )
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["lig1", "lig2"],
            "receptor": ["rec1", "rec2"],
            "weight": [2.0, 0.5],
            "pathway": ["p1", "p2"],
        }
    )
    return expression, lr_pairs


@pytest.mark.parametrize(
    ("transform", "direction", "expected"),
    [
        ("none", "ref_to_cand", [10.0, 14.0]),
        ("none", "cand_to_ref", [12.0, 24.0]),
        ("none", "both", [22.0, 38.0]),
        (
            "log1p",
            "ref_to_cand",
            [2.0 * np.log1p(1.0) * np.log1p(5.0), 0.5 * np.log1p(4.0) * np.log1p(7.0)],
        ),
        (
            "log1p",
            "cand_to_ref",
            [2.0 * np.log1p(3.0) * np.log1p(2.0), 0.5 * np.log1p(6.0) * np.log1p(8.0)],
        ),
        (
            "log1p",
            "both",
            [
                2.0 * (np.log1p(1.0) * np.log1p(5.0) + np.log1p(3.0) * np.log1p(2.0)),
                0.5 * (np.log1p(4.0) * np.log1p(7.0) + np.log1p(6.0) * np.log1p(8.0)),
            ],
        ),
    ],
)
def test_score_pairs_matches_formula_for_transform_and_direction(cci_inputs, transform, direction, expected):
    expression, lr_pairs = cci_inputs

    result = score_pairs(expression, [("ref", "bucket")], lr_pairs, transform, direction)

    np.testing.assert_allclose(result["contribution"], expected)
    assert list(result["pathway"]) == ["p1", "p2"]
    assert list(result["weight"]) == [2.0, 0.5]


def test_score_pairs_are_limited_to_explicit_pairs_and_align_by_cell_id(cci_inputs):
    expression, lr_pairs = cci_inputs
    ordered = score_pairs(expression, [("ref", "bucket")], lr_pairs, "none", "both")
    shuffled = expression.loc[["unused", "bucket", "ref"]]

    result = score_pairs(shuffled, [("ref", "bucket")], lr_pairs, "none", "both")

    assert set(result["reference_cell"]) == {"ref"}
    assert set(result["bucket_cell"]) == {"bucket"}
    assert len(result) == 2
    np.testing.assert_allclose(result["contribution"], ordered["contribution"])


def test_score_pairs_stringifies_integer_cell_ids_before_lookup():
    expression = pd.DataFrame(
        {"lig": [1.0, 3.0], "rec": [2.0, 5.0]},
        index=[0, 1],
    )
    string_expression = expression.copy()
    string_expression.index = ["0", "1"]
    lr_pairs = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [2.0]})

    integer_result = score_pairs(expression, [("0", "1")], lr_pairs, "none", "both")
    string_result = score_pairs(string_expression, [("0", "1")], lr_pairs, "none", "both")

    np.testing.assert_allclose(integer_result["contribution"], string_result["contribution"])


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_score_pairs_rejects_duplicate_names_after_string_conversion(axis):
    if axis == "index":
        expression = pd.DataFrame({"lig": [1.0, 2.0], "rec": [3.0, 4.0]}, index=[0, "0"])
        cell_pairs = [("0", "0")]
    else:
        expression = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["ref", "bucket"], columns=[1, "1"])
        cell_pairs = [("ref", "bucket")]
    lr_pairs = pd.DataFrame({"ligand": ["1"], "receptor": ["1"], "weight": [1.0]})

    with pytest.raises(ValueError, match="duplicate"):
        score_pairs(expression, cell_pairs, lr_pairs, "none", "both")


@pytest.mark.parametrize("column", ["reference_cell", "bucket_cell", "contribution"])
def test_score_pairs_rejects_reserved_output_columns(cci_inputs, column):
    expression, lr_pairs = cci_inputs
    lr_pairs = lr_pairs.iloc[[0]].copy()
    lr_pairs[column] = "existing"

    with pytest.raises(ValueError, match=rf"reserved.*{column}"):
        score_pairs(expression, [("ref", "bucket")], lr_pairs, "none", "both")


def test_select_lr_given_returns_copy_with_metadata_without_mutating_input(cci_inputs):
    _, lr_pairs = cci_inputs
    original = lr_pairs.copy(deep=True)

    selected = select_lr(lr_pairs, "given")

    pd.testing.assert_frame_equal(lr_pairs, original)
    assert selected is not lr_pairs
    assert selected["selection"].tolist() == ["given", "given"]
    assert selected["effect"].isna().all()
    assert selected["pathway"].tolist() == ["p1", "p2"]


@pytest.mark.parametrize("column", ["selection", "effect"])
def test_select_lr_rejects_reserved_output_columns(cci_inputs, column):
    _, lr_pairs = cci_inputs
    lr_pairs[column] = "existing"

    with pytest.raises(ValueError, match=rf"reserved.*{column}"):
        select_lr(lr_pairs, "given")


def test_select_lr_train_topk_is_reserved_for_m3(cci_inputs):
    _, lr_pairs = cci_inputs
    with pytest.raises(ValueError, match="train_topk"):
        select_lr(lr_pairs, "train_topk")


def test_numeric_expression_fast_path_matches_apply_to_numeric():
    expression = pd.DataFrame(
        {
            "float_gene": [1.5, 2.25, 3.0],
            "int_gene": np.array([1, 2, 3], dtype=np.int32),
            "bool_gene": [True, False, True],
        },
        index=["c1", "c2", "c3"],
    )

    result = _numeric_expression(expression)
    legacy = expression.apply(pd.to_numeric, errors="coerce")
    legacy.index = [str(value) for value in legacy.index]
    legacy.columns = [str(value) for value in legacy.columns]

    assert result.equals(legacy)
    assert list(result.dtypes) == list(legacy.dtypes)


def _topk_example():
    expression = pd.DataFrame(
        {
            "lig1": [4.0, 4.0, 0.5, 0.5, 3.0, 1.0, 5.0, 0.4, 0.4, 0.5, 2.0, 1.0],
            "rec1": [0.5, 0.5, 0.4, 6.0, 1.0, 1.0, 0.5, 0.4, 0.4, 7.0, 1.0, 1.0],
            "lig2": [1.0, 1.0, 3.0, 0.2, 2.0, 0.5, 1.0, 4.0, 0.2, 0.2, 2.0, 0.5],
            "rec2": [0.2, 0.2, 0.2, 1.0, 1.0, 0.5, 0.2, 0.2, 5.0, 1.0, 1.0, 0.5],
        },
        index=["s1_A1", "s1_A1b", "s1_A2", "s1_R1", "s1_Ax", "s1_Z", "s2_A1", "s2_A2", "s2_R1", "s2_R2", "s2_Ax", "s2_Z"],
    )
    cells = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2", "s2", "s2"],
            "label": ["A1", "A1", "A2", "R1", "Ax", "Z", "A1", "A2", "R1", "R2", "Ax", "Z"],
        },
        index=expression.index,
    )
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["lig1", "lig2"],
            "receptor": ["rec1", "rec2"],
            "weight": [2.0, 0.5],
            "pathway": ["p1", "p2"],
        }
    )
    contacts = pd.DataFrame(
        {
            "identity_a": ["A1", "A2", "A1", "A1", "A2", "R1", "Ax"],
            "identity_b": ["R1", "R2", "A2", "R2", "R1", "R2", "R1"],
            "contact": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    return expression, cells, lr_pairs, contacts


def _explicit_topk_pairs(cells, contacts, group_names, drop_ambiguity=True, require_contact_record=True):
    contact_map = {}
    for _, row in contacts.iterrows():
        key = tuple(sorted((str(row["identity_a"]), str(row["identity_b"]))))
        contact_map[key] = float(row["contact"])
    pairs = []
    for sample_id, sample_cells in cells.groupby("sample_id", sort=False):
        ids = list(sample_cells.index.astype(str))
        labels = sample_cells["label"].astype(str)
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                left_label = str(labels.loc[left])
                right_label = str(labels.loc[right])
                if drop_ambiguity and (left_label in group_names or right_label in group_names):
                    continue
                key = tuple(sorted((left_label, right_label)))
                if key not in contact_map:
                    if require_contact_record:
                        continue
                    pairs.append((str(sample_id), left, right, None))
                    continue
                pairs.append((str(sample_id), left, right, contact_map[key]))
    return pairs


def test_train_topk_matches_pairwise_score_pairs_and_exclusion_rules():
    expression, cells, lr_pairs, contacts = _topk_example()
    group_names = ["Ax"]
    pairs = _explicit_topk_pairs(cells, contacts, group_names)
    samples = {sample for sample, _left, _right, _contact in pairs}
    assert samples == {"s1", "s2"}
    assert all(left[:2] == right[:2] for _sample, left, right, _contact in pairs)

    without_ambiguity_rule = _explicit_topk_pairs(cells, contacts, group_names, drop_ambiguity=False)
    without_record_rule = _explicit_topk_pairs(cells, contacts, group_names, require_contact_record=False)
    assert len(without_ambiguity_rule) > len(pairs)
    assert len(without_record_rule) > len(pairs)

    contact_pairs = [(left, right) for _sample, left, right, contact in pairs if contact > 0]
    noncontact_pairs = [(left, right) for _sample, left, right, contact in pairs if contact == 0]
    contact_scores = score_pairs(expression, contact_pairs, lr_pairs, "none", "both")
    noncontact_scores = score_pairs(expression, noncontact_pairs, lr_pairs, "none", "both")
    key = contact_scores["ligand"].astype(str) + "|" + contact_scores["receptor"].astype(str)
    expected_effect = (
        contact_scores.groupby(key, sort=False)["contribution"].mean()
        - noncontact_scores.groupby(
            noncontact_scores["ligand"].astype(str) + "|" + noncontact_scores["receptor"].astype(str),
            sort=False,
        )["contribution"].mean()
    )

    selected = train_topk(expression, cells, lr_pairs, contacts, group_names, "none", top_k=10)

    assert selected.n_contact_pairs == len(contact_pairs)
    assert selected.n_noncontact_pairs == len(noncontact_pairs)
    for _, row in selected.table.iterrows():
        key = f"{row['ligand']}|{row['receptor']}"
        np.testing.assert_allclose(row["effect"], expected_effect.loc[key])


def test_train_topk_keeps_positive_effects_stable_order_and_top_k():
    expression = pd.DataFrame(
        {
            "lig_pos": [10.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            "rec_pos": [0.0, 10.0, 0.0, 10.0, 0.0, 0.0],
            "lig_neg": [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            "rec_neg": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
            "lig_tie1": [4.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            "rec_tie1": [0.0, 4.0, 0.0, 4.0, 0.0, 0.0],
            "lig_tie2": [4.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            "rec_tie2": [0.0, 4.0, 0.0, 4.0, 0.0, 0.0],
        },
        index=["s1_A", "s1_B", "s2_A", "s2_B", "s3_A", "s3_C"],
    )
    cells = pd.DataFrame(
        {"sample_id": ["s1", "s1", "s2", "s2", "s3", "s3"], "label": ["A", "B", "A", "B", "A", "C"]},
        index=expression.index,
    )
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["lig_neg", "lig_tie1", "lig_tie2", "lig_pos"],
            "receptor": ["rec_neg", "rec_tie1", "rec_tie2", "rec_pos"],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    contacts = pd.DataFrame(
        {"identity_a": ["A", "A"], "identity_b": ["B", "C"], "contact": [1.0, 0.0]}
    )

    selected = train_topk(expression, cells, lr_pairs, contacts, [], "none", top_k=2)

    assert list(selected.table["ligand"]) == ["lig_pos", "lig_tie1"]
    assert (selected.table["effect"] > 0).all()
    assert selected.table["selection"].tolist() == ["train_topk", "train_topk"]


@pytest.mark.parametrize(
    ("contact_values", "empty_count_name"),
    [([1.0], "n_noncontact_pairs"), ([0.0], "n_contact_pairs")],
)
def test_train_topk_empty_class_returns_empty_table_with_same_columns(contact_values, empty_count_name):
    expression, cells, lr_pairs, _contacts = _topk_example()
    contacts = pd.DataFrame(
        {"identity_a": ["A1"], "identity_b": ["R1"], "contact": contact_values}
    )
    nonempty_contacts = pd.DataFrame(
        {
            "identity_a": ["A1", "A1"],
            "identity_b": ["R1", "A2"],
            "contact": [1.0, 0.0],
        }
    )
    nonempty = train_topk(expression, cells, lr_pairs, nonempty_contacts, ["Ax"], "none", top_k=10)
    selected = train_topk(expression, cells, lr_pairs, contacts, ["Ax"], "none", top_k=10)

    assert selected.table.empty
    assert list(selected.table.columns) == list(nonempty.table.columns)
    assert getattr(selected, empty_count_name) == 0


@pytest.mark.parametrize("column", ["selection", "effect"])
def test_train_topk_rejects_reserved_output_columns(column):
    expression, cells, lr_pairs, contacts = _topk_example()
    lr_pairs = lr_pairs.copy()
    lr_pairs[column] = "existing"

    with pytest.raises(ValueError, match="reserved"):
        train_topk(expression, cells, lr_pairs, contacts, ["Ax"], "none", 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": -1},
        {"top_k": True},
        {"top_k": 1.5},
        {"transform": "sqrt"},
    ],
)
def test_train_topk_rejects_invalid_top_k_and_transform(kwargs):
    expression, cells, lr_pairs, contacts = _topk_example()
    call = {"expression": expression, "cells": cells, "lr_pairs": lr_pairs, "contacts": contacts, "group_names": ["Ax"], "transform": "none", "top_k": 2}
    call.update(kwargs)
    with pytest.raises(ValueError):
        train_topk(**call)


def test_train_topk_rejects_invalid_cells_and_genes():
    expression, cells, lr_pairs, contacts = _topk_example()

    with pytest.raises(ValueError):
        train_topk(expression, cells.drop(columns=["sample_id"]), lr_pairs, contacts, ["Ax"], "none", 2)
    with pytest.raises(ValueError):
        train_topk(expression, cells.drop(columns=["label"]), lr_pairs, contacts, ["Ax"], "none", 2)
    duplicated = cells.copy()
    duplicated.index = ["s1_A1"] * len(duplicated)
    with pytest.raises(ValueError):
        train_topk(expression, duplicated, lr_pairs, contacts, ["Ax"], "none", 2)
    extra = cells.copy()
    extra.loc["missing_cell"] = {"sample_id": "s1", "label": "A1"}
    with pytest.raises(ValueError):
        train_topk(expression, extra, lr_pairs, contacts, ["Ax"], "none", 2)
    bad_lr = lr_pairs.copy()
    bad_lr.loc[0, "ligand"] = "absent_gene"
    with pytest.raises(ValueError):
        train_topk(expression, cells, bad_lr, contacts, ["Ax"], "none", 2)

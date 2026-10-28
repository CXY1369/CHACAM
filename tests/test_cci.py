import numpy as np
import pandas as pd
import pytest

from chacam.cci import _numeric_expression, score_pairs, select_lr


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
    with pytest.raises(NotImplementedError, match="implemented in M3"):
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

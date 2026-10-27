import numpy as np
import pandas as pd
import pytest

from chacam.cci import score_pairs, select_lr


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
    shuffled = expression.loc[["unused", "bucket", "ref"]]

    result = score_pairs(shuffled, [("ref", "bucket")], lr_pairs, "none", "both")

    assert set(result["reference_cell"]) == {"ref"}
    assert set(result["bucket_cell"]) == {"bucket"}
    assert len(result) == 2


def test_select_lr_given_returns_copy_with_metadata_without_mutating_input(cci_inputs):
    _, lr_pairs = cci_inputs
    original = lr_pairs.copy(deep=True)

    selected = select_lr(lr_pairs, "given")

    pd.testing.assert_frame_equal(lr_pairs, original)
    assert selected is not lr_pairs
    assert selected["selection"].tolist() == ["given", "given"]
    assert selected["effect"].isna().all()
    assert selected["pathway"].tolist() == ["p1", "p2"]


def test_select_lr_train_topk_is_reserved_for_m3(cci_inputs):
    _, lr_pairs = cci_inputs
    with pytest.raises(NotImplementedError, match="M3 实现"):
        select_lr(lr_pairs, "train_topk")

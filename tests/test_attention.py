import numpy as np
import pandas as pd
import pytest

from chacam.attention import attention, l1_normalize, softmax_normalize


S_FIG = pd.DataFrame(
    [
        [0.09, 0.46, 0.05, 0.5],
        [0.26, 0.072, 0.48, 0.75],
        [0.00059, 0.058, 0.76, 0.49],
        [0.6, 0, 0.32, 0.22],
    ],
    index=["c1", "c2", "c3", "c4"],
    columns=["P2", "EMS", "ABp", "ABa"],
)


def test_l1_normalization_matches_fig2_values_and_preserves_zero():
    p_col, p_row, product = l1_normalize(S_FIG)
    p_col_fig = np.array(
        [[0.094, 0.78, 0.031, 0.26], [0.27, 0.12, 0.3, 0.38], [0.00063, 0.098, 0.47, 0.25], [0.63, 0, 0.2, 0.11]]
    )
    p_row_fig = np.array(
        [[0.081, 0.42, 0.045, 0.45], [0.17, 0.046, 0.31, 0.48], [0.00046, 0.044, 0.58, 0.37], [0.53, 0, 0.28, 0.19]]
    )
    l_fig = np.array(
        [[0.0077, 0.32, 0.0014, 0.12], [0.045, 0.0056, 0.091, 0.18], [2.8e-07, 0.0044, 0.28, 0.092], [0.33, 0, 0.055, 0.022]]
    )
    for actual, expected in ((p_col.to_numpy(), p_col_fig), (p_row.to_numpy(), p_row_fig), (product.to_numpy(), l_fig)):
        absolute_error = np.abs(actual - expected)
        assert np.all(absolute_error <= 0.015)
        nonzero = expected != 0
        assert np.all(absolute_error[nonzero] / np.abs(expected[nonzero]) <= 0.05)
        assert np.all(actual[~nonzero] == 0)


def test_attention_is_invariant_to_target_gene_column_order():
    target = pd.DataFrame([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], index=["c1", "c2"], columns=["g1", "g2", "g3"])
    reference = pd.DataFrame([[2.0, 1.0, 3.0], [1.0, 3.0, 2.0]], index=["r1", "r2"], columns=["g1", "g2", "g3"])
    ordered = attention(target, reference, normalization="l1")
    shuffled = attention(target[["g3", "g1", "g2"]], reference, normalization="l1")
    pd.testing.assert_frame_equal(ordered.S, shuffled.S)
    pd.testing.assert_frame_equal(ordered.P_col, shuffled.P_col)
    pd.testing.assert_frame_equal(ordered.P_row, shuffled.P_row)
    pd.testing.assert_frame_equal(ordered.L, shuffled.L)


def test_attention_used_genes_follow_reference_column_order():
    target = pd.DataFrame([[1.0, 2.0, 3.0]], index=["c1"], columns=["g1", "g2", "g3"])
    reference = pd.DataFrame([[2.0, 1.0, 3.0]], index=["r1"], columns=["g3", "g1", "g2"])

    result = attention(target, reference, normalization="l1")

    assert result.used_genes == ["g3", "g1", "g2"]


def test_l1_rejects_negative_similarity_with_softmax_hint():
    with pytest.raises(ValueError, match="softmax"):
        l1_normalize(pd.DataFrame([[1.0, -0.1]], columns=["a", "b"]))


def test_softmax_is_stable_and_matches_independent_calculation():
    s = pd.DataFrame([[1000.0, 999.0], [998.0, 997.0]], index=["c1", "c2"], columns=["a", "b"])
    p_col, p_row, product = softmax_normalize(s, temperature=2.0)
    scaled = s.to_numpy() / 2.0
    expected_col = np.exp(scaled - scaled.max(axis=0, keepdims=True))
    expected_col /= expected_col.sum(axis=0, keepdims=True)
    expected_row = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    expected_row /= expected_row.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p_col, expected_col)
    np.testing.assert_allclose(p_row, expected_row)
    np.testing.assert_allclose(product, expected_col * expected_row)
    assert (product.to_numpy() > 0).all()


def test_larger_softmax_temperature_makes_rows_flatter_and_nonpositive_fails():
    s = pd.DataFrame([[1.0, 4.0]], columns=["a", "b"])
    low = softmax_normalize(s, temperature=0.5)[1].iloc[0, 1]
    high = softmax_normalize(s, temperature=5.0)[1].iloc[0, 1]
    assert high < low
    with pytest.raises(ValueError, match="positive"):
        softmax_normalize(s, temperature=0)


def test_attention_aligns_genes_by_name_and_records_zero_vectors():
    target = pd.DataFrame([[0, 0], [2, 0]], index=["zero", "nonzero"], columns=["g2", "g1"])
    reference = pd.DataFrame([[1, 0], [0, 0]], index=["R1", "zero_ref"], columns=["g1", "g2"])
    result = attention(target, reference, normalization="l1")
    assert result.S.loc["zero", "R1"] == 0
    assert result.S.loc["nonzero", "zero_ref"] == 0
    assert result.zero_target_cells == ["zero"]
    assert result.zero_reference_identities == ["zero_ref"]
    assert not result.L.isna().any().any()
    assert (result.L.loc["zero"] == 0).all()


def test_l1_zero_row_and_zero_column_produce_zeros_not_nan():
    s = pd.DataFrame([[1.0, 0.0], [0.0, 0.0]], columns=["a", "b"])
    p_col, p_row, product = l1_normalize(s)
    assert not any(frame.isna().any().any() for frame in [p_col, p_row, product])
    assert (p_col["b"] == 0).all()
    assert (p_row.loc[1] == 0).all()

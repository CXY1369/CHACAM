import numpy as np
import pandas as pd

from chacam.reference import binary_marker, training_mean
from chacam.schema import AmbiguityGroup


def test_training_mean_groups_observed_labels_and_records_gene_alignment():
    expression = pd.DataFrame(
        [[1, 2, 10], [3, 4, 20], [5, 6, 30]],
        index=["c1", "c2", "c3"],
        columns=["g2", "g1", "extra"],
    )
    labels = pd.Series(["Ax", "Ax", "R1"], index=expression.index)
    profiles = training_mean(expression, labels, ["g1", "missing", "g2"], expected_labels=["Ax", "R1"])

    assert list(profiles.index) == ["Ax", "R1"]
    assert list(profiles.columns) == ["g1", "g2"]
    np.testing.assert_allclose(profiles.loc["Ax"], [3.0, 2.0])
    np.testing.assert_allclose(profiles.loc["R1"], [6.0, 5.0])
    assert profiles.attrs["used_feature_genes"] == ["g1", "g2"]
    assert profiles.attrs["missing_feature_genes"] == ["missing"]
    assert profiles.attrs["missing_labels"] == []


def test_training_mean_records_expected_labels_missing_from_training_fold():
    expression = pd.DataFrame([[1, 2]], index=["c1"], columns=["g1", "g2"])
    labels = pd.Series(["A"], index=["c1"])
    profiles = training_mean(expression, labels, ["g1", "g2"], expected_labels=["A", "B"])
    assert list(profiles.index) == ["A"]
    assert profiles.attrs["missing_labels"] == ["B"]


def test_training_mean_rejects_observed_labels_outside_expected_labels():
    expression = pd.DataFrame([[1, 2], [3, 4]], index=["c1", "c2"], columns=["g1", "g2"])
    labels = pd.Series(["A", "B"], index=["c1", "c2"])
    try:
        training_mean(expression, labels, ["g1", "g2"], expected_labels=["A"])
    except ValueError as error:
        assert "B" in str(error)
    else:
        raise AssertionError("observed label outside expected_labels was accepted")


def test_series_labels_with_unmatched_ids_are_rejected_instead_of_positionally_paired():
    expression = pd.DataFrame([[1, 2], [3, 4]], index=["c1", "c2"], columns=["g1", "g2"])
    labels = pd.Series(["A", "B"], index=["other1", "other2"])
    try:
        training_mean(expression, labels, ["g1", "g2"], expected_labels=["A", "B"])
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("unmatched Series IDs were accepted")


def test_series_labels_are_aligned_by_id_when_order_is_shuffled():
    expression = pd.DataFrame([[1, 2], [10, 20]], index=["c1", "c2"], columns=["g1", "g2"])
    ordered = pd.Series(["A", "B"], index=["c1", "c2"])
    shuffled = pd.Series(["B", "A"], index=["c2", "c1"])
    first = training_mean(expression, ordered, ["g1", "g2"], expected_labels=["A", "B"])
    second = training_mean(expression, shuffled, ["g1", "g2"], expected_labels=["A", "B"])
    pd.testing.assert_frame_equal(first, second)


def test_binary_marker_uses_ambiguity_candidate_union_and_records_zero_labels():
    features = pd.DataFrame(
        {"gene": ["g1", "g2", "g3", "missing"], "label": ["A1", "A2", "R1", "R1"]}
    )
    labels = pd.Series(["Ax", "R1", "NoMarker"], index=["c1", "c2", "c3"])
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1", "R2"))
    profiles = binary_marker(features, labels, ["g1", "g2", "g3"], [group])

    assert list(profiles.index) == ["Ax", "R1", "NoMarker"]
    assert list(profiles.columns) == ["g1", "g2", "g3"]
    np.testing.assert_array_equal(profiles.loc["Ax"], [1, 1, 0])
    np.testing.assert_array_equal(profiles.loc["R1"], [0, 0, 1])
    np.testing.assert_array_equal(profiles.loc["NoMarker"], [0, 0, 0])
    assert profiles.attrs["missing_feature_genes"] == []
    assert profiles.attrs["zero_labels"] == ["NoMarker"]


def test_binary_marker_preserves_requested_gene_order_and_reports_requested_missing_genes():
    features = pd.DataFrame(
        {"gene": ["g2", "g1", "g3"], "label": ["A2", "A1", "R1"]}
    )
    labels = pd.Series(["Ax", "R1"], index=["c1", "c2"])
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1", "R2"))

    profiles = binary_marker(features, labels, ["g3", "missing", "g1", "g2"], [group])

    assert list(profiles.columns) == ["g3", "g1", "g2"]
    np.testing.assert_array_equal(profiles.loc["Ax"], [0, 1, 1])
    assert profiles.attrs["missing_feature_genes"] == ["missing"]


def test_binary_marker_requires_label_column():
    features = pd.DataFrame({"gene": ["g1"]})
    labels = pd.Series(["R1"])
    try:
        binary_marker(features, labels, ["g1"], [])
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("binary_marker accepted features without label")

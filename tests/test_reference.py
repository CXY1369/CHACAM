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
    profiles = training_mean(expression, labels, ["g1", "missing", "g2"])

    assert list(profiles.index) == ["Ax", "R1"]
    assert list(profiles.columns) == ["g1", "g2"]
    np.testing.assert_allclose(profiles.loc["Ax"], [3.0, 2.0])
    np.testing.assert_allclose(profiles.loc["R1"], [6.0, 5.0])
    assert profiles.attrs["used_feature_genes"] == ["g1", "g2"]
    assert profiles.attrs["missing_feature_genes"] == ["missing"]
    assert profiles.attrs["missing_labels"] == []


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
    assert profiles.attrs["missing_feature_genes"] == ["missing"]
    assert profiles.attrs["zero_labels"] == ["NoMarker"]


def test_binary_marker_requires_label_column():
    features = pd.DataFrame({"gene": ["g1"]})
    labels = pd.Series(["R1"])
    try:
        binary_marker(features, labels, ["g1"], [])
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("binary_marker accepted features without label")

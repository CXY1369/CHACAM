import numpy as np
import pandas as pd
import pytest

from chacam.schema import AmbiguityGroup
from chacam.triangulation import triangulate


GROUP = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1", "R2"))


def _contacts():
    return pd.DataFrame(
        {
            "identity_a": ["R1", "R1", "R2", "R2"],
            "identity_b": ["A1", "A2", "A1", "A2"],
            "contact": [1.0, 0.0, 0.0, 1.0],
        }
    )


def _lr_table():
    return pd.DataFrame(
        {
            "ligand": ["lig_R1", "lig_R2"],
            "receptor": ["rec_R1", "rec_R2"],
            "weight": [1.0, 1.0],
            "pathway": ["p1", "p2"],
        }
    )


def _resolvable_inputs():
    expression = pd.DataFrame(
        {
            "lig_R1": [1.0, 0.0, 0.0, 0.0],
            "rec_R1": [0.0, 0.0, 10.0, 1.0],
            "lig_R2": [0.0, 1.0, 0.0, 0.0],
            "rec_R2": [0.0, 0.0, 1.0, 10.0],
        },
        index=["r1", "r2", "u", "v"],
    )
    labels = pd.Series(["R1", "R2", "Ax", "Ax"], index=expression.index)
    return labels, expression


def test_triangulate_resolves_with_unanimous_reference_votes():
    labels, expression = _resolvable_inputs()

    result = triangulate(labels, expression, GROUP, _contacts(), _lr_table(), "none", "both", sample_id="heldout")

    assert result.status.to_dict() == {"u": "resolved", "v": "resolved"}
    assert result.labels.to_dict() == {"u": "A1", "v": "A2"}
    assert result.reasons.to_dict() == {"u": "", "v": ""}
    assert set(result.evidence["reference_identity"]) == {"R1", "R2"}


@pytest.mark.parametrize("bucket_count", [1, 3])
def test_triangulate_rejects_bucket_sizes_other_than_two(bucket_count):
    expression = pd.DataFrame({"lig": [1.0] * bucket_count, "rec": [1.0] * bucket_count}, index=[f"u{i}" for i in range(bucket_count)])
    labels = pd.Series(["Ax"] * bucket_count, index=expression.index)
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1",))
    lr_table = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [1.0]})
    contacts = pd.DataFrame({"identity_a": ["R1", "R1"], "identity_b": ["A1", "A2"], "contact": [1.0, 0.0]})

    result = triangulate(labels, expression, group, contacts, lr_table, "none", "both", sample_id="s")

    assert result.status.tolist() == ["unresolved"] * bucket_count
    assert result.reasons.tolist() == ["unresolved_bucket_size"] * bucket_count
    assert result.labels.tolist() == ["Ax"] * bucket_count
    assert result.evidence.empty


def test_triangulate_reports_no_lr_before_reference_votes():
    labels, expression = _resolvable_inputs()
    empty_lr = pd.DataFrame(columns=["ligand", "receptor", "weight", "pathway"])

    result = triangulate(labels, expression, GROUP, _contacts(), empty_lr, "none", "both", sample_id="s")

    assert set(result.reasons) == {"unresolved_no_lr"}
    assert result.evidence.empty


def test_triangulate_reports_no_reference_when_all_references_are_missing():
    _, expression = _resolvable_inputs()
    labels = pd.Series(["Ax", "Ax"], index=["u", "v"])

    result = triangulate(labels, expression, GROUP, _contacts(), _lr_table(), "none", "both", sample_id="s")

    assert set(result.reasons) == {"unresolved_no_reference"}
    assert set(result.labels) == {"Ax"}


def test_triangulate_uses_one_available_reference_without_conflict():
    _, expression = _resolvable_inputs()
    labels = pd.Series(["R1", "Ax", "Ax"], index=["r1", "u", "v"])

    result = triangulate(labels, expression, GROUP, _contacts(), _lr_table(), "none", "both", sample_id="s")

    assert result.labels.to_dict() == {"u": "A1", "v": "A2"}
    assert set(result.status) == {"resolved"}


def test_triangulate_reports_tie_when_bucket_expression_is_identical():
    expression = pd.DataFrame(
        {"lig": [1.0, 0.0, 0.0], "rec": [0.0, 5.0, 5.0]},
        index=["r1", "u", "v"],
    )
    labels = pd.Series(["R1", "Ax", "Ax"], index=expression.index)
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1",))
    contacts = pd.DataFrame({"identity_a": ["R1", "R1"], "identity_b": ["A1", "A2"], "contact": [1.0, 0.0]})
    lr_table = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [1.0]})

    result = triangulate(labels, expression, group, contacts, lr_table, "none", "both", sample_id="s")

    assert set(result.reasons) == {"unresolved_tie"}
    assert set(result.labels) == {"Ax"}


def test_triangulate_reports_reference_conflict():
    labels, expression = _resolvable_inputs()
    expression.loc["u", "rec_R2"] = 10.0
    expression.loc["v", "rec_R2"] = 1.0

    result = triangulate(labels, expression, GROUP, _contacts(), _lr_table(), "none", "both", sample_id="s")

    assert set(result.reasons) == {"unresolved_reference_conflict"}
    assert set(result.labels) == {"Ax"}


def test_triangulate_evidence_sums_to_reference_mean_score():
    expression = pd.DataFrame(
        {"lig": [2.0, 2.0, 0.0, 0.0], "rec": [0.0, 0.0, 5.0, 1.0]},
        index=["r1a", "r1b", "u", "v"],
    )
    labels = pd.Series(["R1", "R1", "Ax", "Ax"], index=expression.index)
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1",))
    contacts = pd.DataFrame({"identity_a": ["R1", "R1"], "identity_b": ["A1", "A2"], "contact": [1.0, 0.0]})
    lr_table = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [1.0]})

    result = triangulate(labels, expression, group, contacts, lr_table, "none", "ref_to_cand", sample_id="s")

    sums = result.evidence.groupby("bucket_cell")["contribution"].sum().to_dict()
    assert sums == {"u": 10.0, "v": 2.0}


def test_triangulate_direction_changes_evidence_and_both_is_the_sum():
    expression = pd.DataFrame(
        {"lig": [2.0, 5.0, 5.0], "rec": [3.0, 7.0, 7.0]},
        index=["r1", "u", "v"],
    )
    labels = pd.Series(["R1", "Ax", "Ax"], index=expression.index)
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1",))
    contacts = pd.DataFrame({"identity_a": ["R1", "R1"], "identity_b": ["A1", "A2"], "contact": [1.0, 0.0]})
    lr_table = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [1.0]})

    forward = triangulate(labels, expression, group, contacts, lr_table, "none", "ref_to_cand", sample_id="s")
    reverse = triangulate(labels, expression, group, contacts, lr_table, "none", "cand_to_ref", sample_id="s")
    both = triangulate(labels, expression, group, contacts, lr_table, "none", "both", sample_id="s")

    assert forward.evidence.iloc[0]["contribution"] == 14.0
    assert reverse.evidence.iloc[0]["contribution"] == 15.0
    assert both.evidence.iloc[0]["contribution"] == 29.0


def test_triangulate_finds_contact_when_contact_columns_are_reversed_and_keeps_sample_id_and_pathway():
    labels, expression = _resolvable_inputs()
    contacts = _contacts().rename(columns={"identity_a": "identity_b", "identity_b": "identity_a"})
    lr_table = _lr_table()

    result = triangulate(labels, expression, GROUP, contacts, lr_table, "none", "both", sample_id="explicit-sample")

    assert result.labels.to_dict() == {"u": "A1", "v": "A2"}
    assert set(result.evidence["sample_id"]) == {"explicit-sample"}
    assert result.evidence["pathway"].tolist() == ["p1", "p2", "p1", "p2", "p1", "p2", "p1", "p2"]


@pytest.mark.parametrize("column", ["sample_id", "group", "reference_identity"])
def test_triangulate_rejects_reserved_evidence_columns_with_clear_error(column):
    labels, expression = _resolvable_inputs()
    lr_table = _lr_table()
    lr_table[column] = "existing"

    with pytest.raises(ValueError) as error:
        triangulate(labels, expression, GROUP, _contacts(), lr_table, "none", "both", sample_id="s")

    message = str(error.value)
    assert "reserved" in message
    assert column in message
    assert "cannot insert" not in message


def test_triangulate_checks_reserved_evidence_columns_before_bucket_size_return():
    expression = pd.DataFrame({"lig": [1.0], "rec": [1.0]}, index=["u"])
    labels = pd.Series(["Ax"], index=["u"])
    group = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1",))
    contacts = pd.DataFrame({"identity_a": ["R1", "R1"], "identity_b": ["A1", "A2"], "contact": [1.0, 0.0]})
    lr_table = pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"], "weight": [1.0], "group": ["existing"]})

    with pytest.raises(ValueError, match="reserved.*group"):
        triangulate(labels, expression, group, contacts, lr_table, "none", "both", sample_id="s")

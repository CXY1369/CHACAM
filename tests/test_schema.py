from pathlib import Path
import re

import pandas as pd
import pytest
import yaml

from chacam.schema import load_config, load_inputs, load_truth


IDENTITIES = ["A1", "A2", "R1", "R2"]


def _write_tables(tmp_path: Path, *, expression=None, cells=None, features=None,
                  lr_pairs=None, contacts=None):
    expression = expression if expression is not None else pd.DataFrame(
        {"cell_id": ["c1", "c2", "c3", "c4"], "g1": [5, 1, 2, 3], "g2": [1, 5, 3, 2]}
    )
    cells = cells if cells is not None else pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "sample_id": ["s1", "s1", "s2", "s2"],
            "label": ["Ax", "R1", "Ax", "R2"],
        }
    )
    features = features if features is not None else pd.DataFrame(
        {"gene": ["g1", "g2"], "label": ["A1", "R1"]}
    )
    lr_pairs = lr_pairs if lr_pairs is not None else pd.DataFrame(
        {"ligand": ["g1"], "receptor": ["g2"]}
    )
    contacts = contacts if contacts is not None else pd.DataFrame(
        {
            "identity_a": ["R1", "R1", "R2", "R2", "A1", "A1"],
            "identity_b": ["A1", "A2", "A1", "A2", "A2", "R2"],
            "contact": [1, 0, 0, 1, 0, 0],
        }
    )
    expression.to_csv(tmp_path / "expression.csv", index=False)
    cells.to_csv(tmp_path / "cells.csv", index=False)
    features.to_csv(tmp_path / "features.csv", index=False)
    lr_pairs.to_csv(tmp_path / "lr_pairs.csv", index=False)
    contacts.to_csv(tmp_path / "contacts.csv", index=False)


def _config_dict(tmp_path: Path, **overrides):
    config = {
        "provenance": "framework_baseline",
        "seed": 7,
        "inputs": {
            "expression": "expression.csv",
            "cells": "cells.csv",
            "features": "features.csv",
            "lr_pairs": "lr_pairs.csv",
            "contacts": "contacts.csv",
            "expression_scale": "counts",
        },
        "reference": {"mode": "training_mean"},
        "attention": {"normalization": "l1", "temperature": 1.0},
        "assignment": {"mode": "unique"},
        "cci": {
            "transform": "log1p",
            "direction": "both",
            "lr_selection": "given",
            "top_k": 2,
        },
        "ambiguity_groups": [
            {"name": "Ax", "candidates": ["A1", "A2"], "references": ["R1", "R2"]}
        ],
        "output_dir": "runs/test",
    }
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split(".", 1)
        if value is None:
            config[section].pop(key, None)
        else:
            config[section][key] = value
    return config


def _write_config(tmp_path: Path, **overrides):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config_dict(tmp_path, **overrides), sort_keys=False))
    return path


def _valid_inputs(tmp_path):
    _write_tables(tmp_path)
    return load_inputs(load_config(_write_config(tmp_path)))


def test_load_config_requires_attention_normalization(tmp_path):
    _write_tables(tmp_path)
    with pytest.raises(ValueError, match="normalization"):
        load_config(_write_config(tmp_path, **{"attention.normalization": None}))


def test_load_config_requires_cci_transform(tmp_path):
    _write_tables(tmp_path)
    with pytest.raises(ValueError, match="transform"):
        load_config(_write_config(tmp_path, **{"cci.transform": None}))


@pytest.mark.parametrize("temperature", [0, -1])
def test_load_config_rejects_nonpositive_softmax_temperature(tmp_path, temperature):
    _write_tables(tmp_path)
    with pytest.raises(ValueError, match="temperature"):
        load_config(_write_config(tmp_path, **{"attention.normalization": "softmax", "attention.temperature": temperature}))


def test_load_config_rejects_explicit_null_softmax_temperature(tmp_path):
    _write_tables(tmp_path)
    config = _config_dict(tmp_path, **{"attention.normalization": "softmax"})
    config["attention"]["temperature"] = None
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="temperature"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section_key", "value"),
    [
        ("reference.mode", "unknown"),
        ("attention.normalization", "unknown"),
        ("assignment.mode", "unknown"),
        ("cci.transform", "unknown"),
        ("cci.direction", "unknown"),
        ("cci.lr_selection", "unknown"),
        ("inputs.expression_scale", "unknown"),
    ],
)
def test_load_config_rejects_unknown_enum_values(tmp_path, section_key, value):
    _write_tables(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write_config(tmp_path, **{section_key: value}))


def test_load_inputs_accepts_valid_tables_and_keeps_truth_separate(tmp_path):
    _write_tables(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    inputs = load_inputs(config)

    assert list(inputs.expression.index) == ["c1", "c2", "c3", "c4"]
    assert list(inputs.cells.index) == ["c1", "c2", "c3", "c4"]
    assert not hasattr(inputs, "truth")
    assert inputs.used_feature_genes == ["g1", "g2"]
    assert inputs.missing_feature_genes == []


def test_duplicate_expression_cell_id_is_rejected(tmp_path):
    _write_tables(tmp_path, expression=pd.DataFrame(
        {"cell_id": ["c1", "c1", "c2", "c3"], "g1": [1, 2, 3, 4], "g2": [1, 2, 3, 4]}
    ))
    with pytest.raises(ValueError, match="c1"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_duplicate_expression_gene_header_is_rejected(tmp_path):
    expression = pd.DataFrame(
        [["c1", 1, 2, 3, 4], ["c2", 2, 1, 4, 3], ["c3", 3, 1, 5, 2], ["c4", 4, 1, 6, 1]],
        columns=["cell_id", "g1", "g1", "g2", "g2"],
    )
    _write_tables(tmp_path, expression=expression)
    with pytest.raises(ValueError, match=r"g1.*g2"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_expression_and_cells_ids_must_match(tmp_path):
    _write_tables(tmp_path, cells=pd.DataFrame(
        {"cell_id": ["c1", "c2", "c3", "other"], "sample_id": ["s1", "s1", "s2", "s2"],
         "label": ["Ax", "R1", "Ax", "R2"]}
    ))
    with pytest.raises(ValueError, match="other"):
        load_inputs(load_config(_write_config(tmp_path)))


@pytest.mark.parametrize(
    "expression",
    [
        pd.DataFrame({"cell_id": ["c1", "c2", "c3", "c4"], "g1": [-1, 1, 2, 3], "g2": [1, 2, 3, 4]}),
        pd.DataFrame({"cell_id": ["c1", "c2", "c3", "c4"], "g1": [1, float("nan"), 2, 3], "g2": [1, 2, 3, 4]}),
    ],
)
def test_expression_values_must_be_nonnegative_and_finite(tmp_path, expression):
    _write_tables(tmp_path, expression=expression)
    with pytest.raises(ValueError, match="g1"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_features_with_no_expression_intersection_are_rejected(tmp_path):
    _write_tables(tmp_path, features=pd.DataFrame({"gene": ["missing"], "label": ["A1"]}))
    with pytest.raises(ValueError, match="missing"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_partial_feature_overlap_is_recorded(tmp_path):
    _write_tables(tmp_path, features=pd.DataFrame({"gene": ["g1", "missing"], "label": ["A1", "R1"]}))
    inputs = load_inputs(load_config(_write_config(tmp_path)))
    assert inputs.used_feature_genes == ["g1"]
    assert inputs.missing_feature_genes == ["missing"]


def test_at_least_two_samples_are_required(tmp_path):
    _write_tables(tmp_path, cells=pd.DataFrame(
        {"cell_id": ["c1", "c2", "c3", "c4"], "sample_id": ["s1"] * 4,
         "label": ["Ax", "R1", "Ax", "R2"]}
    ))
    with pytest.raises(ValueError, match="sample_id"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_cells_cannot_use_reserved_unassigned_label(tmp_path):
    _write_tables(tmp_path, cells=pd.DataFrame(
        {"cell_id": ["c1", "c2", "c3", "c4"], "sample_id": ["s1", "s1", "s2", "s2"],
         "label": ["unassigned", "R1", "Ax", "R2"]}
    ))
    with pytest.raises(ValueError, match="unassigned"):
        load_inputs(load_config(_write_config(tmp_path)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "unassigned"),
        ("candidates", ["unassigned", "A2"]),
        ("references", ["unassigned", "R2"]),
    ],
)
def test_ambiguity_group_parts_cannot_use_reserved_unassigned_name(tmp_path, field, value):
    _write_tables(tmp_path)
    config = _config_dict(tmp_path)
    config["ambiguity_groups"][0][field] = value
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="unassigned"):
        load_config(tmp_path / "config.yaml")


@pytest.mark.parametrize("column", [
    "reference_cell", "bucket_cell", "contribution", "sample_id", "group",
    "reference_identity", "fold", "selection", "effect",
])
def test_lr_pairs_cannot_use_reserved_output_columns(tmp_path, column):
    _write_tables(tmp_path, lr_pairs=pd.DataFrame(
        {"ligand": ["g1"], "receptor": ["g2"], column: ["extra"]}
    ))

    with pytest.raises(ValueError, match=rf"reserved.*{re.escape(column)}"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_reserved_name_check_is_case_sensitive(tmp_path):
    _write_tables(tmp_path, cells=pd.DataFrame(
        {"cell_id": ["c1", "c2", "c3", "c4"], "sample_id": ["s1", "s1", "s2", "s2"],
         "label": ["Unassigned", "R1", "Ax", "R2"]}
    ))
    inputs = load_inputs(load_config(_write_config(tmp_path)))
    assert "Unassigned" in inputs.cells["label"].tolist()


def test_ambiguity_group_must_have_exactly_two_candidates(tmp_path):
    _write_tables(tmp_path)
    config = _config_dict(tmp_path)
    config["ambiguity_groups"][0]["candidates"] = ["A1"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="Ax"):
        load_config(tmp_path / "config.yaml")


def test_ambiguity_candidate_and_reference_must_appear_in_contacts(tmp_path):
    _write_tables(tmp_path, contacts=pd.DataFrame(
        {"identity_a": ["R1", "R1", "R2"], "identity_b": ["A1", "A2", "A1"],
         "contact": [1, 0, 0]}
    ))
    with pytest.raises(ValueError, match="R2|A2"):
        load_inputs(load_config(_write_config(tmp_path)))


@pytest.mark.parametrize(
    ("contacts", "expected"),
    [
        (
            pd.DataFrame({"identity_a": ["R1", "R1", "R2", "R2"], "identity_b": ["A1", "A2", "A1", "A2"], "contact": [1, 1, 0, 1]}),
            "R1",
        ),
        (
            pd.DataFrame({"identity_a": ["R1", "R1", "R2", "R2"], "identity_b": ["A1", "A2", "A1", "A2"], "contact": [0, 0, 0, 1]}),
            "R1",
        ),
    ],
)
def test_d14_requires_exactly_one_explicit_contact_per_reference(tmp_path, contacts, expected):
    _write_tables(tmp_path, contacts=contacts)
    with pytest.raises(ValueError, match=expected):
        load_inputs(load_config(_write_config(tmp_path)))


def test_d14_treats_missing_pair_as_unknown_not_zero(tmp_path):
    contacts = pd.DataFrame(
        {"identity_a": ["R1", "R2", "R2"], "identity_b": ["A1", "A1", "A2"], "contact": [1, 0, 1]}
    )
    _write_tables(tmp_path, contacts=contacts)
    with pytest.raises(ValueError, match="R1.*A2|A2.*R1"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_contacts_are_undirected_and_conflicting_duplicates_are_rejected(tmp_path):
    contacts = pd.DataFrame(
        {
            "identity_a": ["R1", "A2", "R2", "R2", "A1", "A1", "R1"],
            "identity_b": ["A1", "R1", "A1", "A2", "A2", "R2", "A1"],
            "contact": [1, 0, 0, 1, 0, 0, 0],
        }
    )
    _write_tables(tmp_path, contacts=contacts)
    with pytest.raises(ValueError, match="R1.*A1|A1.*R1"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_reference_cannot_be_an_ambiguity_group_name(tmp_path):
    _write_tables(tmp_path)
    config = _config_dict(tmp_path)
    config["ambiguity_groups"][0]["references"] = ["Ax", "R2"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="Ax"):
        load_config(tmp_path / "config.yaml")


def test_composite_ligand_or_receptor_is_rejected(tmp_path):
    _write_tables(tmp_path, lr_pairs=pd.DataFrame({"ligand": ["g1;g2"], "receptor": ["g2"]}))
    with pytest.raises(ValueError, match="multi-subunit|多亚基"):
        load_inputs(load_config(_write_config(tmp_path)))


def test_lr_pairs_with_missing_genes_are_dropped_and_recorded(tmp_path):
    _write_tables(tmp_path, lr_pairs=pd.DataFrame(
        {"ligand": ["g1", "missing"], "receptor": ["g2", "g1"], "weight": [1.0, 2.0]}
    ))
    inputs = load_inputs(load_config(_write_config(tmp_path)))
    assert len(inputs.lr_pairs) == 1
    assert inputs.dropped_lr_pairs == [{"ligand": "missing", "receptor": "g1", "weight": 2.0}]


def test_load_truth_is_explicit_and_validates_cell_ids(tmp_path):
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text("cell_id,true_label\nc1,A1\nc2,R1\n")
    truth = load_truth(truth_path)
    assert list(truth.columns) == ["true_label"]
    assert list(truth.index) == ["c1", "c2"]

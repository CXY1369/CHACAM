from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


CONVERTER_PATH = Path(__file__).parents[1] / "examples" / "c_elegans" / "convert_tintori.py"
IDENTITIES = [
    "ABala",
    "ABalp",
    "ABara",
    "ABarp",
    "ABpla",
    "ABplp",
    "ABpra",
    "ABprp",
    "MSa",
    "MSp",
    "Ca",
    "Cp",
    "Ea",
    "Ep",
    "D",
    "P4",
]


def converter_module():
    spec = importlib.util.spec_from_file_location("convert_tintori", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load converter from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["SeqA", "WBGeneA", "gene-a", "SeqA", ""],
            ["SeqB", "WBGeneB", "gene-b", "SeqB", ""],
            ["SeqC", "WBGeneC", "gene-c", "SeqC", ""],
            ["F19F10.1", "WBGeneF", "F19F10.1", "F19F10.1", ""],
            ["AltA", "WBGeneA", "gene-a", "SeqA", ""],
            ["ambiguous", "WBGeneA", "gene-a", "", ""],
            ["ambiguous", "WBGeneB", "gene-b", "", ""],
        ],
        columns=["Your Input", "WormBase Gene ID", "Public Name", "Sequence Name", "Other Name"],
    )


def make_contact_tables(pairs_by_level: dict[int, list[tuple[str, str]]] | None = None):
    pairs_by_level = pairs_by_level or {}
    return {
        level: pd.DataFrame(pairs, columns=["cell_i", "cell_j"])
        for level, pairs in pairs_by_level.items()
    }


def minimal_conversion_inputs(
    sample_rows: list[dict[str, object]],
    expression_rows: dict[str, list[object]],
    lr_rows: list[dict[str, object]] | None = None,
    features: pd.DataFrame | None = None,
):
    module = converter_module()
    s1 = pd.DataFrame(sample_rows)
    s2 = pd.DataFrame(expression_rows)
    if features is None:
        features = pd.DataFrame({"gene": ["SeqA"], "source": ["test"]})
    lr = pd.DataFrame(
        lr_rows
        or [
            {
                "ligand": "SeqA",
                "receptor": "SeqB",
                "pathway_name": "p",
                "evidence": "e",
                "annotation": "Cell-Cell Contact",
            }
        ]
    )
    contacts = make_contact_tables(
        {
            4: [("ABalp'", "MSa'"), ("ABarp'", "Ca'"), ("ABplp'", "Ea'"), ("ABplp'", "Ep'"), ("ABprp'", "Ep'"), ("ABprp'", "P4'")]
        }
    )
    result = module.convert_tables(
        sample_table=s1,
        expression_table=s2,
        gene_queries=make_queries(),
        lr_table=lr,
        contact_tables=contacts,
        feature_table=features,
        contact_overrides=pd.DataFrame(columns=["identity_a", "identity_b", "contact", "source"]),
    )
    return result


def test_sample_name_parser_extracts_exact_fields_and_rejects_invalid_names():
    module = converter_module()

    parsed = module.parse_sample_name("st258_MSx1_16-cell_r1")

    assert parsed == {"cell_id": "st258", "raw_label": "MSx1", "stage": 16, "sample_id": "r1"}
    assert isinstance(parsed["sample_id"], str)
    with pytest.raises(ValueError, match="sample name"):
        module.parse_sample_name("not-a-tintori-sample")


def test_sample_selection_keeps_only_usable_16_cell_samples_and_validates_labels():
    module = converter_module()
    s1 = pd.DataFrame(
        [
            {"sample": "st1_Ea_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_Ep_16-cell_r1", "Usable Quality?": "No"},
            {"sample": "st3_Ea_8-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st4_unknown_16-cell_r1", "Usable Quality?": "Yes"},
        ]
    )
    with pytest.raises(ValueError, match="raw label"):
        module.select_samples(s1)
    selected = module.select_samples(s1.iloc[:3])
    assert selected[["cell_id", "sample_id", "label", "stage"]].to_dict("records") == [
        {"cell_id": "st1", "sample_id": "r1", "label": "Ea", "stage": 16}
    ]


def test_ms_c_assignment_uses_expression_and_not_x1_x2_numbering():
    module = converter_module()
    s1 = pd.DataFrame(
        [
            {"sample": "st1_MSx1_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_MSx2_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st3_Cx2_16-cell_r2", "Usable Quality?": "Yes"},
            {"sample": "st4_Cx1_16-cell_r2", "Usable Quality?": "Yes"},
        ]
    )
    s2 = pd.DataFrame(
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [1.0, 1.0],
            "st2": [100.0, 2.0],
            "st3": [50.0, 3.0],
            "st4": [1.0, 4.0],
        }
    )
    result = minimal_conversion_inputs(s1.to_dict("records"), s2.to_dict(orient="list"))
    cells = result["cells"]
    assert cells["label"].tolist() == ["MSa", "MSp", "Cp", "Ca"]
    decisions = result["report"]["ms_c_decisions"]
    assert {item["cell_id"]: item["f19f10_1"] for item in decisions} == {
        "st1": 1.0,
        "st2": 100.0,
        "st3": 50.0,
        "st4": 1.0,
    }


def test_ms_c_classifier_uses_raw_expression_when_numbering_is_identical():
    result = minimal_conversion_inputs(
        [
            {"sample": "st1_MSx2_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_MSx2_16-cell_r1", "Usable Quality?": "Yes"},
        ],
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [0.0, 3.0],
            "st2": [100.0, 4.0],
        },
    )
    assert result["cells"]["label"].tolist() == ["MSa", "MSp"]


@pytest.mark.parametrize(
    ("values", "expected_kept"),
    [
        ((25.0, 1.5), True),
        ((24.999, 1.0), False),
        ((25.0, 1.6), False),
    ],
)
def test_ms_c_thresholds_use_closed_boundaries_and_original_rpkm(values, expected_kept):
    module = converter_module()
    s1 = pd.DataFrame(
        [
            {"sample": "st1_MSx1_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_MSx2_16-cell_r1", "Usable Quality?": "Yes"},
        ]
    )
    s2 = pd.DataFrame(
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [values[0], 3.0],
            "st2": [values[1], 4.0],
        }
    )
    result = minimal_conversion_inputs(s1.to_dict("records"), s2.to_dict(orient="list"))
    assert len(result["cells"]) == (2 if expected_kept else 0)
    if expected_kept:
        assert result["expression"].loc[0, "gene-a"] == np.log1p(3.0)


def test_ms_c_groups_are_scoped_by_embryo_and_undetermined_groups_are_deleted():
    module = converter_module()
    samples = [
        {"sample": "st1_MSx1_16-cell_r1", "Usable Quality?": "Yes"},
        {"sample": "st2_MSx2_16-cell_r1", "Usable Quality?": "Yes"},
        {"sample": "st3_MSx1_16-cell_r2", "Usable Quality?": "Yes"},
        {"sample": "st4_MSx2_16-cell_r2", "Usable Quality?": "Yes"},
        {"sample": "st5_Cx1_16-cell_r3", "Usable Quality?": "Yes"},
        {"sample": "st6_Cx1_16-cell_r3", "Usable Quality?": "Yes"},
        {"sample": "st7_Cx1_16-cell_r4", "Usable Quality?": "Yes"},
        {"sample": "st8_Cx2_16-cell_r4", "Usable Quality?": "Yes"},
    ]
    s2 = pd.DataFrame(
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [0.0, 1.0],
            "st2": [0.0, 1.0],
            "st3": [30.0, 1.0],
            "st4": [30.0, 1.0],
            "st5": [40.0, 1.0],
            "st6": [40.0, 1.0],
            "st7": [40.0, 1.0],
            "st8": [0.0, 1.0],
        }
    )
    result = minimal_conversion_inputs(samples, s2.to_dict(orient="list"))
    assert result["cells"]["cell_id"].tolist() == ["st7", "st8"]
    assert result["expression"]["cell_id"].tolist() == ["st7", "st8"]
    assert result["cells"]["label"].tolist() == ["Cp", "Ca"]
    deleted = result["report"]["dropped_cells"]
    assert {item["cell_id"] for item in deleted} == {"st1", "st2", "st3", "st4", "st5", "st6"}


def test_ms_c_classifier_handles_zero_one_two_and_three_cell_groups():
    module = converter_module()
    assert module.classify_ms_c_group("MS", {}) == ({}, "undetermined_MS_group_size_or_threshold")
    assert module.classify_ms_c_group("MS", {"st1": 30.0}) == ({}, "undetermined_MS_group_size_or_threshold")
    assert module.classify_ms_c_group("MS", {"st1": 0.0, "st2": 30.0}) == (
        {"st1": "MSa", "st2": "MSp"},
        "",
    )
    assert module.classify_ms_c_group("C", {"st1": 0.0, "st2": 30.0, "st3": 40.0}) == (
        {},
        "undetermined_C_group_size_or_threshold",
    )


def test_gene_mapping_handles_sequence_names_case_fallback_nonstrings_and_deduplication():
    module = converter_module()
    queries = pd.DataFrame(
        [
            ["SeqA", "WBGeneA", "gene-a", "SeqA"],
            ["SeqB", "WBGeneB", "gene-b", "SeqB"],
        ],
        columns=["Your Input", "WormBase Gene ID", "Public Name", "Sequence Name"],
    )
    aliases, public_names = module.build_gene_alias_map(queries)
    assert module.map_gene_name("SeqA", aliases, public_names) == ("WBGeneA", "gene-a")
    assert module.map_gene_name("seqa", aliases, public_names) == ("WBGeneA", "gene-a")
    assert module.map_gene_name(pd.Timestamp("2020-01-01"), aliases, public_names) is None

    expression = pd.DataFrame(
        {
            "transcript": ["gene-a", "SeqA", "not-known", "ambiguous"],
            "st1": [1.0, 2.0, 3.0, 4.0],
            "st2": [1.0, 2.0, 3.0, 4.0],
        }
    )
    s1 = pd.DataFrame(
        [
            {"sample": "st1_Ea_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_Ep_16-cell_r1", "Usable Quality?": "Yes"},
        ]
    )
    result = minimal_conversion_inputs(
        s1.to_dict("records"),
        expression.to_dict(orient="list"),
    )
    assert result["expression"].columns.tolist() == ["cell_id", "gene-a"]
    assert result["report"]["gene_mapping"]["duplicate_id_dropped"] == 1


def test_gene_mapping_prefers_exact_alias_before_casefold_fallback():
    module = converter_module()
    queries = pd.DataFrame(
        [
            ["Abc", "WBGene1", "gene-one", "Abc"],
            ["abc", "WBGene2", "gene-two", "abc"],
        ],
        columns=["Your Input", "WormBase Gene ID", "Public Name", "Sequence Name"],
    )
    aliases, public_names = module.build_gene_alias_map(queries)
    assert module.map_gene_name("Abc", aliases, public_names) == ("WBGene1", "gene-one")


def test_duplicate_gene_id_keeps_original_table_row_order():
    module = converter_module()
    expression = pd.DataFrame(
        {
            "transcript": ["SeqA", "SeqB", "gene-a"],
            "st1": [1.0, 2.0, 3.0],
        }
    )
    converted, _ = module.convert_expression_table(expression, make_queries(), ["st1"])
    assert converted.columns.tolist() == ["cell_id", "gene-b", "gene-a"]


@pytest.mark.parametrize(
    "bad_rows, message",
    [
        ([{"transcript": "gene-a", "st1": 1.0}, {"transcript": "gene-a", "st1": 2.0}], "two rows"),
        ([{"transcript": "SeqA", "st1": 1.0}, {"transcript": "AltA", "st1": 2.0}], "two rows"),
    ],
)
def test_gene_mapping_rejects_ambiguous_or_duplicate_final_genes(bad_rows, message):
    module = converter_module()
    with pytest.raises(ValueError, match=message):
        module.convert_expression_table(
            pd.DataFrame(bad_rows),
            make_queries(),
            ["st1"],
        )


def test_gene_mapping_rejects_duplicate_final_public_names_from_distinct_ids():
    module = converter_module()
    queries = pd.DataFrame(
        [
            ["SeqA", "WBGeneA", "same", "SeqA"],
            ["SeqB", "WBGeneB", "same", "SeqB"],
        ],
        columns=["Your Input", "WormBase Gene ID", "Public Name", "Sequence Name"],
    )
    with pytest.raises(ValueError, match="final gene"):
        module.convert_expression_table(
            pd.DataFrame({"transcript": ["SeqA", "SeqB"], "st1": [1.0, 2.0]}),
            queries,
            ["st1"],
        )


def test_expression_conversion_reports_unknown_and_nonstring_rows_and_preserves_order():
    module = converter_module()
    expression = pd.DataFrame(
        {
            "transcript": ["SeqB", "unknown", pd.Timestamp("2020-01-01"), "SeqA"],
            "st1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    converted, report = module.convert_expression_table(expression, make_queries(), ["st1"])
    assert converted.columns.tolist() == ["cell_id", "gene-b", "gene-a"]
    assert converted.iloc[0, 1:].tolist() == [np.log1p(1.0), np.log1p(4.0)]
    assert report["unmapped"] == 2


def test_feature_mapping_renames_genes_and_rejects_missing_mapped_gene():
    module = converter_module()
    aliases, public_names = module.build_gene_alias_map(make_queries())
    features = pd.DataFrame({"gene": ["SeqA", "SeqB"], "source": ["a", "b"]})
    mapped = module.convert_features(features, aliases, public_names, {"gene-a", "gene-b"})
    assert mapped.columns.tolist() == ["gene"]
    assert mapped.to_dict("records") == [{"gene": "gene-a"}, {"gene": "gene-b"}]
    with pytest.raises(ValueError, match="missing from expression"):
        module.convert_features(
            pd.DataFrame({"gene": ["SeqC"], "source": ["c"]}),
            aliases,
            public_names,
            {"gene-a"},
        )


def test_convert_tables_features_have_only_gene_column():
    result = minimal_conversion_inputs(
        [
            {"sample": "st1_Ea_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_Ep_16-cell_r1", "Usable Quality?": "Yes"},
        ],
        {
            "transcript": ["F19F10.1", "SeqA", "SeqB"],
            "st1": [1.0, 3.0, 4.0],
            "st2": [2.0, 4.0, 5.0],
        },
    )
    assert result["features"].columns.tolist() == ["gene"]


def test_lr_conversion_filters_both_non_lr_spellings_keeps_missing_expression_and_order():
    module = converter_module()
    aliases, public_names = module.build_gene_alias_map(make_queries())
    lr = pd.DataFrame(
        [
            ["seqa", "SeqB", "p1", "e1", "Cell-Cell Contact"],
            ["SeqA", "SeqC", "p2", "e2", "NON-LR annotation"],
            ["SeqA", "SeqB", "p3", "e3", "non-Ligand-Receptor"],
            ["SeqC", "SeqB", "p4", "e4", "Cell-Cell Contact"],
            ["missing", "SeqB", "p5", "e5", "Cell-Cell Contact"],
        ],
        columns=["ligand", "receptor", "pathway_name", "evidence", "annotation"],
    )
    converted, report = module.convert_lr_pairs(lr, aliases, public_names, {"gene-a", "gene-b", "gene-c"})
    assert converted["ligand"].tolist() == ["gene-a", "gene-c"]
    assert converted["receptor"].tolist() == ["gene-b", "gene-b"]
    assert converted.columns.tolist() == ["ligand", "receptor", "pathway_name", "evidence", "annotation"]
    assert report["non_lr"] == 2
    assert report["unmapped"] == 1


def test_lr_mapped_but_missing_expression_is_retained_and_reported():
    module = converter_module()
    aliases, public_names = module.build_gene_alias_map(make_queries())
    lr = pd.DataFrame(
        [["SeqA", "SeqB", "p", "e", "Cell-Cell Contact"]],
        columns=["ligand", "receptor", "pathway_name", "evidence", "annotation"],
    )
    converted, report = module.convert_lr_pairs(lr, aliases, public_names, {"gene-a"})
    assert converted.to_dict("records") == [
        {
            "ligand": "gene-a",
            "receptor": "gene-b",
            "pathway_name": "p",
            "evidence": "e",
            "annotation": "Cell-Cell Contact",
        }
    ]
    assert report["mapped_but_missing_from_expression"] == [
        {"row": 0, "ligand": "gene-a", "receptor": "gene-b"}
    ]


def test_contacts_use_max_level_canonical_pairs_exact_sources_and_sorted_output():
    module = converter_module()
    pairs = {
        1: [("ABala'", "MSa'")],
        2: [("ABara'", "ABalp'")],
        3: [("Ca'", "ABara'")],
        4: [("ABalp'", "MSa'"), ("ABalp'", "MSa'")],
    }
    overrides = pd.DataFrame(
        [["MSa", "ABala", 0, "CMOS source"], ["ABara", "Ca", 0, "CMOS source"]],
        columns=["identity_a", "identity_b", "contact", "source"],
    )
    levels, contacts = module.convert_contacts(make_contact_tables(pairs), overrides)
    assert len(levels) == 120
    assert levels[["identity_a", "identity_b"]].apply(tuple, axis=1).tolist() == sorted(
        levels[["identity_a", "identity_b"]].apply(tuple, axis=1).tolist()
    )
    assert levels.loc[(levels["identity_a"] == "ABala") & (levels["identity_b"] == "MSa"), "level"].item() == 1
    assert contacts.loc[(contacts["identity_a"] == "ABala") & (contacts["identity_b"] == "MSa"), "source"].item() == "override: CMOS source"
    assert contacts.loc[(contacts["identity_a"] == "ABalp") & (contacts["identity_b"] == "MSa"), "contact"].item() == 1
    assert contacts.loc[(contacts["identity_a"] == "ABala") & (contacts["identity_b"] == "ABalp"), "source"].item() == "cshaper_absent"
    assert not ((contacts["identity_a"] == "ABara") & (contacts["identity_b"] == "ABalp")).any()
    assert set(contacts["contact"]) == {0, 1}


@pytest.mark.parametrize(
    "contact_tables",
    [
        make_contact_tables({1: [("ABala'", "ABalp'")], 4: [("ABalp'", "ABala'")]}),
        make_contact_tables({4: [("ABalp'", "ABala'")], 1: [("ABala'", "ABalp'")]}),
    ],
)
def test_contacts_take_level_four_regardless_of_sheet_order(contact_tables):
    module = converter_module()
    overrides = pd.DataFrame(columns=["identity_a", "identity_b", "contact", "source"])
    levels, contacts = module.convert_contacts(contact_tables, overrides)
    pair = (levels["identity_a"] == "ABala") & (levels["identity_b"] == "ABalp")
    contact_pair = (contacts["identity_a"] == "ABala") & (contacts["identity_b"] == "ABalp")
    assert levels.loc[pair, "level"].item() == 4
    assert contacts.loc[contact_pair, "contact"].item() == 1
    assert contacts.loc[contact_pair, "source"].item() == "cshaper_sheet4"


def test_contacts_reject_self_loops_duplicate_overrides_and_unknown_identities():
    module = converter_module()
    with pytest.raises(ValueError, match="self-contact"):
        module.convert_contacts(make_contact_tables({4: [("ABala'", "ABala'")]}), pd.DataFrame(columns=["identity_a", "identity_b", "contact", "source"]))
    duplicate = pd.DataFrame(
        [["ABala", "MSa", 0, "a"], ["MSa", "ABala", 1, "b"]],
        columns=["identity_a", "identity_b", "contact", "source"],
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.convert_contacts(make_contact_tables(), duplicate)
    unknown = pd.DataFrame(
        [["unknown", "MSa", 0, "a"]],
        columns=["identity_a", "identity_b", "contact", "source"],
    )
    with pytest.raises(ValueError, match="unknown identity"):
        module.convert_contacts(make_contact_tables(), unknown)


def test_same_input_conversion_is_byte_reproducible_and_config_loads(tmp_path):
    module = converter_module()
    s1 = pd.DataFrame(
        [
            {"sample": "st1_Ea_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_Ep_16-cell_r1", "Usable Quality?": "Yes"},
        ]
    )
    s2 = pd.DataFrame(
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [1.0, 3.0],
            "st2": [2.0, 4.0],
        }
    )
    kwargs = {
        "sample_table": s1,
        "expression_table": s2,
        "gene_queries": make_queries(),
        "lr_table": pd.DataFrame([{"ligand": "SeqA", "receptor": "SeqB", "pathway_name": "p", "evidence": "e", "annotation": "a"}]),
        "contact_tables": make_contact_tables(),
        "feature_table": pd.DataFrame({"gene": ["SeqA"], "source": ["test"]}),
        "contact_overrides": pd.DataFrame(columns=["identity_a", "identity_b", "contact", "source"]),
    }
    first = module.convert_tables(**kwargs)
    second = module.convert_tables(**kwargs)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    module.write_conversion_outputs(first, first_dir, {}, CONVERTER_PATH)
    module.write_conversion_outputs(second, second_dir, {}, CONVERTER_PATH)
    for name in ("expression", "cells", "features", "lr_pairs", "contacts", "contact_levels"):
        assert (first_dir / f"{name}.csv").read_bytes() == (second_dir / f"{name}.csv").read_bytes()


def test_conversion_report_uses_repo_relative_and_outside_source_paths(tmp_path):
    module = converter_module()
    result = minimal_conversion_inputs(
        [
            {"sample": "st1_Ea_16-cell_r1", "Usable Quality?": "Yes"},
            {"sample": "st2_Ep_16-cell_r1", "Usable Quality?": "Yes"},
        ],
        {
            "transcript": ["F19F10.1", "SeqA"],
            "st1": [1.0, 3.0],
            "st2": [2.0, 4.0],
        },
    )
    inside = CONVERTER_PATH.parents[2] / "data" / "reference" / "MANIFEST.md"
    outside = tmp_path / "outside.raw"
    outside.write_bytes(b"outside")
    report = module.write_conversion_outputs(
        result,
        tmp_path / "report",
        {"inside": inside, "outside": outside},
        CONVERTER_PATH,
    )
    assert report["source_files_sha256"]["inside"]["path"] == "data/reference/MANIFEST.md"
    assert report["source_files_sha256"]["outside"]["path"] == str(outside)


@pytest.mark.parametrize(
    ("status_text", "expected_dirty"),
    [("", False), (" M file", True)],
)
def test_git_provenance_uses_one_repo_root_and_reports_dirty_state(monkeypatch, status_text, expected_dirty):
    module = converter_module()
    repo_root = CONVERTER_PATH.parents[2]
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if args == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout=status_text)
        raise AssertionError(args)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._git_provenance(repo_root) == ("abc123", expected_dirty)
    assert [call[0] for call in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain"],
    ]
    assert {call[1]["cwd"] for call in calls} == {repo_root}


def test_git_provenance_returns_unknown_state_when_git_is_unavailable(monkeypatch):
    module = converter_module()

    def fake_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._git_provenance(CONVERTER_PATH.parents[2]) == (None, None)


def test_static_features_overrides_and_config_satisfy_contact_validation():
    module = converter_module()
    repo = Path(__file__).parents[1]
    features = pd.read_csv(repo / "examples" / "c_elegans" / "features_tintori16.csv")
    overrides = pd.read_csv(repo / "examples" / "c_elegans" / "contact_overrides.csv")
    assert features.columns.tolist() == ["gene", "source"]
    assert len(features) == 26
    assert features["gene"].is_unique
    assert overrides.columns.tolist() == ["identity_a", "identity_b", "contact", "source"]
    levels, contacts = module.convert_contacts(
        make_contact_tables(
            {
                4: [
                    ("ABalp'", "MSa'"),
                    ("ABarp'", "Ca'"),
                    ("ABplp'", "Ea'"),
                    ("ABplp'", "Ep'"),
                    ("ABprp'", "Ep'"),
                    ("ABprp'", "P4'"),
                ]
            }
        ),
        overrides,
    )
    assert len(levels) == 120
    from chacam.schema import AmbiguityGroup, _validate_contacts, load_config

    config = load_config(repo / "examples" / "c_elegans" / "config_tintori16.yaml")
    assert config.inputs.expression_scale == "log1p"
    _validate_contacts(
        contacts,
        config.ambiguity_groups,
    )

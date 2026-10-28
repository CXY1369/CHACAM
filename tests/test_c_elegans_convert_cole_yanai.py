from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CONVERTER_PATH = Path(__file__).parents[1] / "examples" / "c_elegans" / "convert_cole_yanai.py"


def converter_module():
    if not CONVERTER_PATH.exists():
        raise AssertionError(f"converter has not been implemented: {CONVERTER_PATH}")
    spec = importlib.util.spec_from_file_location("convert_cole_yanai", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load converter from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def s4_row(state_id: str, identities: tuple[str, ...], declared_count: int, stage: int | None = None):
    return {
        "state_id": state_id,
        "identities": identities,
        "declared_count": declared_count,
        "stage": stage,
    }


def test_stage_annotations_are_parsed_before_splitting_identities():
    module = converter_module()

    assert module.parse_s5_state("Ea 15") == ("Ea", 15)
    assert module.parse_s5_state("P4 (28-cell stage)") == ("P4", 28)
    assert module.parse_s5_state("Eala 102") == ("Eala", 102)
    assert module.parse_s4_state("Ea_15") == (("Ea",), 15)
    assert module.parse_s4_state("Ea (28-cell stage)") == (("Ea",), 28)


def test_state_matching_requires_every_identity_to_match_the_x_pattern():
    module = converter_module()
    rows = [
        s4_row("18", ("ABpla", "ABpra"), 9),
        s4_row("23", ("Ca", "Cp"), 6),
        s4_row("118", ("Daa", "Dap", "Dpa", "Dpp"), 14),
    ]

    matched = module.match_s5_state("ABpxa", rows)
    assert matched["identities"] == ("ABpla", "ABpra")
    assert matched["label_kind"] == "equivalence"
    assert module.match_s5_state("Cx", rows)["identities"] == ("Ca", "Cp")
    assert module.match_s5_state("Daa", rows)["label_kind"] == "equivalence"

    with pytest.raises(ValueError, match="exactly one"):
        module.match_s5_state("ABpx", [s4_row("bad", ("ABpla", "ABpr"), 1)])


def test_stage_annotation_disambiguates_s4_rows():
    module = converter_module()
    rows = [
        s4_row("21", ("Ea",), 3, 15),
        s4_row("22", ("Ea",), 4, 28),
        s4_row("118", ("Caa", "Cpa"), 7, 28),
        s4_row("70", ("Caa",), 1, 51),
    ]

    assert module.match_s5_state("Ea 15", rows)["state_id"] == "21"
    assert module.match_s5_state("Caa (51-cell stage)", rows)["state_id"] == "70"


def test_only_the_four_registered_state_corrections_are_allowed():
    module = converter_module()
    rows = [
        s4_row("87", ("ABplpaa", "ABprpaa"), 2),
        s4_row("1", ("P",), 3, 0),
        s4_row("107", ("MSaaap", "MSaapp", "MSpaap", "MSpapp"), 9),
        s4_row("109", ("MSapaa", "MSapap", "MSppaa", "MSppap"), 10),
    ]

    assert module.match_s5_state("Abpxpaa", rows)["state_id"] == "87"
    assert module.match_s5_state("P0", rows)["state_id"] == "1"
    assert module.match_s5_state("MSxaap", rows)["state_id"] == "107"
    assert module.match_s5_state("MSxpaa", rows)["state_id"] == "109"
    with pytest.raises(ValueError, match="exactly one"):
        module.match_s5_state("Unknown", rows)


def test_cell_count_validation_records_only_the_registered_ms_exception():
    module = converter_module()
    rows = [
        s4_row("20", ("MSa", "MSp"), 12),
        s4_row("21", ("Ea",), 2),
    ]

    report = module.validate_cell_counts(rows, ["20", "20", "20", "21", "21"])
    assert report["exceptions"] == [{"state_id": "20", "declared": 12, "observed": 3}]

    with pytest.raises(ValueError, match="state 21"):
        module.validate_cell_counts(rows, ["20", "20", "20", "21"])


def test_embryo_ids_follow_the_table_s5_to_s1_mapping_rule():
    module = converter_module()

    assert module.normalize_embryo_id("13c2") == "13.2"
    assert module.normalize_embryo_id("13c") == "13"
    assert module.normalize_embryo_id("P01") == "1.1"
    with pytest.raises(ValueError, match="embryo"):
        module.normalize_embryo_id("not-an-embryo")


def test_soft_titles_must_match_table_s5_sample_names():
    module = converter_module()
    s5 = pd.DataFrame(
        {
            "GEO entry": ["GSM1", "GSM2"],
            "sample name": ["13_r1_1", "13_r2_1"],
        }
    )
    module.validate_soft_titles(s5, {"GSM1": "13_r1_1", "GSM2": "13_r2_1"})
    with pytest.raises(ValueError, match="SOFT.*sample"):
        module.validate_soft_titles(s5, {"GSM1": "13_r1_1", "GSM2": "wrong"})


def test_out_of_range_cell_removes_its_entire_embryo_and_keeps_closed_boundaries():
    module = converter_module()
    s5 = pd.DataFrame(
        {
            "GEO entry": ["GSM1", "GSM2", "GSM3"],
            "sample name": ["4_r1_1", "4_r1_2", "5_r1_1"],
            "Embryo ID": ["4c1", "4c1", "5c"],
            "Number of transcripts": [94, 106, 100],
        }
    )
    counts = pd.DataFrame(
        {"GSM1": [100], "GSM2": [100], "GSM3": [100]},
        index=["gene"],
    )

    kept, report = module.filter_misaligned_cells(s5, counts)
    assert report["out_of_range_cells"] == ["GSM1"]
    assert report["removed_embryos"] == ["4c1"]
    assert kept["GEO entry"].tolist() == ["GSM3"]

    boundary = s5.iloc[[1, 2]].copy()
    boundary.loc[boundary["GEO entry"] == "GSM2", "Embryo ID"] = "6c"
    kept, report = module.filter_misaligned_cells(boundary, counts)
    assert report["out_of_range_cells"] == []
    assert len(kept) == 2


def test_cpm_uses_all_raw_rows_for_library_size_before_gene_mapping():
    module = converter_module()
    raw_counts = pd.DataFrame(
        {"GSM1": [2, 3, 5], "GSM2": [0, 4, 6]},
        index=["mapped", "unmapped", "mapped2"],
    )

    result = module.to_log1p_cpm(raw_counts, ["GSM1", "GSM2"])
    assert result.loc["GSM1", "mapped"] == pytest.approx(np.log1p(2 / 10 * 1_000_000))
    assert result.loc["GSM2", "mapped2"] == pytest.approx(np.log1p(6 / 10 * 1_000_000))


def test_contact_levels_keep_sheet4_as_one_and_sheet1_to_3_unknown():
    module = converter_module()
    contact_tables = {
        1: pd.DataFrame([["A", "B"]]),
        2: pd.DataFrame([["A", "B"]]),
        3: pd.DataFrame([["A", "C"]]),
        4: pd.DataFrame([["A", "B"]]),
    }

    levels, contacts, missing = module.convert_cole_contacts(contact_tables, ["A", "B", "C"])
    assert levels.loc[levels["identity_a"].eq("A") & levels["identity_b"].eq("B"), "level"].iat[0] == 4
    assert contacts.loc[contacts["identity_a"].eq("A") & contacts["identity_b"].eq("B"), "contact"].iat[0] == 1
    assert not ((contacts["identity_a"] == "A") & (contacts["identity_b"] == "C")).any()
    assert missing == []

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "c_elegans" / "diagnose_stage1.py"


def diagnose_module():
    spec = importlib.util.spec_from_file_location("diagnose_stage1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load diagnosis script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group(name: str = "Amb", references: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")):
    return SimpleNamespace(name=name, candidates=("X", "Y"), references=references)


def test_reference_metrics_require_nonempty_all_correct_reference_bucket_and_count_absent_false_assignments():
    module = diagnose_module()
    cells = pd.DataFrame(
        {
            "sample_id": ["r1"] * 6,
            "label": ["R1", "X", "R2", "R3", "X", "R5"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6"],
    )
    stage1 = pd.DataFrame(
        {
            "fold": ["r1"] * 6,
            "cell_id": cells.index,
            "observed_label": cells["label"].tolist(),
            "stage1_label": ["R1", "R1", "X", "R3", "R4", "R5"],
        }
    )

    metrics = module.summarize_stage1_metrics(stage1, cells, [group()])

    assert metrics["ref_present_total"] == 4
    assert metrics["ref_present_correct"] == 2
    assert metrics["ref_false_when_absent"] == 1


def test_fine_metrics_exclude_ambiguity_group_observations():
    module = diagnose_module()
    cells = pd.DataFrame(
        {"sample_id": ["r1", "r1"], "label": ["R5", "Amb"]},
        index=["c1", "c2"],
    )
    stage1 = pd.DataFrame(
        {
            "fold": ["r1", "r1"],
            "cell_id": ["c1", "c2"],
            "observed_label": ["R5", "Amb"],
            "stage1_label": ["R5", "R5"],
        }
    )

    metrics = module.summarize_stage1_metrics(stage1, cells, [group()])

    assert metrics["fine_correct"] == 1
    assert metrics["fine_total"] == 1


def test_s_rank_true_uses_strictly_greater_values_and_empty_for_missing_identity():
    module = diagnose_module()
    similarity = pd.DataFrame(
        [[0.9, 0.9, 0.1], [0.2, 0.5, 0.5]],
        index=["c1", "c2"],
        columns=["A", "B", "C"],
    )

    ranks = module.rank_true_labels(similarity, pd.Series(["B", "C"], index=["c1", "c2"]))
    missing = module.rank_true_labels(similarity.loc[["c1"]], pd.Series(["Z"], index=["c1"]))

    assert ranks.tolist() == [1, 1]
    assert missing.tolist() == [""]


def test_hvg_uses_training_rows_only_and_stable_column_order_for_equal_variance():
    module = diagnose_module()
    expression = pd.DataFrame(
        {
            "g1": [0.0, 2.0, 0.0],
            "g2": [1.0, -1.0, 100.0],
            "g3": [0.0, 0.0, 100.0],
        },
        index=["train1", "train2", "heldout"],
    )

    selected = module.select_hvg_genes(expression, ["train1", "train2"], 1)
    changed_heldout = expression.copy()
    changed_heldout.loc["heldout"] = [1000.0, 1000.0, 1000.0]
    selected_after_heldout_change = module.select_hvg_genes(changed_heldout, ["train1", "train2"], 1)
    changed_training = expression.copy()
    changed_training.loc["train2", "g2"] = 4.0
    selected_after_training_change = module.select_hvg_genes(changed_training, ["train1", "train2"], 1)

    assert selected == ["g1"]
    assert selected_after_heldout_change == ["g1"]
    assert selected_after_training_change == ["g2"]


def test_detected_genes_use_training_rows_only():
    module = diagnose_module()
    expression = pd.DataFrame(
        {"g1": [0.0, 1.0, 0.0], "g2": [0.0, 0.0, 1.0], "g3": [0.0, 0.0, 0.0]},
        index=["train1", "train2", "heldout"],
    )

    assert module.select_detected_genes(expression, ["train1", "train2"]) == ["g1"]


def _d3_inputs_and_stage1():
    module = diagnose_module()
    cells = pd.DataFrame(
        {
            "sample_id": ["r1"] * 4,
            "label": ["X", "Amb", "Amb", "Z"],
        },
        index=["ref", "u", "v", "other"],
    )
    expression = pd.DataFrame(
        {"ligand": [10.0, 0.0, 0.0, 0.0], "receptor": [0.0, 2.0, 1.0, 0.0]},
        index=cells.index,
    )
    inputs = SimpleNamespace(
        cells=cells,
        expression=expression,
        features=pd.DataFrame({"gene": ["ligand", "receptor"]}),
        lr_pairs=pd.DataFrame({"ligand": ["ligand"], "receptor": ["receptor"], "weight": [1.0]}),
        contacts=pd.DataFrame(
            {"identity_a": ["R", "R"], "identity_b": ["A", "B"], "contact": [1.0, 0.0]}
        ),
        ambiguity_groups=(SimpleNamespace(name="Amb", candidates=("A", "B"), references=("R",)),),
        used_feature_genes=["ligand", "receptor"],
    )
    config = SimpleNamespace(
        reference=SimpleNamespace(mode="training_mean"),
        assignment=SimpleNamespace(mode="unique"),
        attention=SimpleNamespace(normalization="l1", temperature=None),
        cci=SimpleNamespace(
            transform="none", direction="ref_to_cand", lr_selection="given", top_k=1
        ),
        ambiguity_groups=inputs.ambiguity_groups,
    )
    stage1 = pd.DataFrame(
        {
            "fold": ["r1"] * 4,
            "cell_id": cells.index,
            "observed_label": cells["label"].tolist(),
            "stage1_label": ["R", "Amb", "Amb", "Z"],
        }
    )
    return module, inputs, config, stage1


def test_d3_observed_uses_observed_labels_and_wrong_reference_lists_the_misassigned_cell():
    module, inputs, config, stage1 = _d3_inputs_and_stage1()

    d3, wrong = module.run_d3(inputs, config, stage1)

    stage1_result = d3.loc[d3["refs"].eq("stage1")]
    observed_result = d3.loc[d3["refs"].eq("observed")]
    assert set(stage1_result["status"]) == {"resolved"}
    assert set(observed_result["status"]) == {"unresolved"}
    assert observed_result["unresolved_reason"].tolist() == ["unresolved_no_reference"] * 2
    assert wrong.to_dict("records") == [
        {
            "fold": "r1",
            "group": "Amb",
            "reference_identity": "R",
            "reference_cell_id": "ref",
            "observed_label": "X",
        }
    ]


def test_d3_uses_d2_stage1_even_when_baseline_predictions_are_different():
    module, inputs, config, stage1 = _d3_inputs_and_stage1()
    baseline = stage1.assign(
        stage1_label="Z", final_label="wrong", status="unresolved", unresolved_reason="baseline-only"
    )

    d3_without_baseline, _ = module.run_d3(inputs, config, stage1)
    d3_with_different_baseline, _ = module.run_d3(inputs, config, stage1, baseline_predictions=baseline)
    consistency = module.check_consistency(stage1, d3_with_different_baseline, baseline, {"Amb"})

    pd.testing.assert_frame_equal(d3_without_baseline, d3_with_different_baseline)
    assert consistency["triangulation_match"] is False


def _consistency_fixture():
    d2 = pd.DataFrame(
        [
            {"fold": "r1", "cell_id": "amb1", "observed_label": "Amb", "stage1_label": "Amb"},
            {"fold": "r1", "cell_id": "amb2", "observed_label": "Amb", "stage1_label": "Amb"},
            {"fold": "r1", "cell_id": "fine1", "observed_label": "X", "stage1_label": "X"},
        ]
    )
    d3 = pd.DataFrame(
        [
            {
                "fold": "r1",
                "group": "Amb",
                "refs": "stage1",
                "cell_id": "amb1",
                "final_label": "A",
                "status": "resolved",
                "unresolved_reason": "",
            },
            {
                "fold": "r1",
                "group": "Amb",
                "refs": "stage1",
                "cell_id": "amb2",
                "final_label": "Amb",
                "status": "unresolved",
                "unresolved_reason": "",
            },
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "cell_id": "amb1",
                "stage1_label": "Amb",
                "final_label": "A",
                "status": "resolved",
                "unresolved_reason": float("nan"),
            },
            {
                "cell_id": "amb2",
                "stage1_label": "Amb",
                "final_label": "Amb",
                "status": "unresolved",
                "unresolved_reason": "",
            },
            {
                "cell_id": "fine1",
                "stage1_label": "X",
                "final_label": "X",
                "status": "assigned",
                "unresolved_reason": "",
            },
        ]
    )
    return d2, d3, baseline, {"Amb"}


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unchanged", {"stage1_match": True, "triangulation_match": True}),
        ("stage1 rows empty", {"stage1_match": True, "triangulation_match": False}),
        ("one ambiguity row removed", {"stage1_match": True, "triangulation_match": False}),
        ("duplicate existing cell row", {"stage1_match": True, "triangulation_match": False}),
        ("group changed", {"stage1_match": True, "triangulation_match": False}),
        ("final_label only", {"stage1_match": True, "triangulation_match": False}),
        ("status only", {"stage1_match": True, "triangulation_match": False}),
        ("unresolved_reason only", {"stage1_match": True, "triangulation_match": False}),
        ("fine stage1_label changed", {"stage1_match": False, "triangulation_match": True}),
        ("fine baseline row removed", {"stage1_match": False, "triangulation_match": True}),
    ],
)
def test_consistency_checks_complete_stage1_and_triangulation_sets(case, expected):
    module = diagnose_module()
    d2, d3, baseline, group_names = _consistency_fixture()
    if case == "stage1 rows empty":
        d3 = d3.iloc[0:0].copy()
    elif case == "one ambiguity row removed":
        d3 = d3.iloc[[0]].copy()
    elif case == "duplicate existing cell row":
        d3 = pd.concat([d3, d3.iloc[[0]]], ignore_index=True)
    elif case == "group changed":
        d3 = d3.copy()
        d3.loc[d3.index[0], "group"] = "Other"
    elif case == "final_label only":
        d3 = d3.copy()
        d3.loc[d3.index[0], "final_label"] = "B"
    elif case == "status only":
        d3 = d3.copy()
        d3.loc[d3.index[0], "status"] = "unresolved"
    elif case == "unresolved_reason only":
        d3 = d3.copy()
        d3.loc[d3.index[0], "unresolved_reason"] = "unresolved_tie"
    elif case == "fine stage1_label changed":
        baseline = baseline.copy()
        baseline.loc[baseline["cell_id"].eq("fine1"), "stage1_label"] = "Y"
    elif case == "fine baseline row removed":
        baseline = baseline.loc[baseline["cell_id"].ne("fine1")].copy()

    assert module.check_consistency(d2, d3, baseline, group_names) == expected


def test_t2_reference_absent_without_assignment_is_not_false_assignment():
    module = diagnose_module()
    cells = pd.DataFrame(
        {
            "sample_id": ["r1"] * 6,
            "label": ["R1", "X", "R2", "R3", "X", "R5"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6"],
    )
    stage1 = pd.DataFrame(
        {
            "fold": ["r1"] * 6,
            "cell_id": cells.index,
            "observed_label": cells["label"].tolist(),
            "stage1_label": ["R1", "R1", "X", "R3", "R4", "R5"],
        }
    )

    metrics = module.summarize_stage1_metrics(
        stage1, cells, [group(references=("R1", "R2", "R3", "R4", "R5", "R6"))]
    )

    assert metrics["ref_false_when_absent"] == 1


def test_t3_wrong_reference_is_not_reported_when_group_is_unresolved():
    module, inputs, config, stage1 = _d3_inputs_and_stage1()
    unresolved_stage1 = stage1.assign(stage1_label=["R", "Amb", "Z", "Z"])

    _d3, wrong = module.run_d3(inputs, config, unresolved_stage1)

    assert wrong.empty


@pytest.mark.parametrize(
    ("consistency", "missing_key"),
    [
        ({"stage1_match": True, "triangulation_match": True}, None),
        ({"stage1_match": False, "triangulation_match": True}, "stage1_match"),
        ({"stage1_match": True, "triangulation_match": False}, "triangulation_match"),
    ],
)
def test_require_consistency(consistency, missing_key):
    module = diagnose_module()
    if missing_key is None:
        module.require_consistency(consistency)
    else:
        with pytest.raises(RuntimeError, match=missing_key):
            module.require_consistency(consistency)


def test_main_stops_before_writing_when_consistency_fails(monkeypatch, tmp_path):
    module = diagnose_module()
    output_dir = tmp_path / "runs" / "m4"
    config = SimpleNamespace(output_dir=output_dir, ambiguity_groups=(group(),))
    empty = pd.DataFrame()
    write_called = False

    monkeypatch.setattr(module.chacam, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module.chacam,
        "load_inputs",
        lambda _config: SimpleNamespace(expression=empty, cells=empty),
    )
    monkeypatch.setattr(module, "build_d1_tables", lambda _expression, _cells: (empty, empty))
    monkeypatch.setattr(module, "run_d2", lambda _inputs, _config: (empty, empty, empty))
    monkeypatch.setattr(module, "run_d3", lambda _inputs, _config, _d2: (empty, empty))
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: empty)
    monkeypatch.setattr(
        module,
        "check_consistency",
        lambda _d2, _d3, _baseline, _group_names: {
            "stage1_match": True,
            "triangulation_match": False,
        },
    )

    def fail_if_called(*_args, **_kwargs):
        nonlocal write_called
        write_called = True
        raise AssertionError("_write_outputs must not be called")

    monkeypatch.setattr(module, "_write_outputs", fail_if_called)

    with pytest.raises(RuntimeError, match="triangulation_match"):
        module.main()

    assert not write_called
    diagnosis_dir = output_dir.parent / "c_elegans_tintori16_diagnosis"
    assert not list(diagnosis_dir.iterdir())


def test_d1_tables_include_zero_label_rows_and_count_positive_expression_values():
    module = diagnose_module()
    cells = pd.DataFrame(
        {
            "sample_id": ["r1", "r1", "r2"],
            "label": ["A", "B", "A"],
        },
        index=["c1", "c2", "c3"],
    )
    expression = pd.DataFrame(
        {"g1": [0.0, 2.0, 0.0], "g2": [1.0, 0.0, 0.0]},
        index=cells.index,
    )

    counts, quality = module.build_d1_tables(expression, cells)

    assert len(counts) == 4
    assert counts.loc[(counts["fold"] == "r1") & (counts["identity"] == "A"), "n_training"].item() == 1
    assert counts.loc[(counts["fold"] == "r2") & (counts["identity"] == "B"), "n_heldout"].item() == 0
    assert quality.set_index("cell_id").loc["c1", "n_detected"] == 1
    assert quality.set_index("cell_id").loc["c2", "n_detected"] == 1


def test_csv_serialization_is_byte_for_byte_repeatable():
    module = diagnose_module()
    frame = pd.DataFrame({"fold": ["r1", "r2"], "value": [1, 2]})

    first = module.serialize_csv(frame, ["fold", "value"])
    second = module.serialize_csv(frame.copy(), ["fold", "value"])

    assert first == second
    assert first.endswith("\n")

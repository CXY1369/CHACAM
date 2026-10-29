from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BENCHMARK_PATH = Path(__file__).parents[1] / "examples" / "c_elegans" / "pseudo_ambiguity.py"


def benchmark_module():
    if not BENCHMARK_PATH.exists():
        raise AssertionError(f"benchmark has not been implemented: {BENCHMARK_PATH}")
    spec = importlib.util.spec_from_file_location("pseudo_ambiguity", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signal_inputs():
    cells = pd.DataFrame(
        {
            "sample_id": ["e1", "e1", "e1", "e1", "e1", "e2", "e2", "e2", "e2", "e2"],
            "label": ["Ea", "Ep", "R", "S", "T", "Ea", "Ep", "R", "S", "T"],
        },
        index=["e1_x", "e1_y", "e1_r", "e1_s", "e1_t", "e2_x", "e2_y", "e2_r", "e2_s", "e2_t"],
    )
    expression = pd.DataFrame(
        {
            "L": [0, 0, 10, 0, 0, 0, 0, 10, 0, 0],
            "Rec": [10, 0, 0, 10, 0, 10, 0, 0, 10, 0],
        },
        index=cells.index,
        dtype=float,
    )
    lr_pairs = pd.DataFrame({"ligand": ["L"], "receptor": ["Rec"], "weight": [1.0]})
    contacts = pd.DataFrame(
        {
            "identity_a": ["R", "R", "R", "R"],
            "identity_b": ["Ea", "Ep", "S", "T"],
            "contact": [1, 0, 1, 0],
        }
    )
    return cells, expression, lr_pairs, contacts


def decision_rows(module, cells=None, expression=None):
    base_cells, base_expression, lr_pairs, contacts = signal_inputs()
    cells = base_cells if cells is None else cells
    expression = base_expression if expression is None else expression
    return module.run_decision(
        dataset="synthetic",
        stage=15,
        pair=("Ea", "Ep"),
        fold="e1",
        cells=cells,
        expression=expression,
        lr_pairs=lr_pairs,
        contacts=contacts,
        group_names=(),
    )


def test_sister_pairs_use_only_last_character_and_fine_labels():
    module = benchmark_module()

    pairs = module.find_sister_pairs(
        ["Ea", "Ep", "ABala", "ABalp", "Eal", "Ear", "D", "P4", "Ca", "Cpa", "Eq"]
    )

    assert pairs == [("ABala", "ABalp"), ("Ea", "Ep"), ("Eal", "Ear")]


def test_reference_filter_requires_different_known_contacts_and_one_fold_cell():
    module = benchmark_module()
    cells = pd.DataFrame(
        {
            "sample_id": ["e1", "e1", "e1", "e1", "e1", "e1"],
            "label": ["Ea", "Ep", "Rgood", "Runknown", "Rsame", "Rtwo"],
        },
        index=["x", "y", "good", "unknown", "same", "two1"],
    )
    cells = pd.concat(
        [cells, pd.DataFrame({"sample_id": ["e1"], "label": ["Rtwo"]}, index=["two2"])]
    )
    contacts = pd.DataFrame(
        {
            "identity_a": ["Rgood", "Rgood", "Runknown", "Rsame", "Rsame", "Rtwo", "Rtwo"],
            "identity_b": ["Ea", "Ep", "Ea", "Ea", "Ep", "Ea", "Ep"],
            "contact": [1, 0, 1, 1, 1, 1, 0],
        }
    )

    assert module.eligible_references(("Ea", "Ep"), cells, contacts, ["Ea", "Ep", "Rgood", "Runknown", "Rsame", "Rtwo"]) == ("Rgood",)


def test_changing_held_out_truth_keeps_predictions_by_cell_but_changes_correctness():
    module = benchmark_module()
    cells, expression, _lr_pairs, _contacts = signal_inputs()
    original = decision_rows(module, cells, expression)
    swapped_cells = cells.copy()
    swapped_cells.loc[["e1_x", "e1_y"], "label"] = ["Ep", "Ea"]
    swapped = decision_rows(module, swapped_cells, expression)

    oracle_original = original[0]
    oracle_swapped = swapped[0]
    predictions_original = {
        oracle_original["cell_x"]: oracle_original["final_x"],
        oracle_original["cell_y"]: oracle_original["final_y"],
    }
    predictions_swapped = {
        oracle_swapped["cell_x"]: oracle_swapped["final_x"],
        oracle_swapped["cell_y"]: oracle_swapped["final_y"],
    }
    assert predictions_original == predictions_swapped
    assert oracle_original["correct"] != oracle_swapped["correct"]


def test_topk_uses_training_embryos_only():
    module = benchmark_module()
    cells, expression, lr_pairs, contacts = signal_inputs()
    changed = expression.copy()
    changed.loc[["e1_x", "e1_y", "e1_r", "e1_s", "e1_t"], :] = [[999, 0], [0, 999], [999, 999], [999, 0], [0, 999]]

    baseline = module.select_fold_topk(expression, cells, ("Ea", "Ep"), "e1", lr_pairs, contacts, ())
    altered = module.select_fold_topk(changed, cells, ("Ea", "Ep"), "e1", lr_pairs, contacts, ())
    pd.testing.assert_frame_equal(baseline.table, altered.table)


def test_merged_labels_and_group_names_are_used_for_training():
    module = benchmark_module()
    cells, _expression, _lr_pairs, _contacts = signal_inputs()

    merged, training, group_names = module.prepare_fold(cells, ("Ea", "Ep"), "e1", ("EQ",))

    assert set(merged.loc[["e1_x", "e1_y", "e2_x"], "label"]) == {"Ea+Ep"}
    assert set(training["label"]) == {"Ea+Ep", "R", "S", "T"}
    assert set(group_names) == {"EQ", "Ea+Ep"}


def test_separable_signal_is_correct_oracle_and_opposite_under_swap_control():
    module = benchmark_module()
    cells, expression, lr_pairs, contacts = signal_inputs()
    rows = []
    for fold in ("e1", "e2"):
        rows.extend(
            module.run_decision(
                dataset="synthetic",
                stage=15,
                pair=("Ea", "Ep"),
                fold=fold,
                cells=cells,
                expression=expression,
                lr_pairs=lr_pairs,
                contacts=contacts,
                group_names=(),
            )
        )

    oracle = [row for row in rows if row["mode"] == "oracle"]
    swap = [row for row in rows if row["mode"] == "swap_control"]
    assert [row["correct"] for row in oracle] == [1, 1]
    assert [row["correct"] for row in swap] == [0, 0]


def test_swap_control_reuses_the_oracle_topk(monkeypatch):
    module = benchmark_module()
    cells, expression, lr_pairs, contacts = signal_inputs()
    calls = []
    original = module.select_fold_topk

    def wrapped(*args, **kwargs):
        calls.append(args[5].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "select_fold_topk", wrapped)
    decision_rows(module, cells, expression)
    assert len(calls) == 1


def test_no_bucket_is_unresolved_and_excluded_from_denominators():
    module = benchmark_module()
    cells, expression, lr_pairs, contacts = signal_inputs()
    cells.loc["e1_y", "label"] = "Ea"
    rows = module.run_decision(
        dataset="synthetic",
        stage=15,
        pair=("Ea", "Ep"),
        fold="e1",
        cells=cells,
        expression=expression,
        lr_pairs=lr_pairs,
        contacts=contacts,
        group_names=(),
    )

    assert {row["status"] for row in rows} == {"no_bucket"}
    assert all(row["correct"] is None for row in rows)
    summary, _per_embryo = module.summarize_decisions(pd.DataFrame(rows))
    assert summary.loc[summary["mode"].eq("oracle"), "evaluable"].iat[0] == 0


def test_no_reference_with_empty_topk_short_circuits_to_no_reference():
    module = benchmark_module()
    cells, expression, _lr_pairs, _contacts = signal_inputs()
    empty_contacts = pd.DataFrame(columns=["identity_a", "identity_b", "contact"])
    empty_lr = pd.DataFrame(columns=["ligand", "receptor", "weight"])
    rows = module.run_decision(
        dataset="synthetic",
        stage=15,
        pair=("Ea", "Ep"),
        fold="e1",
        cells=cells,
        expression=expression,
        lr_pairs=empty_lr,
        contacts=empty_contacts,
        group_names=(),
    )
    assert len(rows) == 2
    assert {row["mode"] for row in rows} == {"oracle", "swap_control"}
    assert all(row["status"] == "unresolved" for row in rows)
    assert all(row["unresolved_reason"] == "unresolved_no_reference" for row in rows)
    assert all(row["n_lr"] == 0 for row in rows)
    assert all(row["references"] == "" for row in rows)
    assert all(row["final_x"] == row["group"] and row["final_y"] == row["group"] for row in rows)
    assert all(row["correct"] is None for row in rows)


def test_no_reference_with_nonempty_topk_does_not_triangulate(monkeypatch):
    module = benchmark_module()
    cells, expression, lr_pairs, contacts = signal_inputs()
    contacts = contacts.loc[~contacts["identity_b"].isin(["Ea", "Ep"])].reset_index(drop=True)
    calls = []
    original_select = module.select_fold_topk

    def wrapped_select(*args, **kwargs):
        calls.append(1)
        return original_select(*args, **kwargs)

    def fail_triangulate(*args, **kwargs):
        raise AssertionError("triangulate must not be called without references")

    monkeypatch.setattr(module, "select_fold_topk", wrapped_select)
    monkeypatch.setattr(module, "triangulate", fail_triangulate)
    rows = module.run_decision(
        dataset="synthetic",
        stage=15,
        pair=("Ea", "Ep"),
        fold="e1",
        cells=cells,
        expression=expression,
        lr_pairs=lr_pairs,
        contacts=contacts,
        group_names=(),
    )
    assert len(calls) == 1
    assert len(rows) == 2
    assert all(row["unresolved_reason"] == "unresolved_no_reference" for row in rows)
    assert all(row["n_lr"] == 1 for row in rows)
    assert all(row["final_x"] == row["group"] and row["final_y"] == row["group"] for row in rows)


def test_reference_with_empty_topk_remains_no_lr():
    module = benchmark_module()
    cells, expression, _lr_pairs, contacts = signal_inputs()
    empty_lr = pd.DataFrame(columns=["ligand", "receptor", "weight"])
    rows = module.run_decision(
        dataset="synthetic",
        stage=15,
        pair=("Ea", "Ep"),
        fold="e1",
        cells=cells,
        expression=expression,
        lr_pairs=empty_lr,
        contacts=contacts,
        group_names=(),
    )
    assert all(row["status"] == "unresolved" for row in rows)
    assert all(row["unresolved_reason"] == "unresolved_no_lr" for row in rows)


def test_decision_binomial_tail_matches_six_of_eight():
    module = benchmark_module()
    rows = []
    for number in range(8):
        rows.append(
            {
                "dataset": "synthetic",
                "stage": 15,
                "pair": "Ea+Ep",
                "group": "Ea+Ep",
                "fold": f"e{number}",
                "mode": "oracle",
                "cell_x": "x",
                "cell_y": "y",
                "references": "R",
                "n_lr": 1,
                "status": "resolved",
                "unresolved_reason": "",
                "final_x": "Ea" if number < 6 else "Ep",
                "final_y": "Ep" if number < 6 else "Ea",
                "correct": int(number < 6),
            }
        )
    summary, _per_embryo = module.summarize_decisions(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["correct"] == 6
    assert row["descriptive_binomial_p"] == pytest.approx(37 / 256)


def test_embryo_sign_test_uses_wins_losses_and_excludes_ties():
    module = benchmark_module()
    rows = []
    for fold, correct_values in [("win", [1] * 10), ("loss1", [0]), ("loss2", [0]), ("loss3", [0]), ("loss4", [0])]:
        for number, correct in enumerate(correct_values):
            rows.append(
                {
                    "dataset": "synthetic",
                    "stage": 15,
                    "pair": "Ea+Ep",
                    "group": "Ea+Ep",
                    "fold": fold,
                    "mode": "oracle",
                    "cell_x": "x",
                    "cell_y": "y",
                    "references": "R",
                    "n_lr": 1,
                    "status": "resolved",
                    "unresolved_reason": "",
                    "final_x": "Ea" if correct else "Ep",
                    "final_y": "Ep" if correct else "Ea",
                    "correct": correct,
                }
            )
    summary, per_embryo = module.summarize_decisions(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert (row["embryo_wins"], row["embryo_losses"], row["embryo_ties"], row["embryo_no_resolved"]) == (1, 4, 0, 0)
    assert row["sign_test_p"] == pytest.approx(31 / 32)
    assert set(per_embryo["outcome"]) == {"win", "loss"}

    two_wins_one_tie = pd.DataFrame(
        [
            dict(rows[0], fold="w1", correct=1),
            dict(rows[0], fold="w2", correct=1),
            dict(rows[0], fold="tie", correct=1),
            dict(rows[0], fold="tie", correct=0),
        ]
    )
    two_wins_summary, _ = module.summarize_decisions(two_wins_one_tie)
    assert two_wins_summary.iloc[0]["sign_test_p"] == pytest.approx(1 / 4)

    tie = pd.DataFrame(rows[:2]).copy()
    tie.loc[0, "fold"] = "tie"
    tie.loc[1, "fold"] = "tie"
    tie.loc[0, "correct"] = 1
    tie.loc[1, "correct"] = 0
    tie_summary, tie_per_embryo = module.summarize_decisions(tie)
    assert tie_summary.iloc[0]["embryo_ties"] == 1
    assert tie_per_embryo.iloc[0]["a_e"] == pytest.approx(0.5)


def test_swap_control_mismatch_raises_before_results_are_written(monkeypatch):
    module = benchmark_module()
    oracle = {"status": "resolved", "unresolved_reason": "", "correct": 1}
    bad_swap = {"status": "unresolved", "unresolved_reason": "unresolved_tie", "correct": None}
    with pytest.raises(RuntimeError, match="swap_control"):
        module.validate_swap_consistency(oracle, bad_swap)

    cells, expression, lr_pairs, contacts = signal_inputs()
    called = []
    original = module.validate_swap_consistency

    def wrapped(oracle_row, swap_row):
        called.append((oracle_row, swap_row))
        return original(oracle_row, swap_row)

    monkeypatch.setattr(module, "validate_swap_consistency", wrapped)
    decision_rows(module, cells, expression)
    assert len(called) == 1

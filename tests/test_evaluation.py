import pandas as pd
import pytest

from chacam.schema import AmbiguityGroup
from chacam.evaluation import evaluate


AX = AmbiguityGroup(name="Ax", candidates=("A1", "A2"), references=("R1", "R2"))
BX = AmbiguityGroup(name="Bx", candidates=("B1", "B2"), references=("R3", "R4"))
UNRESOLVED = (
    "unresolved_bucket_size",
    "unresolved_no_lr",
    "unresolved_no_reference",
    "unresolved_tie",
    "unresolved_reference_conflict",
)


def _metric(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _hand_example():
    predictions = pd.DataFrame(
        [
            ["s1_wrong", "s1", "s1", "A2", "A2", "assigned", ""],
            ["s1_ok", "s1", "s1", "R1", "R1", "assigned", ""],
            ["s1_un", "s1", "s1", "unassigned", "unassigned", "unassigned", ""],
            ["s1_res_ok", "s1", "s1", "Ax", "A1", "resolved", ""],
            ["s1_res_bad", "s1", "s1", "Ax", "A2", "resolved", ""],
            ["s1_conf1", "s1", "s1", "Bx", "Bx", "unresolved", "unresolved_reference_conflict"],
            ["s1_conf2", "s1", "s1", "Bx", "Bx", "unresolved", "unresolved_reference_conflict"],
            ["s1_r1", "s1", "s1", "R1", "R1", "assigned", ""],
            ["s1_r2", "s1", "s1", "R2", "R2", "assigned", ""],
            ["s1_r3", "s1", "s1", "R3", "R3", "assigned", ""],
            ["s1_r4", "s1", "s1", "R4", "R4", "assigned", ""],
            ["s2_a", "s2", "s2", "A1", "A1", "assigned", ""],
            ["s2_b", "s2", "s2", "A1", "A1", "assigned", ""],
            ["s2_c", "s2", "s2", "A1", "A1", "assigned", ""],
        ],
        columns=["cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason"],
    )
    cells = pd.DataFrame(
        {
            "label": [
                "A1", "R1", "R2", "Ax", "Ax", "Bx", "Bx", "R1", "R2", "R3", "R4", "A1", "A1", "A1"
            ]
        },
        index=predictions["cell_id"],
    )
    truth = pd.DataFrame(
        {
            "true_label": [
                "A1", "R1", "R2", "A1", "A1", "B1", "B2", "R1", "R2", "R3", "R4", "A1", "A1", "A1"
            ]
        },
        index=predictions["cell_id"],
    )
    return predictions, cells, truth


def test_evaluate_hand_example_matches_fold_and_pooled_counts():
    predictions, cells, truth = _hand_example()

    result = evaluate(predictions, cells, [AX, BX], truth)

    assert result["folds"]["s1"]["stage1_accuracy"] == _metric(9, 11)
    assert result["folds"]["s1"]["resolution_rate"] == _metric(2, 4)
    assert result["folds"]["s1"]["reference_agreement"] == _metric(1, 2)
    assert result["folds"]["s1"]["fine_accuracy"] == _metric(1, 2)
    assert result["folds"]["s1"]["named_accuracy"] == _metric(6, 8)
    assert result["folds"]["s2"]["stage1_accuracy"] == _metric(3, 3)
    assert result["folds"]["s2"]["resolution_rate"] == _metric(0, 0)
    assert result["folds"]["s2"]["reference_agreement"] == _metric(0, 0)
    assert result["folds"]["s2"]["fine_accuracy"] == _metric(0, 0)
    assert result["folds"]["s2"]["named_accuracy"] == _metric(3, 3)
    assert result["pooled"]["stage1_accuracy"] == _metric(12, 14)
    assert result["pooled"]["resolution_rate"] == _metric(2, 4)
    assert result["pooled"]["reference_agreement"] == _metric(1, 2)
    assert result["pooled"]["fine_accuracy"] == _metric(1, 2)
    assert result["pooled"]["named_accuracy"] == _metric(9, 11)
    assert result["pooled"]["stage1_accuracy"]["value"] != (
        result["folds"]["s1"]["stage1_accuracy"]["value"] + result["folds"]["s2"]["stage1_accuracy"]["value"]
    ) / 2


def test_evaluate_reference_agreement_denominator_uses_positive_conditions():
    rows = []

    def add(fold, cells):
        for cell_id, stage1, status, reason, extras in cells:
            rows.append(
                [f"{fold}_{cell_id}", fold, fold, stage1, stage1 if status != "resolved" else "A1", status, reason]
            )
            for extra_id, extra_label in extras:
                rows.append([f"{fold}_{extra_id}", fold, fold, extra_label, extra_label, "assigned", ""])

    add("f_ok", [("u", "Ax", "resolved", "", [("r1", "R1"), ("r2", "R2")]), ("v", "Ax", "resolved", "", [])])
    add("f_nolr", [("u", "Ax", "unresolved", "unresolved_no_lr", [("r1", "R1"), ("r2", "R2")]), ("v", "Ax", "unresolved", "unresolved_no_lr", [])])
    add("f_size1", [("u", "Ax", "unresolved", "unresolved_bucket_size", [("r1", "R1"), ("r2", "R2")])])
    add(
        "f_size3",
        [
            ("u", "Ax", "unresolved", "unresolved_bucket_size", [("r1", "R1"), ("r2", "R2")]),
            ("v", "Ax", "unresolved", "unresolved_bucket_size", []),
            ("w", "Ax", "unresolved", "unresolved_bucket_size", []),
        ],
    )
    add("f_none", [("r1", "R1", "assigned", "", [("r2", "R2")])])
    add("f_oneref", [("u", "Ax", "resolved", "", [("r1", "R1")]), ("v", "Ax", "resolved", "", [])])

    predictions = pd.DataFrame(
        rows,
        columns=["cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason"],
    )
    cells = pd.DataFrame({"label": predictions["stage1_label"].to_list()}, index=predictions["cell_id"])
    result = evaluate(predictions, cells, [AX], None)

    assert result["folds"]["f_ok"]["reference_agreement"] == _metric(1, 1)
    assert result["folds"]["f_nolr"]["reference_agreement"] == _metric(0, 0)
    assert result["folds"]["f_size1"]["reference_agreement"] == _metric(0, 0)
    assert result["folds"]["f_size3"]["reference_agreement"] == _metric(0, 0)
    assert result["folds"]["f_none"]["reference_agreement"] == _metric(0, 0)
    assert result["folds"]["f_oneref"]["reference_agreement"] == _metric(0, 0)
    assert result["pooled"]["reference_agreement"] == _metric(1, 1)


def test_evaluate_unresolved_reasons_list_all_codes_and_pool_by_sum():
    predictions = pd.DataFrame(
        [
            ["a", "s1", "s1", "Ax", "Ax", "unresolved", "unresolved_bucket_size"],
            ["b", "s1", "s1", "Ax", "Ax", "unresolved", "unresolved_no_lr"],
            ["c", "s2", "s2", "Ax", "Ax", "unresolved", "unresolved_tie"],
            ["d", "s2", "s2", "R1", "R1", "assigned", ""],
        ],
        columns=["cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason"],
    )
    cells = pd.DataFrame({"label": ["Ax", "Ax", "Ax", "R1"]}, index=predictions["cell_id"])

    result = evaluate(predictions, cells, [AX], None)

    assert set(result["pooled"]["unresolved_reasons"]) == set(UNRESOLVED)
    assert result["folds"]["s1"]["unresolved_reasons"]["unresolved_bucket_size"] == 1
    assert result["folds"]["s1"]["unresolved_reasons"]["unresolved_no_lr"] == 1
    assert result["folds"]["s1"]["unresolved_reasons"]["unresolved_tie"] == 0
    assert result["folds"]["s2"]["unresolved_reasons"]["unresolved_tie"] == 1
    for code in UNRESOLVED:
        assert result["pooled"]["unresolved_reasons"][code] == (
            result["folds"]["s1"]["unresolved_reasons"][code] + result["folds"]["s2"]["unresolved_reasons"][code]
        )


def test_evaluate_zero_denominator_is_null_and_missing_truth_nulls_truth_metrics():
    predictions = pd.DataFrame(
        [["c1", "s1", "s1", "R1", "R1", "assigned", ""]],
        columns=["cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason"],
    )
    cells = pd.DataFrame({"label": ["R1"]}, index=["c1"])

    result = evaluate(predictions, cells, [AX], None)

    assert result["truth_provided"] is False
    for name in ("fine_accuracy", "named_accuracy"):
        assert result["pooled"][name] == {"numerator": None, "denominator": None, "value": None}
        assert result["folds"]["s1"][name] == {"numerator": None, "denominator": None, "value": None}
    assert result["pooled"]["resolution_rate"]["value"] is None
    assert result["pooled"]["reference_agreement"]["value"] is None


def test_evaluate_is_aligned_by_cell_id_not_row_order():
    predictions, cells, truth = _hand_example()
    shuffled_predictions = predictions.sample(frac=1, random_state=1).reset_index(drop=True)
    shuffled_cells = cells.sample(frac=1, random_state=2)
    shuffled_truth = truth.sample(frac=1, random_state=3)

    original = evaluate(predictions, cells, [AX, BX], truth)
    shuffled = evaluate(shuffled_predictions, shuffled_cells, [AX, BX], shuffled_truth)

    assert original == shuffled


def test_evaluate_rejects_truth_that_misses_predicted_cells():
    predictions, cells, truth = _hand_example()
    truth = truth.drop(index=["s1_wrong", "s2_c"])

    with pytest.raises(ValueError, match="s1_wrong"):
        evaluate(predictions, cells, [AX, BX], truth)

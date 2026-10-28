import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from chacam.pipeline import fit_fold, run
from chacam.cci import train_topk
from chacam.reference import training_mean
from chacam.schema import load_config, load_inputs
from chacam.synthetic import make_synthetic, write_synthetic


ROOT = Path(__file__).parents[1]


def _write_config(
    tmp_path: Path,
    *,
    reference_mode="training_mean",
    n_samples=4,
    truth="data/truth.csv",
    lr_selection="given",
    contact_signal=True,
    seed=0,
    output_dir="runs/output",
    data=None,
):
    if data is None:
        data = make_synthetic(seed=seed, n_samples=n_samples, contact_signal=contact_signal)
        write_synthetic(data, tmp_path / "data")
    config = {
        "provenance": "framework_baseline",
        "seed": seed,
        "inputs": {
            "expression": "data/expression.csv",
            "cells": "data/cells.csv",
            "features": "data/features.csv",
            "lr_pairs": "data/lr_pairs.csv",
            "contacts": "data/contacts.csv",
            "expression_scale": "counts",
        },
        "reference": {"mode": reference_mode},
        "attention": {"normalization": "l1", "temperature": 1.0},
        "assignment": {"mode": "unique"},
        "cci": {"transform": "log1p", "direction": "both", "lr_selection": lr_selection, "top_k": 50},
        "ambiguity_groups": [{"name": "Ax", "candidates": ["A1", "A2"], "references": ["R1", "R2"]}],
        "output_dir": output_dir,
    }
    if truth is not None:
        config["inputs"]["truth"] = truth
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data, config_path


def _output_files(output_dir: Path):
    return {
        path.relative_to(output_dir): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def test_demo_runs_and_writes_all_m2_outputs_without_metrics(tmp_path):
    script = ROOT / "examples" / "synthetic" / "run_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir = tmp_path / "runs" / "synthetic_demo"

    assert "status:" in completed.stdout
    assert "unresolved_reason:" in completed.stdout
    assert output_dir.joinpath("predictions.csv").exists()
    assert output_dir.joinpath("assignment_trace.csv").exists()
    assert output_dir.joinpath("cci_evidence.csv").exists()
    assert output_dir.joinpath("manifest.json").exists()
    assert output_dir.joinpath("attention", "sample_001.npz").exists()
    assert output_dir.joinpath("topk", "sample_001.csv").exists()
    assert output_dir.joinpath("metrics.json").exists()
    metrics = json.loads(output_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert "pooled" in metrics
    for name in ("stage1_accuracy", "resolution_rate", "reference_agreement", "fine_accuracy", "named_accuracy"):
        assert name in completed.stdout
        assert name in metrics["pooled"]


def test_pipeline_output_format_manifest_and_npz_are_readable(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=3)

    result = run(config_path)
    predictions = pd.read_csv(result.output_dir / "predictions.csv")
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    topk = pd.read_csv(result.output_dir / "topk" / "sample_001.csv")
    attention = np.load(result.output_dir / "attention" / "sample_001.npz", allow_pickle=False)

    assert result.output_dir == tmp_path / "runs" / "output"
    assert list(predictions.columns) == [
        "cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason", "provenance"
    ]
    assert len(predictions) == len(data.cells)
    assert predictions["cell_id"].is_unique
    assert predictions["cell_id"].tolist() == data.cells["cell_id"].tolist()
    assert (predictions["fold"] == predictions["sample_id"]).all()
    assert set(topk["selection"]) == {"given"}
    assert set(attention.files) == {"S", "P_col", "P_row", "L", "rows", "cols"}
    assert attention["rows"].dtype.kind in "US"
    assert attention["cols"].dtype.kind in "US"
    assert manifest["provenance"] == "framework_baseline"
    assert manifest["versions"]["chacam"]
    assert set(manifest["versions"]) == {"chacam", "python", "numpy", "pandas"}
    assert manifest["input_dimensions"] == {"cells": len(data.cells), "genes": len(data.expression.columns) - 1, "samples": 3}
    assert "used_feature_genes" in manifest
    assert "missing_feature_genes" in manifest
    assert "dropped_lr_pairs" in manifest
    assert manifest["seed"] == 0
    assert manifest["lr_selection"] == "given"
    assert set(manifest["folds"]) == {"sample_001", "sample_002", "sample_003"}


def test_given_mode_manifest_has_null_pair_counts_and_selected_count(tmp_path):
    _data, config_path = _write_config(tmp_path, n_samples=3, lr_selection="given")

    result = run(config_path)
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    selected_count = len(pd.read_csv(result.output_dir / "topk" / "sample_001.csv"))

    for fold in manifest["folds"].values():
        assert fold["n_selected_lr"] == selected_count
        assert fold["n_contact_pairs"] is None
        assert fold["n_noncontact_pairs"] is None


def test_fit_fold_integer_sample_id_matches_string_and_excludes_leaveout_cells(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=4, lr_selection="train_topk")
    data.cells["sample_id"] = data.cells["sample_id"].str.extract(r"(\d+)")[0].astype(int)
    data.cells.to_csv(tmp_path / "data" / "cells.csv", index=False)

    config = load_config(config_path)
    inputs = load_inputs(config)
    fit_string = fit_fold(inputs, config, "1")
    fit_integer = fit_fold(inputs, config, 1)
    all_cells = train_topk(
        inputs.expression,
        inputs.cells,
        inputs.lr_pairs,
        inputs.contacts,
        [group.name for group in config.ambiguity_groups],
        config.cci.transform,
        config.cci.top_k,
    )

    assert fit_integer.profiles.equals(fit_string.profiles)
    assert fit_integer.lr_table.equals(fit_string.lr_table)
    assert fit_integer.n_contact_pairs == fit_string.n_contact_pairs
    assert fit_integer.n_noncontact_pairs == fit_string.n_noncontact_pairs
    assert fit_integer.missing_labels == fit_string.missing_labels
    assert fit_integer.n_contact_pairs < all_cells.n_contact_pairs


def test_fit_fold_rejects_missing_integer_sample_id(tmp_path):
    _data, config_path = _write_config(tmp_path, n_samples=4)
    config = load_config(config_path)
    inputs = load_inputs(config)

    with pytest.raises(ValueError, match="99"):
        fit_fold(inputs, config, 99)


def test_attention_npz_rows_and_cols_match_fold_input_and_reference_order(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=3)
    run(config_path)
    config = load_config(config_path)
    inputs = load_inputs(config)
    fold = "sample_001"
    target_ids = data.cells.loc[data.cells["sample_id"] == fold, "cell_id"].tolist()
    training_ids = data.cells.loc[data.cells["sample_id"] != fold, "cell_id"].tolist()
    profiles = training_mean(
        inputs.expression.loc[training_ids],
        inputs.cells.loc[training_ids, "label"],
        inputs.used_feature_genes,
        expected_labels=list(dict.fromkeys(inputs.cells["label"].tolist())),
    )
    saved = np.load(tmp_path / "runs" / "output" / "attention" / f"{fold}.npz", allow_pickle=False)

    assert saved["rows"].tolist() == target_ids
    assert saved["cols"].tolist() == profiles.index.astype(str).tolist()


@pytest.mark.parametrize("lr_selection", ["given", "train_topk"])
def test_pipeline_predictions_do_not_depend_on_truth_content(tmp_path, lr_selection):
    data = make_synthetic(seed=0, n_samples=3)
    write_synthetic(data, tmp_path / "data")
    shuffled = data.truth.copy()
    shuffled["true_label"] = shuffled["true_label"].iloc[::-1].to_list()
    shuffled.to_csv(tmp_path / "data" / "truth_shuffled.csv", index=False)

    outputs = []
    for truth in (None, "data/truth.csv", "data/truth_shuffled.csv"):
        _, config_path = _write_config(
            tmp_path, n_samples=3, truth=truth, data=data, lr_selection=lr_selection
        )
        result = run(config_path)
        outputs.append(_output_files(result.output_dir))

    prediction_keys = [
        "predictions.csv",
        "assignment_trace.csv",
        "cci_evidence.csv",
    ]
    for left, right in ((outputs[0], outputs[1]), (outputs[1], outputs[2])):
        for name in prediction_keys:
            assert left[Path(name)] == right[Path(name)]
        for name in left:
            if name.parts[0] in {"attention", "topk"}:
                assert left[name] == right[name]

    def _manifest_without_truth(payload):
        manifest = json.loads(payload)
        manifest["config"].pop("config_path", None)
        manifest["config"].get("inputs", {}).pop("truth", None)
        return manifest

    assert _manifest_without_truth(outputs[0][Path("manifest.json")]) == _manifest_without_truth(
        outputs[1][Path("manifest.json")]
    )
    assert _manifest_without_truth(outputs[1][Path("manifest.json")]) == _manifest_without_truth(
        outputs[2][Path("manifest.json")]
    )

    metrics = [json.loads(output[Path("metrics.json")]) for output in outputs]
    assert metrics[0]["truth_provided"] is False
    assert metrics[1]["truth_provided"] is True
    assert metrics[2]["truth_provided"] is True
    for name in ("fine_accuracy", "named_accuracy"):
        assert metrics[0]["pooled"][name] == {"numerator": None, "denominator": None, "value": None}
        assert metrics[1]["pooled"][name] != metrics[0]["pooled"][name]
        assert metrics[1]["pooled"][name] != metrics[2]["pooled"][name]


@pytest.mark.parametrize("reference_mode", ["training_mean", "binary_marker"])
def test_pipeline_records_same_leave_one_missing_labels_for_both_reference_modes(tmp_path, reference_mode):
    data, config_path = _write_config(tmp_path, reference_mode=reference_mode, n_samples=3)
    mask = data.cells["cell_id"].eq("sample_001_B6")
    data.cells.loc[mask, "label"] = "OnlyInFirstSample"
    data.cells.to_csv(tmp_path / "data" / "cells.csv", index=False)

    result = run(config_path)
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["folds"]) == 3
    assert manifest["folds"]["sample_001"]["missing_labels"] == ["OnlyInFirstSample"]
    assert manifest["folds"]["sample_002"]["missing_labels"] == []
    assert manifest["folds"]["sample_003"]["missing_labels"] == []


def test_pipeline_cleans_only_framework_outputs_and_old_fold_files(tmp_path):
    _, config_path = _write_config(tmp_path, n_samples=3)
    first = run(config_path)
    (first.output_dir / "attention" / "old.npz").write_bytes(b"old")
    (first.output_dir / "topk" / "old.csv").write_text("old", encoding="utf-8")
    (first.output_dir / "metrics.json").write_text("old", encoding="utf-8")
    (first.output_dir / "notes.txt").write_text("keep", encoding="utf-8")

    run(config_path)

    assert not (first.output_dir / "attention" / "old.npz").exists()
    assert not (first.output_dir / "topk" / "old.csv").exists()
    metrics_text = (first.output_dir / "metrics.json").read_text(encoding="utf-8")
    assert metrics_text != "old"
    json.loads(metrics_text)
    assert (first.output_dir / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_pipeline_validates_lr_columns_before_cleaning_previous_outputs(tmp_path):
    _, config_path = _write_config(tmp_path, n_samples=3)
    first = run(config_path)
    predictions_path = first.output_dir / "predictions.csv"
    lr_path = tmp_path / "data" / "lr_pairs.csv"
    lr_pairs = pd.read_csv(lr_path)
    lr_pairs["sample_id"] = "conflict"
    lr_pairs.to_csv(lr_path, index=False)

    with pytest.raises(ValueError, match="reserved.*sample_id"):
        run(config_path)

    assert predictions_path.exists()


def test_pipeline_is_byte_reproducible_for_same_configuration(tmp_path):
    _, config_path = _write_config(tmp_path, n_samples=3)
    result = run(config_path)
    first = _output_files(result.output_dir)

    run(config_path)
    second = _output_files(result.output_dir)

    assert first == second


def test_pipeline_contact_signal_wiring_meets_m2_thresholds(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=10)
    result = run(config_path)
    truth = data.truth.set_index("cell_id")["true_label"]
    predictions = result.predictions.set_index("cell_id")
    ambiguous = predictions["stage1_label"].eq("Ax")
    resolved = ambiguous & predictions["status"].eq("resolved")
    named = predictions["status"].eq("assigned") | predictions["status"].eq("resolved")

    resolution_rate = resolved[ambiguous].mean()
    named_accuracy = (predictions.loc[named, "final_label"] == truth.loc[named.index[named]]).mean()

    assert resolution_rate >= 0.6
    assert named_accuracy >= 0.9


def test_pipeline_rejects_sample_ids_containing_slashes(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=3)
    data.cells.loc[data.cells["sample_id"] == "sample_001", "sample_id"] = "bad/sample"
    data.cells.to_csv(tmp_path / "data" / "cells.csv", index=False)

    with pytest.raises(ValueError, match="slash"):
        run(config_path)


def test_load_config_rejects_missing_truth_path(tmp_path):
    _, config_path = _write_config(tmp_path, n_samples=3, truth="data/does-not-exist.csv")

    with pytest.raises(ValueError, match="truth"):
        load_config(config_path)


def test_run_rejects_truth_that_does_not_cover_all_cells(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=3)
    data.truth.iloc[1:].to_csv(tmp_path / "data" / "truth.csv", index=False)

    with pytest.raises(ValueError, match="cell"):
        run(config_path)


def test_truth_error_leaves_previous_outputs_unchanged(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=3)
    first = run(config_path)
    snapshot = _output_files(first.output_dir)
    data.truth.iloc[1:].to_csv(tmp_path / "data" / "truth.csv", index=False)

    with pytest.raises(ValueError):
        run(config_path)

    assert _output_files(first.output_dir) == snapshot


@pytest.mark.parametrize("reference_mode", ["training_mean", "binary_marker"])
def test_fit_fold_ignores_heldout_expression_and_labels(tmp_path, reference_mode):
    _data, config_path = _write_config(tmp_path, n_samples=4, lr_selection="train_topk", reference_mode=reference_mode)
    config = load_config(config_path)
    inputs = load_inputs(config)
    sample_id = "sample_001"
    before = fit_fold(inputs, config, sample_id)

    rng = np.random.default_rng(1)
    heldout = inputs.cells["sample_id"].astype(str) == sample_id
    inputs.expression = inputs.expression.copy()
    inputs.expression.loc[heldout] = inputs.expression.loc[heldout] * rng.uniform(1.5, 3.0, size=inputs.expression.loc[heldout].shape)
    heldout_labels = inputs.cells.loc[heldout, "label"].astype(str).to_numpy()
    rng.shuffle(heldout_labels)
    inputs.cells = inputs.cells.copy()
    inputs.cells.loc[heldout, "label"] = heldout_labels
    after = fit_fold(inputs, config, sample_id)

    assert after.profiles.equals(before.profiles)
    assert after.lr_table.equals(before.lr_table)
    assert after.n_contact_pairs == before.n_contact_pairs
    assert after.n_noncontact_pairs == before.n_noncontact_pairs


def test_fit_fold_changes_when_training_expression_changes(tmp_path):
    _data, config_path = _write_config(tmp_path, n_samples=4, lr_selection="train_topk")
    config = load_config(config_path)
    inputs = load_inputs(config)
    before = fit_fold(inputs, config, "sample_001")

    train = inputs.cells["sample_id"].astype(str) == "sample_002"
    inputs.expression = inputs.expression.copy()
    inputs.expression.loc[train] = inputs.expression.loc[train] * 4.0
    after = fit_fold(inputs, config, "sample_001")

    assert not after.lr_table["effect"].equals(before.lr_table["effect"])


def test_fit_fold_rejects_missing_sample_id(tmp_path):
    _data, config_path = _write_config(tmp_path, n_samples=3)
    config = load_config(config_path)
    inputs = load_inputs(config)

    with pytest.raises(ValueError, match="no_such_sample"):
        fit_fold(inputs, config, "no_such_sample")


def test_demo_loso_writes_all_outputs_and_topk_is_train_selected(tmp_path):
    script = ROOT / "examples" / "synthetic" / "run_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "config_loso.yaml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir = tmp_path / "runs" / "synthetic_loso"
    assert completed.returncode == 0
    assert output_dir.joinpath("metrics.json").exists()
    assert output_dir.joinpath("predictions.csv").exists()
    topk = pd.read_csv(output_dir / "topk" / "sample_001.csv")
    assert set(topk["selection"]) == {"train_topk"}
    assert (topk["effect"] > 0).all()
    assert topk["effect"].is_monotonic_decreasing
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for fold in manifest["folds"].values():
        assert "n_selected_lr" in fold
        assert "n_contact_pairs" in fold
        assert "n_noncontact_pairs" in fold


def test_loso_configuration_is_byte_reproducible(tmp_path):
    _data, config_path = _write_config(tmp_path, n_samples=3, lr_selection="train_topk")
    result = run(config_path)
    first = _output_files(result.output_dir)
    run(config_path)
    assert first == _output_files(result.output_dir)


def test_given_mode_named_accuracy_and_resolution_match_m2_wiring(tmp_path):
    data, config_path = _write_config(tmp_path, n_samples=10)
    result = run(config_path)
    truth = data.truth.set_index("cell_id")["true_label"]
    predictions = result.predictions.set_index("cell_id")
    ambiguous = predictions["stage1_label"].eq("Ax")
    resolved = ambiguous & predictions["status"].eq("resolved")
    named = predictions["status"].eq("assigned") | predictions["status"].eq("resolved")
    resolution_rate = float(resolved[ambiguous].mean())
    named_accuracy = float((predictions.loc[named, "final_label"] == truth.loc[named.index[named]]).mean())
    assert resolution_rate == 0.9
    assert named_accuracy == 1.0


def _control_pooled(tmp_path, seed, contact_signal):
    work = tmp_path / f"seed_{seed}_{int(contact_signal)}"
    work.mkdir()
    data, config_path = _write_config(
        work,
        n_samples=20,
        seed=seed,
        contact_signal=contact_signal,
        lr_selection="train_topk",
        truth="data/truth.csv",
    )
    result = run(config_path)
    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    return data, metrics["pooled"]


def test_positive_and_negative_controls_meet_frozen_thresholds(tmp_path):
    positive = []
    negative = []
    for seed in range(5):
        _data, pooled = _control_pooled(tmp_path, seed, True)
        positive.append(pooled)
        _data, pooled = _control_pooled(tmp_path, seed, False)
        negative.append(pooled)

    def _combine(items, name):
        numerator = sum(item[name]["numerator"] for item in items)
        denominator = sum(item[name]["denominator"] for item in items)
        return numerator / denominator, numerator, denominator

    pos_fine, _pos_fine_n, _pos_fine_d = _combine(positive, "fine_accuracy")
    pos_res, _pos_res_n, _pos_res_d = _combine(positive, "resolution_rate")
    neg_fine, _neg_fine_n, neg_fine_d = _combine(negative, "fine_accuracy")
    neg_res, _neg_res_n, _neg_res_d = _combine(negative, "resolution_rate")

    assert pos_fine >= 0.9
    assert pos_res >= 0.8
    assert 0.35 <= neg_fine <= 0.65
    assert neg_fine_d >= 40
    assert (pos_res - neg_res) >= 0.2

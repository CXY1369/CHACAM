import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from chacam.pipeline import run
from chacam.reference import training_mean
from chacam.schema import load_config, load_inputs
from chacam.synthetic import make_synthetic, write_synthetic


ROOT = Path(__file__).parents[1]


def _write_config(tmp_path: Path, *, reference_mode="training_mean", n_samples=4, truth="data/truth.csv"):
    data = make_synthetic(seed=0, n_samples=n_samples)
    write_synthetic(data, tmp_path / "data")
    config = {
        "provenance": "framework_baseline",
        "seed": 0,
        "inputs": {
            "expression": "data/expression.csv",
            "cells": "data/cells.csv",
            "features": "data/features.csv",
            "lr_pairs": "data/lr_pairs.csv",
            "contacts": "data/contacts.csv",
            "truth": truth,
            "expression_scale": "counts",
        },
        "reference": {"mode": reference_mode},
        "attention": {"normalization": "l1", "temperature": 1.0},
        "assignment": {"mode": "unique"},
        "cci": {"transform": "log1p", "direction": "both", "lr_selection": "given", "top_k": 50},
        "ambiguity_groups": [{"name": "Ax", "candidates": ["A1", "A2"], "references": ["R1", "R2"]}],
        "output_dir": "runs/output",
    }
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
    assert not output_dir.joinpath("metrics.json").exists()


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


def test_pipeline_does_not_read_truth_and_source_has_no_truth_loader_reference(tmp_path):
    _, config_path = _write_config(tmp_path, n_samples=3, truth="data/does-not-exist.csv")

    result = run(config_path)
    source = (ROOT / "src" / "chacam" / "pipeline.py").read_text(encoding="utf-8")

    assert result.predictions.shape[0] > 0
    assert "load_truth" not in source
    assert "inputs.truth" not in source


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
    assert not (first.output_dir / "metrics.json").exists()
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

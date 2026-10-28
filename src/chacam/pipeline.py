"""End-to-end M2 leave-one-sample pipeline and deterministic output writer."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment import capacities, greedy_assign
from .attention import attention
from .cci import select_lr
from .reference import binary_marker, training_mean
from .schema import RESERVED_LABEL, RunConfig, load_config, load_inputs
from .triangulation import triangulate


@dataclass
class RunResult:
    output_dir: Path
    predictions: pd.DataFrame


def _config_echo(config: RunConfig) -> dict[str, object]:
    truth_path = getattr(config.inputs, "truth", None)
    return {
        "provenance": config.provenance,
        "seed": config.seed,
        "inputs": {
            "expression": str(config.inputs.expression),
            "cells": str(config.inputs.cells),
            "features": str(config.inputs.features),
            "lr_pairs": str(config.inputs.lr_pairs),
            "contacts": str(config.inputs.contacts),
            "truth": None if truth_path is None else str(truth_path),
            "expression_scale": config.inputs.expression_scale,
        },
        "reference": {"mode": config.reference.mode},
        "attention": {
            "normalization": config.attention.normalization,
            "temperature": config.attention.temperature,
        },
        "assignment": {"mode": config.assignment.mode},
        "cci": {
            "transform": config.cci.transform,
            "direction": config.cci.direction,
            "lr_selection": config.cci.lr_selection,
            "top_k": config.cci.top_k,
        },
        "ambiguity_groups": [
            {
                "name": group.name,
                "candidates": list(group.candidates),
                "references": list(group.references),
            }
            for group in config.ambiguity_groups
        ],
        "output_dir": str(config.output_dir),
        "config_path": str(config.config_path),
    }


def _clean_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("predictions.csv", "assignment_trace.csv", "cci_evidence.csv", "manifest.json", "metrics.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for directory_name in ("attention", "topk"):
        path = output_dir / directory_name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        path.mkdir()


def _reference_profiles(config: RunConfig, inputs, training_ids: list[str], expected_labels: list[str]):
    training_labels = inputs.cells.loc[training_ids, "label"].astype(str)
    if config.reference.mode == "training_mean":
        return training_mean(
            inputs.expression.loc[training_ids],
            training_labels,
            inputs.used_feature_genes,
            expected_labels=expected_labels,
        )
    if config.reference.mode == "binary_marker":
        return binary_marker(
            inputs.features,
            training_labels,
            inputs.used_feature_genes,
            inputs.ambiguity_groups,
        )
    raise ValueError(f"unsupported reference mode: {config.reference.mode}")


def run(config: str | Path | RunConfig) -> RunResult:
    """Run M2 leave-one-sample assignment and triangulation."""

    if isinstance(config, (str, Path)):
        config = load_config(config)
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a path or RunConfig")
    inputs = load_inputs(config)
    selected_lr = select_lr(inputs.lr_pairs, config.cci.lr_selection)
    expected_labels = list(dict.fromkeys(inputs.cells["label"].astype(str).tolist()))
    sample_ids = list(dict.fromkeys(inputs.cells["sample_id"].astype(str).tolist()))
    invalid_sample_ids = [sample_id for sample_id in sample_ids if "/" in sample_id or "\\" in sample_id]
    if invalid_sample_ids:
        raise ValueError(f"sample_id cannot contain slash or backslash: {', '.join(invalid_sample_ids)}")

    _clean_outputs(config.output_dir)
    group_by_name = {group.name: group for group in config.ambiguity_groups}
    assignment_traces: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    predictions_by_cell: dict[str, dict[str, object]] = {}
    fold_manifest: dict[str, dict[str, object]] = {}

    sample_values = inputs.cells["sample_id"].astype(str)
    cell_ids = inputs.cells.index.astype(str)
    for sample_id in sample_ids:
        fold_mask = sample_values.to_numpy() == sample_id
        target_ids = cell_ids[fold_mask].tolist()
        training_ids = cell_ids[~fold_mask].tolist()
        training_labels = inputs.cells.loc[training_ids, "label"].astype(str)
        missing_labels = [
            label for label in expected_labels if label not in set(training_labels.tolist())
        ]
        fold_manifest[sample_id] = {"missing_labels": missing_labels}
        profiles = _reference_profiles(config, inputs, training_ids, expected_labels)
        target_expression = inputs.expression.loc[target_ids, profiles.columns]
        attention_result = attention(
            target_expression,
            profiles,
            config.attention.normalization,
            config.attention.temperature,
        )
        np.savez(
            config.output_dir / "attention" / f"{sample_id}.npz",
            S=attention_result.S.to_numpy(dtype=float),
            P_col=attention_result.P_col.to_numpy(dtype=float),
            P_row=attention_result.P_row.to_numpy(dtype=float),
            L=attention_result.L.to_numpy(dtype=float),
            rows=np.asarray(attention_result.S.index.astype(str).tolist(), dtype=str),
            cols=np.asarray(attention_result.S.columns.astype(str).tolist(), dtype=str),
        )
        assignment = greedy_assign(
            attention_result.L,
            capacities(profiles.index, config.ambiguity_groups, config.assignment.mode),
        )
        trace = assignment.trace.copy()
        trace.insert(0, "fold", sample_id)
        assignment_traces.append(trace)
        triangulations = {}
        for group in config.ambiguity_groups:
            triangulation = triangulate(
                assignment.labels,
                inputs.expression,
                group,
                inputs.contacts,
                selected_lr,
                config.cci.transform,
                config.cci.direction,
                sample_id=sample_id,
            )
            triangulations[group.name] = triangulation
            evidence = triangulation.evidence.copy()
            evidence.insert(0, "fold", sample_id)
            evidence_frames.append(evidence)

        for cell_id in target_ids:
            stage1_label = str(assignment.labels.loc[cell_id])
            if stage1_label == RESERVED_LABEL:
                final_label = RESERVED_LABEL
                status = "unassigned"
                reason = ""
            elif stage1_label in group_by_name:
                triangulation = triangulations[stage1_label]
                final_label = str(triangulation.labels.loc[cell_id])
                status = str(triangulation.status.loc[cell_id])
                reason = str(triangulation.reasons.loc[cell_id])
            else:
                final_label = stage1_label
                status = "assigned"
                reason = ""
            predictions_by_cell[cell_id] = {
                "cell_id": cell_id,
                "sample_id": sample_id,
                "fold": sample_id,
                "stage1_label": stage1_label,
                "final_label": final_label,
                "status": status,
                "unresolved_reason": reason,
                "provenance": config.provenance,
            }
        selected_lr.to_csv(config.output_dir / "topk" / f"{sample_id}.csv", index=False)

    prediction_rows = [predictions_by_cell[str(cell_id)] for cell_id in inputs.cells.index.astype(str)]
    predictions = pd.DataFrame(
        prediction_rows,
        columns=[
            "cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason", "provenance"
        ],
    )
    predictions.to_csv(config.output_dir / "predictions.csv", index=False)

    trace_columns = ["fold", "order", "cell_id", "identity", "L", "tie", "status"]
    assignment_trace = pd.concat(assignment_traces, ignore_index=True) if assignment_traces else pd.DataFrame(columns=trace_columns)
    assignment_trace.reindex(columns=trace_columns).to_csv(config.output_dir / "assignment_trace.csv", index=False)
    evidence_columns = [
        "fold", "sample_id", "group", "reference_identity", "reference_cell", "bucket_cell",
        *[column for column in selected_lr.columns if column not in {"reference_cell", "bucket_cell", "contribution"}],
        "contribution",
    ]
    evidence_columns = list(dict.fromkeys(evidence_columns))
    cci_evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame(columns=evidence_columns)
    cci_evidence.reindex(columns=evidence_columns).to_csv(config.output_dir / "cci_evidence.csv", index=False)

    manifest = {
        "config": _config_echo(config),
        "provenance": config.provenance,
        "versions": {
            "chacam": version("chacam"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_dimensions": {
            "cells": int(len(inputs.cells)),
            "genes": int(len(inputs.expression.columns)),
            "samples": int(inputs.cells["sample_id"].nunique()),
        },
        "used_feature_genes": inputs.used_feature_genes,
        "missing_feature_genes": inputs.missing_feature_genes,
        "dropped_lr_pairs": inputs.dropped_lr_pairs,
        "seed": config.seed,
        "lr_selection": config.cci.lr_selection,
        "folds": fold_manifest,
    }
    with (config.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return RunResult(output_dir=config.output_dir, predictions=predictions)

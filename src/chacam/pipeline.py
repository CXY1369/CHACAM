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
from .cci import select_lr, train_topk
from .evaluation import compute_metrics
from .reference import binary_marker, training_mean
from .schema import RESERVED_LABEL, Inputs, RunConfig, load_config, load_inputs
from .triangulation import triangulate


@dataclass
class RunResult:
    output_dir: Path
    predictions: pd.DataFrame


@dataclass
class FoldFit:
    profiles: pd.DataFrame
    lr_table: pd.DataFrame
    missing_labels: list[str]
    n_contact_pairs: int | None
    n_noncontact_pairs: int | None


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


def _fold_ids(cells: pd.DataFrame, sample_id: str) -> tuple[list[str], list[str]]:
    sample_values = cells["sample_id"].astype(str)
    sample_key = str(sample_id)
    if not sample_values.eq(sample_key).any():
        raise ValueError(f"sample_id {sample_id!r} is absent from cells")
    cell_ids = cells.index.astype(str)
    fold_mask = sample_values.to_numpy() == sample_key
    return cell_ids[fold_mask].tolist(), cell_ids[~fold_mask].tolist()


def fit_fold(inputs: Inputs, config: RunConfig, sample_id: str) -> FoldFit:
    """Fit reference profiles and the fold L-R table from training cells only."""

    _target_ids, training_ids = _fold_ids(inputs.cells, sample_id)
    expected_labels = list(dict.fromkeys(inputs.cells["label"].astype(str).tolist()))
    training_labels = inputs.cells.loc[training_ids, "label"].astype(str)
    missing_labels = [label for label in expected_labels if label not in set(training_labels.tolist())]
    profiles = _reference_profiles(config, inputs, training_ids, expected_labels)
    if config.cci.lr_selection == "given":
        return FoldFit(
            profiles=profiles,
            lr_table=select_lr(inputs.lr_pairs, "given"),
            missing_labels=missing_labels,
            n_contact_pairs=None,
            n_noncontact_pairs=None,
        )
    if config.cci.lr_selection == "train_topk":
        selected = train_topk(
            inputs.expression,
            inputs.cells.loc[training_ids],
            inputs.lr_pairs,
            inputs.contacts,
            [group.name for group in config.ambiguity_groups],
            config.cci.transform,
            config.cci.top_k,
        )
        return FoldFit(
            profiles=profiles,
            lr_table=selected.table,
            missing_labels=missing_labels,
            n_contact_pairs=selected.n_contact_pairs,
            n_noncontact_pairs=selected.n_noncontact_pairs,
        )
    raise ValueError(f"unsupported lr_selection: {config.cci.lr_selection}")


def run(config: str | Path | RunConfig) -> RunResult:
    """Run leave-one-sample assignment, triangulation, and metrics."""

    if isinstance(config, (str, Path)):
        config = load_config(config)
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a path or RunConfig")
    inputs = load_inputs(config)
    sample_ids = list(dict.fromkeys(inputs.cells["sample_id"].astype(str).tolist()))
    invalid_sample_ids = [sample_id for sample_id in sample_ids if "/" in sample_id or "\\" in sample_id]
    if invalid_sample_ids:
        raise ValueError(f"sample_id cannot contain slash or backslash: {', '.join(invalid_sample_ids)}")

    group_by_name = {group.name: group for group in config.ambiguity_groups}
    assignment_traces: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    attention_by_fold: dict[str, object] = {}
    lr_by_fold: dict[str, pd.DataFrame] = {}
    predictions_by_cell: dict[str, dict[str, object]] = {}
    fold_manifest: dict[str, dict[str, object]] = {}

    for sample_id in sample_ids:
        target_ids, _training_ids = _fold_ids(inputs.cells, sample_id)
        fit = fit_fold(inputs, config, sample_id)
        fold_manifest[sample_id] = {
            "missing_labels": fit.missing_labels,
            "n_selected_lr": int(len(fit.lr_table)),
            "n_contact_pairs": fit.n_contact_pairs,
            "n_noncontact_pairs": fit.n_noncontact_pairs,
        }
        target_expression = inputs.expression.loc[target_ids, fit.profiles.columns]
        attention_result = attention(
            target_expression,
            fit.profiles,
            config.attention.normalization,
            config.attention.temperature,
        )
        attention_by_fold[sample_id] = attention_result
        lr_by_fold[sample_id] = fit.lr_table
        assignment = greedy_assign(
            attention_result.L,
            capacities(fit.profiles.index, config.ambiguity_groups, config.assignment.mode),
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
                fit.lr_table,
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

    prediction_rows = [predictions_by_cell[str(cell_id)] for cell_id in inputs.cells.index.astype(str)]
    predictions = pd.DataFrame(
        prediction_rows,
        columns=[
            "cell_id", "sample_id", "fold", "stage1_label", "final_label", "status", "unresolved_reason", "provenance"
        ],
    )
    metrics = {
        "provenance": config.provenance,
        **compute_metrics(predictions, inputs.cells, config.ambiguity_groups, config.inputs.truth),
    }

    _clean_outputs(config.output_dir)
    for sample_id, attention_result in attention_by_fold.items():
        np.savez(
            config.output_dir / "attention" / f"{sample_id}.npz",
            S=attention_result.S.to_numpy(dtype=float),
            P_col=attention_result.P_col.to_numpy(dtype=float),
            P_row=attention_result.P_row.to_numpy(dtype=float),
            L=attention_result.L.to_numpy(dtype=float),
            rows=np.asarray(attention_result.S.index.astype(str).tolist(), dtype=str),
            cols=np.asarray(attention_result.S.columns.astype(str).tolist(), dtype=str),
        )
        lr_by_fold[sample_id].to_csv(config.output_dir / "topk" / f"{sample_id}.csv", index=False)
    predictions.to_csv(config.output_dir / "predictions.csv", index=False)

    trace_columns = ["fold", "order", "cell_id", "identity", "L", "tie", "status"]
    assignment_trace = pd.concat(assignment_traces, ignore_index=True) if assignment_traces else pd.DataFrame(columns=trace_columns)
    assignment_trace.reindex(columns=trace_columns).to_csv(config.output_dir / "assignment_trace.csv", index=False)
    lr_columns = []
    for table in lr_by_fold.values():
        lr_columns.extend(
            column for column in table.columns if column not in {"reference_cell", "bucket_cell", "contribution"}
        )
    evidence_columns = [
        "fold", "sample_id", "group", "reference_identity", "reference_cell", "bucket_cell",
        *lr_columns,
        "contribution",
    ]
    evidence_columns = list(dict.fromkeys(evidence_columns))
    cci_evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame(columns=evidence_columns)
    cci_evidence.reindex(columns=evidence_columns).to_csv(config.output_dir / "cci_evidence.csv", index=False)
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

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

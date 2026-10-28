"""Fold-wise and pooled metrics for the framework baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .schema import AmbiguityGroup, load_truth


UNRESOLVED_REASONS = (
    "unresolved_bucket_size",
    "unresolved_no_lr",
    "unresolved_no_reference",
    "unresolved_tie",
    "unresolved_reference_conflict",
)


def _ratio(numerator: int | None, denominator: int | None) -> dict[str, int | float | None]:
    if numerator is None or denominator is None:
        return {"numerator": None, "denominator": None, "value": None}
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": None if denominator == 0 else numerator / denominator,
    }


def _empty_reasons() -> dict[str, int]:
    return {code: 0 for code in UNRESOLVED_REASONS}


def _fold_metrics(
    fold_pred: pd.DataFrame,
    cells: pd.DataFrame,
    groups: Sequence[AmbiguityGroup],
    truth: pd.DataFrame | None,
) -> dict:
    stage1 = fold_pred["stage1_label"].astype(str)
    final = fold_pred["final_label"].astype(str)
    status = fold_pred["status"].astype(str)
    reason = fold_pred["unresolved_reason"].fillna("").astype(str)
    observed = cells.loc[fold_pred.index, "label"].astype(str)
    group_names = {group.name for group in groups}

    ambiguous = stage1.isin(group_names)
    resolved = status.eq("resolved")
    named = status.isin(["assigned", "resolved"])
    reasons = _empty_reasons()
    for code in UNRESOLVED_REASONS:
        reasons[code] = int(reason.eq(code).sum())

    agreement_num = 0
    agreement_den = 0
    for group in groups:
        available = sum(int((stage1 == reference).any()) for reference in group.references)
        bucket = fold_pred.index[stage1.eq(group.name)]
        if available < 2 or len(bucket) != 2:
            continue
        if reason.loc[bucket].eq("unresolved_no_lr").any():
            continue
        agreement_den += 1
        if status.loc[bucket].eq("resolved").all():
            agreement_num += 1

    if truth is None:
        fine = _ratio(None, None)
        named_metric = _ratio(None, None)
    else:
        true_label = truth.loc[fold_pred.index, "true_label"].astype(str)
        fine = _ratio(int((final[resolved] == true_label[resolved]).sum()), int(resolved.sum()))
        named_metric = _ratio(int((final[named] == true_label[named]).sum()), int(named.sum()))

    return {
        "stage1_accuracy": _ratio(int((stage1 == observed).sum()), int(len(fold_pred))),
        "resolution_rate": _ratio(int((resolved & ambiguous).sum()), int(ambiguous.sum())),
        "reference_agreement": _ratio(agreement_num, agreement_den),
        "fine_accuracy": fine,
        "named_accuracy": named_metric,
        "unresolved_reasons": reasons,
    }


def _pool_metrics(fold_metrics: dict[str, dict], truth_provided: bool) -> dict:
    names = ("stage1_accuracy", "resolution_rate", "reference_agreement", "fine_accuracy", "named_accuracy")
    pooled: dict[str, object] = {}
    for name in names:
        if not truth_provided and name in {"fine_accuracy", "named_accuracy"}:
            pooled[name] = _ratio(None, None)
            continue
        numerator = sum(int(fold[name]["numerator"]) for fold in fold_metrics.values())
        denominator = sum(int(fold[name]["denominator"]) for fold in fold_metrics.values())
        pooled[name] = _ratio(numerator, denominator)
    reasons = _empty_reasons()
    for fold in fold_metrics.values():
        for code in UNRESOLVED_REASONS:
            reasons[code] += int(fold["unresolved_reasons"][code])
    pooled["unresolved_reasons"] = reasons
    return pooled


def evaluate(
    predictions: pd.DataFrame,
    cells: pd.DataFrame,
    groups: Sequence[AmbiguityGroup],
    truth: pd.DataFrame | None,
) -> dict:
    """Compute fold-wise and pooled metrics without reading files."""

    if not isinstance(predictions, pd.DataFrame) or not isinstance(cells, pd.DataFrame):
        raise TypeError("predictions and cells must be pandas DataFrames")
    frame = predictions.copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    duplicate_predictions = frame.loc[frame["cell_id"].duplicated(keep=False), "cell_id"].unique().tolist()
    if duplicate_predictions:
        raise ValueError(f"predictions contains duplicate cell_id values: {', '.join(duplicate_predictions[:5])}")
    frame = frame.set_index("cell_id", drop=False)
    cells = cells.copy()
    cells.index = cells.index.astype(str)
    duplicate_cells = cells.index[cells.index.duplicated()].unique().tolist()
    if duplicate_cells:
        raise ValueError(
            f"cells contains duplicate cell_id values after string conversion: {', '.join(duplicate_cells[:5])}"
        )
    missing_cells = [cell_id for cell_id in frame.index if cell_id not in cells.index]
    if missing_cells:
        raise ValueError(
            f"predictions contains cell_id values absent from cells: {', '.join(missing_cells[:5])}"
        )
    if truth is not None:
        truth = truth.copy()
        truth.index = truth.index.astype(str)
        duplicate_truth = truth.index[truth.index.duplicated()].unique().tolist()
        if duplicate_truth:
            raise ValueError(
                f"truth contains duplicate cell_id values after string conversion: {', '.join(duplicate_truth[:5])}"
            )
        missing = [cell_id for cell_id in frame.index if cell_id not in truth.index]
        if missing:
            raise ValueError(f"truth is missing cell_id values: {', '.join(missing[:5])}")

    folds: dict[str, dict] = {}
    fold_order = list(dict.fromkeys(frame["fold"].astype(str).tolist()))
    for fold in fold_order:
        fold_pred = frame.loc[frame["fold"].astype(str) == fold]
        folds[fold] = _fold_metrics(fold_pred, cells, groups, truth)
    return {
        "truth_provided": truth is not None,
        "pooled": _pool_metrics(folds, truth is not None),
        "folds": folds,
    }


def compute_metrics(
    predictions: pd.DataFrame,
    cells: pd.DataFrame,
    groups: Sequence[AmbiguityGroup],
    truth_path: Path | None,
) -> dict:
    """Read truth when a path is given, then evaluate."""

    truth = None if truth_path is None else load_truth(truth_path)
    return evaluate(predictions, cells, groups, truth)

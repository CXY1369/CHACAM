"""Reference profile builders for the two M1 representations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


ReferenceProfiles = pd.DataFrame


def _feature_alignment(expression_columns: Iterable[str], genes: Iterable[str]) -> tuple[list[str], list[str]]:
    expression_columns = [str(column) for column in expression_columns]
    requested = list(dict.fromkeys(str(gene) for gene in genes))
    used = [gene for gene in requested if gene in expression_columns]
    missing = [gene for gene in requested if gene not in expression_columns]
    if not used:
        raise ValueError("no requested feature genes are present in the expression matrix")
    return used, missing


def _label_series(labels: pd.Series | Iterable[str], index: pd.Index) -> pd.Series:
    if isinstance(labels, pd.DataFrame):
        if "label" not in labels.columns:
            raise ValueError("labels DataFrame must contain a label column")
        labels = labels["label"]
    if isinstance(labels, pd.Series):
        if labels.index.equals(index):
            aligned = labels.copy()
        elif set(index).issubset(set(labels.index)):
            aligned = labels.reindex(index)
        else:
            raise ValueError("labels cannot be aligned to expression rows by cell ID")
    else:
        values = list(labels)
        if len(values) != len(index):
            raise ValueError("labels and expression must have the same number of rows")
        aligned = pd.Series(values, index=index)
    if aligned.isna().any():
        raise ValueError("labels contain missing values")
    return aligned.astype(str)


def _set_metadata(profiles: pd.DataFrame, used: list[str], missing: list[str], **extra: object) -> pd.DataFrame:
    profiles.attrs["used_feature_genes"] = used
    profiles.attrs["missing_feature_genes"] = missing
    profiles.attrs.update(extra)
    return profiles


def training_mean(
    expression: pd.DataFrame,
    labels: pd.Series | Iterable[str],
    genes: Iterable[str],
    *,
    expected_labels: Iterable[str],
) -> ReferenceProfiles:
    """Build one mean expression row for each observed label."""

    used, missing = _feature_alignment(expression.columns, genes)
    aligned_labels = _label_series(labels, expression.index)
    expected = list(dict.fromkeys(str(label) for label in expected_labels))
    expected_set = set(expected)
    unexpected_labels = sorted(set(aligned_labels.tolist()) - expected_set)
    if unexpected_labels:
        raise ValueError(
            "observed labels are not in expected_labels: " + ", ".join(unexpected_labels)
        )
    values = expression.loc[:, used].apply(pd.to_numeric, errors="raise")
    profiles = values.assign(_label=aligned_labels).groupby("_label", sort=False, observed=False).mean()
    profiles.index.name = "identity"
    observed = set(aligned_labels.tolist())
    missing_labels = [label for label in expected if label not in observed]
    return _set_metadata(profiles, used, missing, missing_labels=missing_labels)


def _group_parts(group: object) -> tuple[str, tuple[str, ...]]:
    if hasattr(group, "name") and hasattr(group, "candidates"):
        return str(group.name), tuple(str(value) for value in group.candidates)
    if isinstance(group, dict):
        return str(group["name"]), tuple(str(value) for value in group["candidates"])
    raise ValueError("ambiguity_groups entries must expose name and candidates")


def binary_marker(
    features: pd.DataFrame,
    labels: pd.Series | Iterable[str],
    genes: Iterable[str],
    ambiguity_groups: Iterable[object],
) -> ReferenceProfiles:
    """Build a binary marker profile, expanding ambiguity labels by union."""

    if "gene" not in features.columns or "label" not in features.columns:
        raise ValueError("binary_marker requires gene and label columns in features")
    available_genes = list(dict.fromkeys(str(gene) for gene in genes))
    feature_genes = list(dict.fromkeys(features["gene"].astype(str).tolist()))
    used = [gene for gene in available_genes if gene in feature_genes]
    missing = [gene for gene in feature_genes if gene not in available_genes]
    if not used:
        raise ValueError("no feature genes are present in the expression matrix")
    if isinstance(labels, pd.Series):
        label_index = labels.index
        label_values: pd.Series | list[str] = labels
    else:
        label_values = list(labels)
        label_index = pd.RangeIndex(len(label_values))
    aligned_labels = _label_series(label_values, label_index)
    label_order = list(dict.fromkeys(aligned_labels.tolist()))
    feature_labels = features["label"].astype(str)
    groups = dict(_group_parts(group) for group in ambiguity_groups)
    rows: list[list[int]] = []
    zero_labels: list[str] = []
    for label in label_order:
        marker_labels = {label}
        if label not in set(feature_labels) and label in groups:
            marker_labels.update(groups[label])
        marker_genes = set(features.loc[feature_labels.isin(marker_labels), "gene"].astype(str))
        row = [int(gene in marker_genes) for gene in used]
        if not any(row):
            zero_labels.append(label)
        rows.append(row)
    profiles = pd.DataFrame(rows, index=label_order, columns=used, dtype=int)
    profiles.index.name = "identity"
    return _set_metadata(profiles, used, missing, zero_labels=zero_labels)

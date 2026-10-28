"""Reproducible stage-one diagnostics for the Tintori 16-cell data."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chacam
from chacam.assignment import capacities, greedy_assign
from chacam.attention import attention
from chacam.pipeline import fit_fold
from chacam.reference import training_mean
from chacam.schema import RESERVED_LABEL
from chacam.triangulation import triangulate


GENE_SET_ORDER = ("tintori26", "hvg500", "hvg2000", "detected")
NORMALIZATION_ORDER = ("l1", "softmax")
ASSIGNMENT_ORDER = ("unique", "many_to_one")

D2_CELL_COLUMNS = [
    "gene_set",
    "normalization",
    "assignment",
    "fold",
    "cell_id",
    "observed_label",
    "stage1_label",
    "s_argmax",
    "s_rank_true",
]
D2_SUMMARY_COLUMNS = [
    "gene_set",
    "normalization",
    "assignment",
    "fine_correct",
    "fine_total",
    "s_top1_correct",
    "ref_present_correct",
    "ref_present_total",
    "ref_false_when_absent",
    "unassigned",
]
D2_CONFUSION_COLUMNS = [
    "gene_set",
    "normalization",
    "assignment",
    "observed_label",
    "stage1_label",
    "count",
]
D3_COLUMNS = ["fold", "group", "refs", "cell_id", "final_label", "status", "unresolved_reason"]
WRONG_REFERENCE_COLUMNS = [
    "fold",
    "group",
    "reference_identity",
    "reference_cell_id",
    "observed_label",
]


def _ordered_unique(values: pd.Series | list[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _sample_ids(cells: pd.DataFrame) -> list[str]:
    return _ordered_unique(cells["sample_id"].tolist())


def _fold_ids(cells: pd.DataFrame, sample_id: str) -> tuple[list[str], list[str]]:
    sample_values = cells["sample_id"].astype(str)
    sample_key = str(sample_id)
    target_ids = cells.index.astype(str)[sample_values.eq(sample_key)].tolist()
    if not target_ids:
        raise ValueError(f"sample_id {sample_id!r} is absent from cells")
    training_ids = cells.index.astype(str)[~sample_values.eq(sample_key)].tolist()
    return target_ids, training_ids


def select_hvg_genes(expression: pd.DataFrame, training_ids: list[str], n: int) -> list[str]:
    """Select the top-n population-variance genes from training rows only."""

    training = expression.loc[training_ids]
    variances = training.apply(pd.to_numeric, errors="raise").var(axis=0, ddof=0).to_numpy(dtype=float)
    order = np.argsort(-variances, kind="stable")[:n]
    return [str(training.columns[index]) for index in order]


def select_detected_genes(expression: pd.DataFrame, training_ids: list[str]) -> list[str]:
    """Select genes detected in at least one training cell, preserving column order."""

    training = expression.loc[training_ids]
    detected = training.apply(pd.to_numeric, errors="raise").gt(0).any(axis=0)
    return [str(gene) for gene in training.columns[detected]]


def rank_true_labels(similarity: pd.DataFrame, observed_labels: pd.Series) -> pd.Series:
    """Return strict-greater ranks, or an empty string when the label is absent."""

    rows: list[int | str] = []
    for cell_id in similarity.index:
        observed_label = str(observed_labels.loc[cell_id])
        if observed_label not in similarity.columns:
            rows.append("")
            continue
        value = similarity.loc[cell_id, observed_label]
        rows.append(int((similarity.loc[cell_id] > value).sum()) + 1)
    return pd.Series(rows, index=similarity.index, name="s_rank_true", dtype=object)


def build_d1_tables(expression: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the identity-count and per-cell quality tables."""

    sample_ids = _sample_ids(cells)
    identities = _ordered_unique(cells["label"].tolist())
    labels = cells["label"].astype(str)
    samples = cells["sample_id"].astype(str)
    total_by_identity = labels.value_counts()
    count_rows = []
    for fold in sample_ids:
        heldout = samples.eq(fold)
        for identity in identities:
            count_rows.append(
                {
                    "fold": fold,
                    "identity": identity,
                    "n_heldout": int((heldout & labels.eq(identity)).sum()),
                    "n_training": int(total_by_identity.get(identity, 0) - (heldout & labels.eq(identity)).sum()),
                }
            )
    identity_counts = pd.DataFrame(count_rows, columns=["fold", "identity", "n_heldout", "n_training"])

    numeric_expression = expression.loc[cells.index].apply(pd.to_numeric, errors="raise")
    quality_rows = []
    for cell_id in cells.index.astype(str):
        quality_rows.append(
            {
                "cell_id": cell_id,
                "sample_id": str(cells.loc[cell_id, "sample_id"]),
                "label": str(cells.loc[cell_id, "label"]),
                "n_detected": int((numeric_expression.loc[cell_id] > 0).sum()),
            }
        )
    cell_quality = pd.DataFrame(quality_rows, columns=["cell_id", "sample_id", "label", "n_detected"])
    return (
        identity_counts.sort_values(["fold", "identity"], kind="mergesort").reset_index(drop=True),
        cell_quality.sort_values(["cell_id"], kind="mergesort").reset_index(drop=True),
    )


def _group_names(ambiguity_groups: list[object] | tuple[object, ...]) -> set[str]:
    return {str(group.name) for group in ambiguity_groups}


def _reference_identities(ambiguity_groups: list[object] | tuple[object, ...]) -> list[str]:
    return _ordered_unique([reference for group in ambiguity_groups for reference in group.references])


def summarize_stage1_metrics(
    stage1_cells: pd.DataFrame,
    cells: pd.DataFrame,
    ambiguity_groups: list[object] | tuple[object, ...],
) -> dict[str, int]:
    """Summarize D2 metrics for one gene/normalization/assignment combination."""

    group_names = _group_names(ambiguity_groups)
    observed = stage1_cells["observed_label"].astype(str)
    stage1 = stage1_cells["stage1_label"].astype(str)
    fine = ~observed.isin(group_names)
    metrics = {
        "fine_correct": int((fine & stage1.eq(observed)).sum()),
        "fine_total": int(fine.sum()),
        "s_top1_correct": int((fine & stage1_cells["s_argmax"].astype(str).eq(observed)).sum())
        if "s_argmax" in stage1_cells
        else 0,
        "ref_present_correct": 0,
        "ref_present_total": 0,
        "ref_false_when_absent": 0,
        "unassigned": int(stage1.eq(RESERVED_LABEL).sum()),
    }
    references = _reference_identities(ambiguity_groups)
    for fold, fold_rows in stage1_cells.assign(_observed=observed).groupby("fold", sort=False):
        fold_observed = fold_rows["_observed"].astype(str)
        fold_stage1 = fold_rows["stage1_label"].astype(str)
        for reference in references:
            assigned = fold_rows.loc[fold_stage1.eq(reference), "cell_id"].astype(str).tolist()
            observed_present = bool(fold_observed.eq(reference).any())
            if observed_present:
                metrics["ref_present_total"] += 1
                if assigned:
                    assigned_observed = cells.loc[assigned, "label"].astype(str)
                    if assigned_observed.eq(reference).all():
                        metrics["ref_present_correct"] += 1
            elif assigned:
                metrics["ref_false_when_absent"] += 1
    return metrics


def run_d2(inputs: object, config: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete D2 grid and return cells, summary, and confusion tables."""

    expected_labels = _ordered_unique(inputs.cells["label"].tolist())
    sample_ids = _sample_ids(inputs.cells)
    cell_rows: list[dict[str, object]] = []
    for fold in sample_ids:
        target_ids, training_ids = _fold_ids(inputs.cells, fold)
        gene_sets = {
            "tintori26": list(inputs.used_feature_genes),
            "hvg500": select_hvg_genes(inputs.expression, training_ids, 500),
            "hvg2000": select_hvg_genes(inputs.expression, training_ids, 2000),
            "detected": select_detected_genes(inputs.expression, training_ids),
        }
        observed = inputs.cells.loc[target_ids, "label"].astype(str)
        for gene_set in GENE_SET_ORDER:
            training_expression = inputs.expression.loc[training_ids, gene_sets[gene_set]]
            training_labels = inputs.cells.loc[training_ids, "label"].astype(str)
            profiles = training_mean(
                training_expression,
                training_labels,
                gene_sets[gene_set],
                expected_labels=expected_labels,
            )
            profiles.attrs = {}
            target_expression = inputs.expression.loc[target_ids, profiles.columns]
            for normalization in NORMALIZATION_ORDER:
                attention_result = attention(target_expression, profiles, normalization, None)
                s_argmax = attention_result.S.idxmax(axis=1).astype(str)
                s_rank_true = rank_true_labels(attention_result.S, observed)
                for assignment_mode in ASSIGNMENT_ORDER:
                    assignment = greedy_assign(
                        attention_result.L,
                        capacities(profiles.index, inputs.ambiguity_groups, assignment_mode),
                    )
                    for cell_id in target_ids:
                        cell_rows.append(
                            {
                                "gene_set": gene_set,
                                "normalization": normalization,
                                "assignment": assignment_mode,
                                "fold": str(fold),
                                "cell_id": str(cell_id),
                                "observed_label": str(observed.loc[cell_id]),
                                "stage1_label": str(assignment.labels.loc[cell_id]),
                                "s_argmax": str(s_argmax.loc[cell_id]),
                                "s_rank_true": s_rank_true.loc[cell_id],
                            }
                        )
    d2_cells = pd.DataFrame(cell_rows, columns=D2_CELL_COLUMNS)
    d2_cells = d2_cells.sort_values(
        ["gene_set", "normalization", "assignment", "fold", "cell_id"], kind="mergesort"
    ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for combination, combination_rows in d2_cells.groupby(
        ["gene_set", "normalization", "assignment"], sort=False
    ):
        metrics = summarize_stage1_metrics(combination_rows, inputs.cells, inputs.ambiguity_groups)
        summary_rows.append({**dict(zip(["gene_set", "normalization", "assignment"], combination)), **metrics})
        fine = ~combination_rows["observed_label"].astype(str).isin(_group_names(inputs.ambiguity_groups))
        wrong = combination_rows.loc[
            fine & combination_rows["stage1_label"].astype(str).ne(combination_rows["observed_label"].astype(str))
        ]
        if not wrong.empty:
            confusion = wrong.groupby(["observed_label", "stage1_label"], sort=True).size().reset_index(name="count")
            for row in confusion.to_dict("records"):
                confusion_rows.append({**dict(zip(["gene_set", "normalization", "assignment"], combination)), **row})
    d2_summary = pd.DataFrame(summary_rows, columns=D2_SUMMARY_COLUMNS)
    d2_summary = d2_summary.sort_values(
        ["gene_set", "normalization", "assignment"], kind="mergesort"
    ).reset_index(drop=True)
    d2_confusion = pd.DataFrame(confusion_rows, columns=D2_CONFUSION_COLUMNS)
    d2_confusion = d2_confusion.sort_values(
        ["gene_set", "normalization", "assignment", "observed_label", "stage1_label"], kind="mergesort"
    ).reset_index(drop=True)
    return d2_cells, d2_summary, d2_confusion


def _select_d2_baseline(d2_stage1: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "cell_id", "stage1_label"}
    missing = required.difference(d2_stage1.columns)
    if missing:
        raise ValueError(f"D2 stage-one table is missing columns: {', '.join(sorted(missing))}")
    selected = d2_stage1
    if {"gene_set", "normalization", "assignment"}.issubset(d2_stage1.columns):
        selected = d2_stage1.loc[
            d2_stage1["gene_set"].eq("tintori26")
            & d2_stage1["normalization"].eq("l1")
            & d2_stage1["assignment"].eq("unique")
        ]
    if selected.duplicated(["fold", "cell_id"]).any():
        raise AssertionError("each D2 baseline cell must have exactly one stage-one row")
    return selected.copy()


def run_d3(
    inputs: object,
    config: object,
    d2_stage1: pd.DataFrame,
    baseline_predictions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run D3 from the current D2 stage-one labels; baseline is intentionally unused."""

    del baseline_predictions
    selected_d2 = _select_d2_baseline(d2_stage1)
    d3_rows: list[dict[str, object]] = []
    wrong_rows: list[dict[str, object]] = []
    for fold in _sample_ids(inputs.cells):
        target_ids, _training_ids = _fold_ids(inputs.cells, fold)
        fold_d2 = selected_d2.loc[selected_d2["fold"].astype(str).eq(str(fold))].set_index("cell_id")
        if set(fold_d2.index.astype(str)) != set(target_ids) or len(fold_d2) != len(target_ids):
            raise AssertionError(f"D2 must contain exactly one row for every held-out cell in fold {fold}")
        stage1_labels = fold_d2.reindex(target_ids)["stage1_label"].astype(str)
        stage1_labels.index = pd.Index(target_ids)
        observed_labels = inputs.cells.loc[target_ids, "label"].astype(str)
        observed_labels.index = pd.Index(target_ids)
        fit = fit_fold(inputs, config, fold)
        lr_genes = list(
            dict.fromkeys(
                [str(gene) for gene in fit.lr_table["ligand"].tolist()]
                + [str(gene) for gene in fit.lr_table["receptor"].tolist()]
            )
        )
        triangulation_expression = inputs.expression.loc[:, lr_genes]
        for ambiguity_group in config.ambiguity_groups:
            triangulations = {
                "stage1": triangulate(
                    stage1_labels,
                    triangulation_expression,
                    ambiguity_group,
                    inputs.contacts,
                    fit.lr_table,
                    config.cci.transform,
                    config.cci.direction,
                    sample_id=str(fold),
                ),
                "observed": triangulate(
                    observed_labels,
                    triangulation_expression,
                    ambiguity_group,
                    inputs.contacts,
                    fit.lr_table,
                    config.cci.transform,
                    config.cci.direction,
                    sample_id=str(fold),
                ),
            }
            stage1_result = triangulations["stage1"]
            if (stage1_result.status == "resolved").any():
                for reference_identity in ambiguity_group.references:
                    reference_cells = stage1_labels.index[stage1_labels.eq(reference_identity)]
                    for reference_cell_id in reference_cells:
                        observed_label = str(observed_labels.loc[reference_cell_id])
                        if observed_label != str(reference_identity):
                            wrong_rows.append(
                                {
                                    "fold": str(fold),
                                    "group": str(ambiguity_group.name),
                                    "reference_identity": str(reference_identity),
                                    "reference_cell_id": str(reference_cell_id),
                                    "observed_label": observed_label,
                                }
                            )
            for refs, result in triangulations.items():
                for cell_id in result.labels.index.astype(str):
                    d3_rows.append(
                        {
                            "fold": str(fold),
                            "group": str(ambiguity_group.name),
                            "refs": refs,
                            "cell_id": str(cell_id),
                            "final_label": str(result.labels.loc[cell_id]),
                            "status": str(result.status.loc[cell_id]),
                            "unresolved_reason": str(result.reasons.loc[cell_id]),
                        }
                    )
    d3 = pd.DataFrame(d3_rows, columns=D3_COLUMNS)
    d3 = d3.sort_values(["fold", "group", "refs", "cell_id"], kind="mergesort").reset_index(drop=True)
    wrong = pd.DataFrame(wrong_rows, columns=WRONG_REFERENCE_COLUMNS)
    wrong = wrong.sort_values(
        ["fold", "group", "reference_identity", "reference_cell_id"], kind="mergesort"
    ).reset_index(drop=True)
    return d3, wrong


def _normal_reason(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def check_consistency(
    d2_stage1: pd.DataFrame,
    d3: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    group_names: set[str],
) -> dict[str, bool]:
    """Compare freshly computed D2/D3 stage-one outputs with the M4 predictions."""

    d2_selected = _select_d2_baseline(d2_stage1).copy()
    d2_selected["cell_id"] = d2_selected["cell_id"].astype(str)
    baseline = baseline_predictions.copy()
    baseline["cell_id"] = baseline["cell_id"].astype(str)
    d2_cell_ids = d2_selected["cell_id"]
    baseline_cell_ids = baseline["cell_id"]
    d2_has_duplicates = d2_cell_ids.duplicated().any()
    baseline_has_duplicates = baseline_cell_ids.duplicated().any()
    same_stage1_cells = set(d2_cell_ids) == set(baseline_cell_ids)
    stage1_match = False
    if not d2_has_duplicates and not baseline_has_duplicates and same_stage1_cells:
        d2_indexed = d2_selected.set_index("cell_id")
        baseline_indexed = baseline.set_index("cell_id")
        stage1_match = d2_indexed["stage1_label"].astype(str).equals(
            baseline_indexed.reindex(d2_indexed.index)["stage1_label"].astype(str)
        )

    d3_stage1 = d3.loc[d3["refs"].astype(str).eq("stage1")].copy()
    expected_stage1 = d2_selected.loc[
        d2_selected["stage1_label"].astype(str).isin({str(name) for name in group_names}),
        ["cell_id", "stage1_label"],
    ]
    expected_pairs = set(zip(expected_stage1["cell_id"], expected_stage1["stage1_label"].astype(str)))
    actual_pairs = set(zip(d3_stage1["cell_id"].astype(str), d3_stage1["group"].astype(str)))
    complete_d3_set = (
        not d3_stage1["cell_id"].astype(str).duplicated().any()
        and actual_pairs == expected_pairs
    )
    triangulation_match = False
    if complete_d3_set and d3_stage1.empty:
        triangulation_match = True
    elif complete_d3_set and not baseline_has_duplicates:
        baseline_indexed = baseline.set_index("cell_id")
        triangulation_match = True
        for row in d3_stage1.itertuples(index=False):
            if row.cell_id not in baseline_indexed.index:
                triangulation_match = False
                break
            expected = baseline_indexed.loc[row.cell_id]
            if (
                str(row.final_label) != str(expected["final_label"])
                or str(row.status) != str(expected["status"])
                or _normal_reason(row.unresolved_reason) != _normal_reason(expected["unresolved_reason"])
            ):
                triangulation_match = False
                break
    return {"stage1_match": bool(stage1_match), "triangulation_match": bool(triangulation_match)}


def require_consistency(consistency: dict[str, bool]) -> None:
    """Stop before writing outputs when any consistency check fails."""

    failed = [key for key, value in consistency.items() if not value]
    if failed:
        raise RuntimeError("consistency check failed: " + ", ".join(failed))


def serialize_csv(frame: pd.DataFrame, columns: list[str]) -> str:
    """Serialize an output table with the repository's deterministic CSV settings."""

    return frame.reindex(columns=columns).to_csv(index=False, lineterminator="\n", na_rep="")


def _write_outputs(output_dir: Path, tables: dict[str, tuple[pd.DataFrame, list[str]]]) -> None:
    for filename, (frame, columns) in tables.items():
        (output_dir / filename).write_text(serialize_csv(frame, columns), encoding="utf-8")


def main() -> None:
    config_path = Path(__file__).with_name("config_tintori16.yaml")
    started = time.perf_counter()
    config = chacam.load_config(config_path)
    inputs = chacam.load_inputs(config)
    output_dir = config.output_dir.parent / "c_elegans_tintori16_diagnosis"
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    d1_counts, d1_quality = build_d1_tables(inputs.expression, inputs.cells)
    d2_cells, d2_summary, d2_confusion = run_d2(inputs, config)
    d3, d3_wrong = run_d3(inputs, config, d2_cells)
    baseline = pd.read_csv(config.output_dir / "predictions.csv")
    group_names = {str(group.name) for group in config.ambiguity_groups}
    consistency = check_consistency(d2_cells, d3, baseline, group_names)
    require_consistency(consistency)

    _write_outputs(
        output_dir,
        {
            "d1_identity_counts.csv": (d1_counts, ["fold", "identity", "n_heldout", "n_training"]),
            "d1_cell_quality.csv": (d1_quality, ["cell_id", "sample_id", "label", "n_detected"]),
            "d2_stage1_cells.csv": (d2_cells, D2_CELL_COLUMNS),
            "d2_stage1_summary.csv": (d2_summary, D2_SUMMARY_COLUMNS),
            "d2_confusion.csv": (d2_confusion, D2_CONFUSION_COLUMNS),
            "d3_triangulation.csv": (d3, D3_COLUMNS),
            "d3_wrong_reference.csv": (d3_wrong, WRONG_REFERENCE_COLUMNS),
        },
    )

    print("d2_stage1_summary.csv:")
    print(d2_summary.to_string(index=False))
    print("d3 counts by refs, group, unresolved_reason:")
    d3_counts = (
        d3.assign(unresolved_reason=d3["unresolved_reason"].map(_normal_reason))
        .groupby(["refs", "group", "unresolved_reason"], dropna=False, sort=True)
        .size()
        .reset_index(name="count")
    )
    print(d3_counts.to_string(index=False) if not d3_counts.empty else "<none>")
    print("d3_wrong_reference.csv:")
    print(d3_wrong.to_string(index=False) if not d3_wrong.empty else "<none>")
    print(f"consistency: {consistency}")
    print(f"elapsed_seconds: {time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()

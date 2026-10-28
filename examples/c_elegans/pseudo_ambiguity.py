"""Run the preregistered M6 pseudo-ambiguity benchmark."""

from __future__ import annotations

import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from chacam.cci import train_topk
from chacam.schema import AmbiguityGroup
from chacam.triangulation import triangulate


TOP_K = 50
TRANSFORM = "none"
DIRECTION = "both"
TINTORI_GROUPS = ("ABalx", "ABarx", "ABplx", "ABprx")
DECISION_COLUMNS = [
    "dataset",
    "stage",
    "pair",
    "group",
    "fold",
    "mode",
    "cell_x",
    "cell_y",
    "references",
    "n_lr",
    "status",
    "unresolved_reason",
    "final_x",
    "final_y",
    "correct",
]


def find_sister_pairs(fine_labels: Iterable[str], existing_labels: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Return all preregistered sister pairs from fine labels."""

    labels = sorted({str(label) for label in fine_labels})
    existing = set(str(label) for label in existing_labels)
    pairs: list[tuple[str, str]] = []
    for first, second in combinations(labels, 2):
        if len(first) != len(second) or first[:-1] != second[:-1]:
            continue
        suffixes = {first[-1], second[-1]}
        if suffixes not in ({"a", "p"}, {"l", "r"}):
            continue
        group_name = f"{first}+{second}"
        if group_name in existing:
            raise ValueError(f"pseudo-group name already exists as a label: {group_name}")
        pairs.append((first, second))
    return pairs


def _contact_map(contacts: pd.DataFrame) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in contacts.to_dict(orient="records"):
        first, second = sorted((str(row["identity_a"]), str(row["identity_b"])))
        key = (first, second)
        value = float(row["contact"])
        if key in result and result[key] != value:
            raise ValueError(f"conflicting contact records for {first} and {second}")
        result[key] = value
    return result


def eligible_references(
    pair: tuple[str, str],
    fold_cells: pd.DataFrame,
    contacts: pd.DataFrame,
    fine_labels: Iterable[str],
) -> tuple[str, ...]:
    """Find fine references with one recorded, differing contact value and one cell in the fold."""

    first, second = pair
    contact_values = _contact_map(contacts)
    labels = fold_cells["label"].astype(str)
    references = []
    for reference in sorted({str(label) for label in fine_labels} - {first, second}):
        key_first = tuple(sorted((reference, first)))
        key_second = tuple(sorted((reference, second)))
        if key_first not in contact_values or key_second not in contact_values:
            continue
        if contact_values[key_first] == contact_values[key_second]:
            continue
        if int(labels.eq(reference).sum()) != 1:
            continue
        references.append(reference)
    return tuple(references)


def prepare_fold(
    cells: pd.DataFrame,
    pair: tuple[str, str],
    fold: str,
    group_names: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Merge a pair in every cell and return merged cells, training cells, and group names."""

    first, second = pair
    group_name = f"{first}+{second}"
    merged = cells.copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    merged["label"] = merged["label"].astype(str)
    merged.loc[merged["label"].isin(pair), "label"] = group_name
    training = merged.loc[merged["sample_id"].ne(str(fold))].copy()
    all_groups = tuple(sorted(set(str(name) for name in group_names) | {group_name}))
    return merged, training, all_groups


def select_fold_topk(
    expression: pd.DataFrame,
    cells: pd.DataFrame,
    pair: tuple[str, str],
    fold: str,
    lr_pairs: pd.DataFrame,
    contacts: pd.DataFrame,
    group_names: Iterable[str],
    *,
    top_k: int = TOP_K,
):
    """Select Top-K once for one leave-one-embryo fold and one pseudo-group."""

    _merged, training, all_groups = prepare_fold(cells, pair, fold, group_names)
    training_expression = expression.loc[training.index]
    return train_topk(
        training_expression,
        training,
        lr_pairs,
        contacts,
        all_groups,
        TRANSFORM,
        top_k,
    )


def _swap_candidate_contacts(contacts: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
    """Swap the two candidate contact values for every reference with both records present."""

    first, second = pair
    swapped = contacts.copy()
    values = _contact_map(contacts)
    for reference in sorted({name for key in values for name in key} - {first, second}):
        first_key = tuple(sorted((reference, first)))
        second_key = tuple(sorted((reference, second)))
        if first_key not in values or second_key not in values:
            continue
        for index, row in swapped.iterrows():
            key = tuple(sorted((str(row["identity_a"]), str(row["identity_b"]))))
            if key == first_key:
                swapped.at[index, "contact"] = values[second_key]
            elif key == second_key:
                swapped.at[index, "contact"] = values[first_key]
    return swapped


def _decision_row(
    *,
    dataset: str,
    stage: int | str,
    pair: tuple[str, str],
    fold: str,
    mode: str,
    cell_x: str,
    cell_y: str,
    references: tuple[str, ...],
    n_lr: int,
    status: str,
    reason: str,
    final_x: str,
    final_y: str,
    correct: int | None,
) -> dict[str, Any]:
    first, second = pair
    return {
        "dataset": dataset,
        "stage": stage,
        "pair": f"{first}+{second}",
        "group": f"{first}+{second}",
        "fold": str(fold),
        "mode": mode,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "references": ";".join(references),
        "n_lr": n_lr,
        "status": status,
        "unresolved_reason": reason,
        "final_x": final_x,
        "final_y": final_y,
        "correct": correct,
    }


def _run_mode(
    *,
    dataset: str,
    stage: int | str,
    pair: tuple[str, str],
    fold: str,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    selected_lr: pd.DataFrame,
    contacts: pd.DataFrame,
    references: tuple[str, ...],
    mode: str,
) -> dict[str, Any]:
    first, second = pair
    cell_x_values = cells.index[(cells["sample_id"].astype(str) == str(fold)) & (cells["label"].astype(str) == first)]
    cell_y_values = cells.index[(cells["sample_id"].astype(str) == str(fold)) & (cells["label"].astype(str) == second)]
    if len(cell_x_values) != 1 or len(cell_y_values) != 1:
        return _decision_row(
            dataset=dataset,
            stage=stage,
            pair=pair,
            fold=fold,
            mode=mode,
            cell_x="",
            cell_y="",
            references=references,
            n_lr=len(selected_lr),
            status="no_bucket",
            reason="unresolved_bucket_size",
            final_x="",
            final_y="",
            correct=None,
        )

    cell_x, cell_y = str(cell_x_values[0]), str(cell_y_values[0])
    group_name = f"{first}+{second}"
    group = AmbiguityGroup(name=group_name, candidates=pair, references=references)
    merged = cells.copy()
    merged["label"] = merged["label"].astype(str)
    merged.loc[merged["label"].isin(pair), "label"] = group_name
    fold_labels = merged.loc[merged["sample_id"].astype(str).eq(str(fold)), "label"]
    contacts_for_mode = contacts if mode == "oracle" else _swap_candidate_contacts(contacts, pair)
    result = triangulate(
        fold_labels,
        expression,
        group,
        contacts_for_mode,
        selected_lr,
        TRANSFORM,
        DIRECTION,
        sample_id=str(fold),
    )
    bucket_ids = [cell_x, cell_y]
    statuses = [str(result.status.loc[cell_id]) for cell_id in bucket_ids]
    reasons = [str(result.reasons.loc[cell_id]) for cell_id in bucket_ids]
    if len(set(statuses)) != 1 or len(set(reasons)) != 1:
        raise RuntimeError(f"triangulation returned inconsistent cell statuses for {dataset} {pair} {fold}")
    status = statuses[0]
    reason = reasons[0]
    final_x = str(result.labels.loc[cell_x])
    final_y = str(result.labels.loc[cell_y])
    correct = int(final_x == first and final_y == second) if status == "resolved" else None
    return _decision_row(
        dataset=dataset,
        stage=stage,
        pair=pair,
        fold=fold,
        mode=mode,
        cell_x=cell_x,
        cell_y=cell_y,
        references=references,
        n_lr=len(selected_lr),
        status=status,
        reason=reason,
        final_x=final_x,
        final_y=final_y,
        correct=correct,
    )


def validate_swap_consistency(oracle: Mapping[str, Any], swap_control: Mapping[str, Any]) -> None:
    """Fail closed unless the swap-control invariants hold for one decision."""

    if oracle["status"] != swap_control["status"] or oracle["unresolved_reason"] != swap_control["unresolved_reason"]:
        raise RuntimeError("swap_control status or unresolved_reason differs from oracle")
    if oracle["status"] == "resolved" and swap_control["correct"] != 1 - oracle["correct"]:
        raise RuntimeError("swap_control correctness is not the complement of oracle")


def run_decision(
    *,
    dataset: str,
    stage: int | str,
    pair: tuple[str, str],
    fold: str,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    contacts: pd.DataFrame,
    group_names: Iterable[str],
    fine_labels: Iterable[str] | None = None,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Run one oracle/swap-control decision and validate it before returning rows."""

    if fine_labels is None:
        fine_labels = sorted(set(cells["label"].astype(str)) - set(str(name) for name in group_names))
    fold_cells = cells.loc[cells["sample_id"].astype(str).eq(str(fold))]
    references = eligible_references(pair, fold_cells, contacts, fine_labels)
    selected = select_fold_topk(
        expression,
        cells,
        pair,
        fold,
        lr_pairs,
        contacts,
        group_names,
        top_k=top_k,
    )
    selected_lr = selected.table
    oracle = _run_mode(
        dataset=dataset,
        stage=stage,
        pair=pair,
        fold=fold,
        cells=cells,
        expression=expression,
        selected_lr=selected_lr,
        contacts=contacts,
        references=references,
        mode="oracle",
    )
    swap_control = _run_mode(
        dataset=dataset,
        stage=stage,
        pair=pair,
        fold=fold,
        cells=cells,
        expression=expression,
        selected_lr=selected_lr,
        contacts=contacts,
        references=references,
        mode="swap_control",
    )
    validate_swap_consistency(oracle, swap_control)
    return [oracle, swap_control]


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_factor = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_quantile(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _beta_cdf(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else _beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else _beta_quantile(1.0 - alpha / 2.0, successes + 1, trials - successes)
    return lower, upper


def _per_embryo(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    native = decisions.loc[decisions["stage"].astype(str) != "all"]
    for (dataset, stage, mode, fold), frame in native.groupby(["dataset", "stage", "mode", "fold"], sort=True):
        resolved = frame.loc[frame["status"].eq("resolved")]
        resolved_count = len(resolved)
        correct = int(resolved["correct"].sum()) if resolved_count else 0
        a_e = correct / resolved_count if resolved_count else np.nan
        if resolved_count == 0:
            outcome = "no_resolved"
        elif a_e > 0.5:
            outcome = "win"
        elif a_e < 0.5:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "mode": mode,
                "fold": fold,
                "resolved": resolved_count,
                "correct": correct,
                "a_e": a_e,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows, columns=["dataset", "stage", "mode", "fold", "resolved", "correct", "a_e", "outcome"])


def _summary_row(dataset: str, stage: int | str, mode: str, decisions: pd.DataFrame, embryo_rows: pd.DataFrame) -> dict[str, Any]:
    evaluable = decisions.loc[decisions["status"].ne("no_bucket")]
    with_reference = evaluable.loc[evaluable["references"].astype(str).ne("")]
    resolved = decisions.loc[decisions["status"].eq("resolved")]
    n_resolved = len(resolved)
    n_correct = int(resolved["correct"].sum()) if n_resolved else 0
    embryos = embryo_rows.loc[(embryo_rows["dataset"] == dataset) & (embryo_rows["mode"] == mode)]
    if str(stage) != "all":
        embryos = embryos.loc[embryos["stage"].astype(str).eq(str(stage))]
    wins = int((embryos["outcome"] == "win").sum())
    losses = int((embryos["outcome"] == "loss").sum())
    ties = int((embryos["outcome"] == "tie").sum())
    no_resolved = int((embryos["outcome"] == "no_resolved").sum())
    sign_p = _binomial_upper_tail(wins, wins + losses)
    cp_lower, cp_upper = _clopper_pearson(n_correct, n_resolved)
    return {
        "dataset": dataset,
        "stage": stage,
        "mode": mode,
        "evaluable": len(evaluable),
        "with_reference": len(with_reference),
        "resolved": n_resolved,
        "correct": n_correct,
        "coverage": n_resolved / len(evaluable) if len(evaluable) else np.nan,
        "accuracy": n_correct / n_resolved if n_resolved else np.nan,
        "embryo_wins": wins,
        "embryo_losses": losses,
        "embryo_ties": ties,
        "embryo_no_resolved": no_resolved,
        "sign_test_p": sign_p,
        "descriptive_binomial_p": _binomial_upper_tail(n_correct, n_resolved),
        "descriptive_cp_lower": cp_lower,
        "descriptive_cp_upper": cp_upper,
    }


def summarize_decisions(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stage and dataset summaries plus embryo-level outcomes."""

    decisions = decisions.copy()
    decisions["stage"] = decisions["stage"].astype(object)
    embryo_rows = _per_embryo(decisions)
    summary_rows: list[dict[str, Any]] = []
    native_groups = decisions.loc[decisions["stage"].astype(str) != "all"].groupby(
        ["dataset", "stage", "mode"], sort=True
    )
    for (dataset, stage, mode), frame in native_groups:
        summary_rows.append(_summary_row(dataset, stage, mode, frame, embryo_rows))
    for (dataset, mode), frame in decisions.loc[decisions["stage"].astype(str) != "all"].groupby(
        ["dataset", "mode"], sort=True
    ):
        embryo_subset = embryo_rows.loc[embryo_rows["dataset"].eq(dataset) & embryo_rows["mode"].eq(mode)]
        summary_rows.append(_summary_row(dataset, "all", mode, frame, embryo_subset))
    columns = [
        "dataset", "stage", "mode", "evaluable", "with_reference", "resolved", "correct", "coverage", "accuracy",
        "embryo_wins", "embryo_losses", "embryo_ties", "embryo_no_resolved", "sign_test_p",
        "descriptive_binomial_p", "descriptive_cp_lower", "descriptive_cp_upper",
    ]
    return pd.DataFrame(summary_rows, columns=columns), embryo_rows


def _load_stage_dataset(dataset: str, stage: int, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    cells = pd.read_csv(root / "cells.csv").set_index("cell_id")
    expression = pd.read_csv(root / "expression.csv").set_index("cell_id")
    lr_pairs = pd.read_csv(root / "lr_pairs.csv")
    expression_genes = set(str(column) for column in expression.columns)
    lr_pairs = lr_pairs.loc[
        lr_pairs["ligand"].astype(str).isin(expression_genes)
        & lr_pairs["receptor"].astype(str).isin(expression_genes)
    ].reset_index(drop=True)
    contacts = pd.read_csv(root / "contacts.csv")
    if dataset == "Tintori 16":
        group_names = TINTORI_GROUPS
        fine_labels = tuple(sorted(set(cells["label"].astype(str)) - set(group_names)))
    else:
        group_names = tuple(sorted(cells.loc[cells["label_kind"].eq("equivalence"), "label"].astype(str).unique()))
        fine_labels = tuple(sorted(cells.loc[cells["label_kind"].eq("fine"), "label"].astype(str).unique()))
    return cells, expression, lr_pairs, contacts, tuple(group_names), fine_labels


def run_dataset(
    *,
    dataset: str,
    stage: int,
    root: Path,
) -> list[dict[str, Any]]:
    cells, expression, lr_pairs, contacts, group_names, fine_labels = _load_stage_dataset(dataset, stage, root)
    pairs = find_sister_pairs(fine_labels, set(cells["label"].astype(str)))
    rows: list[dict[str, Any]] = []
    folds = sorted(cells["sample_id"].astype(str).unique())
    for pair in pairs:
        for fold in folds:
            rows.extend(
                run_decision(
                    dataset=dataset,
                    stage=stage,
                    pair=pair,
                    fold=fold,
                    cells=cells,
                    expression=expression,
                    lr_pairs=lr_pairs,
                    contacts=contacts,
                    group_names=group_names,
                    fine_labels=fine_labels,
                )
            )
    return rows


def write_outputs(decisions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write all outputs only after all benchmark decisions have passed validation."""

    summary, per_embryo = summarize_decisions(decisions)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    decisions.loc[:, DECISION_COLUMNS].sort_values(["dataset", "stage", "pair", "fold", "mode"]).to_csv(
        output_dir / "decisions.csv", index=False, lineterminator="\n"
    )
    summary.to_csv(output_dir / "summary.csv", index=False, lineterminator="\n")
    per_embryo.to_csv(output_dir / "per_embryo.csv", index=False, lineterminator="\n")
    return summary, per_embryo


def main() -> None:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    reference = repo_root / "data" / "reference"
    rows: list[dict[str, Any]] = []
    tintori_root = reference / "derived"
    rows.extend(run_dataset(dataset="Tintori 16", stage=16, root=tintori_root))
    for stage in (15, 28, 51):
        rows.extend(
            run_dataset(
                dataset="Cole-Yanai",
                stage=stage,
                root=reference / "derived_cole_yanai" / f"stage{stage}",
            )
        )
    decisions = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    output_dir = repo_root / "runs" / "pseudo_ambiguity"
    summary, _per_embryo = write_outputs(decisions, output_dir)
    print(summary.to_string(index=False))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()

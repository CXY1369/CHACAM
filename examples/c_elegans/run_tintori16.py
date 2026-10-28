"""Run CHACAM on the converted Tintori 16-cell data and print diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import chacam


MS_C_LABELS = {"MSa", "MSp", "Ca", "Cp"}


def _reference_check(predictions, inputs, config, sample_ids):
    rows = []
    for sample_id in sample_ids:
        fold_predictions = predictions.loc[predictions["sample_id"].astype(str).eq(str(sample_id))]
        for group in config.ambiguity_groups:
            for reference_identity in group.references:
                assigned_ids = [
                    str(cell_id) for cell_id in fold_predictions.loc[
                        fold_predictions["stage1_label"].astype(str).eq(reference_identity), "cell_id"
                    ]
                ]
                if not assigned_ids:
                    assigned_ids = [None]
                for cell_id in assigned_ids:
                    observed_label = None if cell_id is None else str(inputs.cells.loc[cell_id, "label"])
                    rows.append(
                        {
                            "fold": str(sample_id),
                            "group": group.name,
                            "reference_identity": reference_identity,
                            "assigned_cell_id": "" if cell_id is None else cell_id,
                            "observed_label": "" if observed_label is None else observed_label,
                            "match": bool(cell_id is not None and observed_label == reference_identity),
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "fold",
            "group",
            "reference_identity",
            "assigned_cell_id",
            "observed_label",
            "match",
        ],
    )


def _ms_c_stage1(predictions, inputs, sample_ids):
    rows = []
    for sample_id in sample_ids:
        fold_predictions = predictions.loc[predictions["sample_id"].astype(str).eq(str(sample_id))]
        for cell_id in fold_predictions["cell_id"].astype(str):
            observed_label = str(inputs.cells.loc[cell_id, "label"])
            if observed_label not in MS_C_LABELS:
                continue
            assigned_label = str(
                fold_predictions.loc[fold_predictions["cell_id"].astype(str).eq(cell_id), "stage1_label"].item()
            )
            rows.append(
                {
                    "fold": str(sample_id),
                    "cell_id": str(cell_id),
                    "observed_label": observed_label,
                    "assigned_label": assigned_label,
                    "match": assigned_label == observed_label,
                }
            )
    return pd.DataFrame(rows, columns=["fold", "cell_id", "observed_label", "assigned_label", "match"])


def _topk_diagnostics(config, inputs, sample_ids):
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for sample_id in sample_ids:
        selected = pd.read_csv(config.output_dir / "topk" / f"{sample_id}.csv")
        fold = manifest["folds"][str(sample_id)]
        positive = chacam.train_topk(
            inputs.expression.loc[inputs.cells.index[inputs.cells["sample_id"].astype(str).ne(str(sample_id))]],
            inputs.cells.loc[inputs.cells["sample_id"].astype(str).ne(str(sample_id))],
            inputs.lr_pairs,
            inputs.contacts,
            [group.name for group in config.ambiguity_groups],
            config.cci.transform,
            max(1, len(inputs.lr_pairs)),
        )
        effects = selected["effect"].tolist() if "effect" in selected.columns else []
        ranks = {}
        for rank in (1, 25, 50):
            if rank <= len(effects):
                ranks[str(rank)] = float(effects[rank - 1])
        rows.append(
            {
                "fold": str(sample_id),
                "contact_pairs": fold["n_contact_pairs"],
                "noncontact_pairs": fold["n_noncontact_pairs"],
                "effect_positive_pairs": int((positive.table["effect"] > 0).sum()),
                "selected_pairs": len(selected),
                "effect_ranks": ranks,
            }
        )
    return rows


def main() -> None:
    config_path = Path(__file__).with_name("config_tintori16.yaml")
    started = time.perf_counter()
    config = chacam.load_config(config_path)
    result = chacam.run(config)
    elapsed = time.perf_counter() - started
    inputs = chacam.load_inputs(config)
    sample_ids = list(dict.fromkeys(inputs.cells["sample_id"].astype(str).tolist()))

    predictions = result.predictions
    print("status counts by fold:")
    print(predictions.groupby(["fold", "status"], sort=False).size().to_string())
    print("status counts summary:")
    print(predictions["status"].value_counts(sort=False).to_string())
    print("unresolved reasons:")
    reasons = predictions.loc[predictions["unresolved_reason"].astype(str) != ""]
    print(reasons["unresolved_reason"].value_counts(sort=False).to_string() if not reasons.empty else "<none>")

    metrics = json.loads((config.output_dir / "metrics.json").read_text(encoding="utf-8"))
    print("metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    reference_check = _reference_check(predictions, inputs, config, sample_ids)
    reference_path = config.output_dir.parent / "c_elegans_tintori16_reference_check.csv"
    reference_check.to_csv(reference_path, index=False, lineterminator="\n")
    print("reference check:")
    print(reference_check.to_string(index=False))

    ms_c = _ms_c_stage1(predictions, inputs, sample_ids)
    print("MS/C first-stage assignments:")
    print(ms_c.to_string(index=False) if not ms_c.empty else "<none>")

    print("top-k diagnostics:")
    print(json.dumps(_topk_diagnostics(config, inputs, sample_ids), ensure_ascii=False, indent=2, sort_keys=True))
    print("output files:")
    print("\n".join(str(path.relative_to(config.output_dir)) for path in sorted(config.output_dir.rglob("*")) if path.is_file()))
    print(f"elapsed_seconds: {elapsed:.6f}")
    print(f"reference_check: {reference_path}")


if __name__ == "__main__":
    main()

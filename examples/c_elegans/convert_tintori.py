"""Convert Tintori et al. 16-cell data to the CHACAM input contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTITIES = [
    "ABala",
    "ABalp",
    "ABara",
    "ABarp",
    "ABpla",
    "ABplp",
    "ABpra",
    "ABprp",
    "MSa",
    "MSp",
    "Ca",
    "Cp",
    "Ea",
    "Ep",
    "D",
    "P4",
]
RAW_LABELS = {
    "ABalx",
    "ABarx",
    "ABplx",
    "ABprx",
    "Ea",
    "Ep",
    "D",
    "P4",
    "MSx1",
    "MSx2",
    "Cx1",
    "Cx2",
}
SAMPLE_PATTERN = re.compile(r"^(st\d+)_(.+)_(\d+)-cell_r(\d+)$")
GENE_QUERY_COLUMNS = ("Your Input", "Public Name", "Sequence Name")
CONTACT_COLUMNS = ["identity_a", "identity_b", "contact", "source"]
CASEFOLD_ALIAS_PREFIX = "\x00casefold:"


def _json_value(value: Any) -> Any:
    """Make pandas/numpy scalar values stable and JSON serializable."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _display_identifier(value: Any) -> Any:
    value = _json_value(value)
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def parse_sample_name(sample: Any) -> dict[str, Any]:
    """Parse a Tintori sample label into the fields used by ``cells.csv``."""

    if not isinstance(sample, str):
        raise ValueError(f"invalid sample name {sample!r}: sample name must be a string")
    match = SAMPLE_PATTERN.fullmatch(sample)
    if match is None:
        raise ValueError(f"invalid sample name {sample!r}")
    cell_id, raw_label, stage, replicate = match.groups()
    return {
        "cell_id": cell_id,
        "raw_label": raw_label,
        "stage": int(stage),
        "sample_id": f"r{replicate}",
    }


def select_samples(sample_table: pd.DataFrame) -> pd.DataFrame:
    """Select usable 16-cell samples in their original Table S1 order."""

    required = {"sample", "Usable Quality?"}
    missing = required.difference(sample_table.columns)
    if missing:
        raise ValueError(f"Table S1 is missing columns: {', '.join(sorted(missing))}")

    parsed_rows = [parse_sample_name(value) for value in sample_table["sample"].tolist()]
    selected_rows = []
    for source_row, parsed in zip(sample_table.to_dict(orient="records"), parsed_rows):
        if str(source_row["Usable Quality?"]) != "Yes" or parsed["stage"] != 16:
            continue
        if parsed["raw_label"] not in RAW_LABELS:
            raise ValueError(f"raw label {parsed['raw_label']!r} is outside the M4 16-cell label set")
        selected_rows.append(parsed)
    selected = pd.DataFrame(selected_rows, columns=["cell_id", "sample_id", "raw_label", "stage"])
    selected.insert(3, "label", selected["raw_label"])
    return selected


def build_gene_alias_map(gene_queries: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build the v2 alias-to-WormBase map and WormBase-to-public-name map."""

    required = {"WormBase Gene ID", *GENE_QUERY_COLUMNS}
    missing = required.difference(gene_queries.columns)
    if missing:
        raise ValueError(f"gene query table is missing columns: {', '.join(sorted(missing))}")
    aliases: dict[str, set[str]] = {}
    public_names: dict[str, str] = {}
    for row in gene_queries.to_dict(orient="records"):
        wormbase_id = row["WormBase Gene ID"]
        if not isinstance(wormbase_id, str) or not wormbase_id.startswith("WBGene"):
            continue
        public_name = row["Public Name"]
        if not isinstance(public_name, str) or not public_name.strip():
            continue
        public_names[wormbase_id] = public_name.strip()
        for column in GENE_QUERY_COLUMNS:
            alias = row[column]
            if isinstance(alias, str) and alias.strip():
                alias = alias.strip()
                aliases.setdefault(alias, set()).add(wormbase_id)
                aliases.setdefault(f"{CASEFOLD_ALIAS_PREFIX}{alias.lower()}", set()).add(wormbase_id)
    return aliases, public_names


def map_gene_name(
    name: Any,
    aliases: dict[str, set[str]],
    public_names: dict[str, str],
) -> tuple[str, str] | None:
    """Map one name by exact alias first, then lowercase alias, if unique."""

    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    candidates = aliases.get(stripped)
    if candidates is None:
        candidates = aliases.get(f"{CASEFOLD_ALIAS_PREFIX}{stripped.lower()}")
    if candidates is None or len(candidates) != 1:
        return None
    wormbase_id = next(iter(candidates))
    public_name = public_names.get(wormbase_id)
    if public_name is None:
        return None
    return wormbase_id, public_name


def _map_expression_rows(
    expression_table: pd.DataFrame,
    gene_queries: pd.DataFrame,
    cell_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "transcript" not in expression_table.columns:
        raise ValueError("Table S2 must contain a transcript column")
    missing_cells = [cell_id for cell_id in cell_ids if cell_id not in expression_table.columns]
    if missing_cells:
        raise ValueError(f"Table S2 is missing selected cells: {', '.join(missing_cells)}")

    aliases, public_names = build_gene_alias_map(gene_queries)
    report: dict[str, Any] = {
        "raw_rows": int(len(expression_table)),
        "mapped": 0,
        "renamed": 0,
        "unmapped": 0,
        "duplicate_id_dropped": 0,
        "dropped_rows": [],
    }
    mapped_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(expression_table.to_dict(orient="records")):
        original = row["transcript"]
        mapped = map_gene_name(original, aliases, public_names)
        if mapped is None:
            report["unmapped"] += 1
            report["dropped_rows"].append(
                {
                    "row": row_number,
                    "original_identifier": _display_identifier(original),
                    "reason": "unmapped_or_ambiguous_name",
                }
            )
            continue
        wormbase_id, public_name = mapped
        original_stripped = original.strip() if isinstance(original, str) else None
        if original_stripped != public_name:
            report["renamed"] += 1
        report["mapped"] += 1
        mapped_rows.append(
            {
                "row": row_number,
                "original_identifier": _display_identifier(original),
                "original_name": original_stripped,
                "wormbase_id": wormbase_id,
                "gene": public_name,
                "values": [row[cell_id] for cell_id in cell_ids],
            }
        )

    keep_rows: list[dict[str, Any]] = []
    for wormbase_id, group in _group_records(mapped_rows, "wormbase_id"):
        if len(group) == 1:
            keep_rows.append(group[0])
            continue
        preferred = [record for record in group if record["original_name"] == record["gene"]]
        if len(preferred) != 1:
            raise ValueError(
                f"two rows map to {wormbase_id} without exactly one original public-name row"
            )
        keep_rows.append(preferred[0])
        for record in group:
            if record is preferred[0]:
                continue
            report["duplicate_id_dropped"] += 1
            report["dropped_rows"].append(
                {
                    "row": record["row"],
                    "original_identifier": record["original_identifier"],
                    "reason": "duplicate_wormbase_id",
                }
            )

    keep_rows.sort(key=lambda record: record["row"])
    final_names = [record["gene"] for record in keep_rows]
    if len(final_names) != len(set(final_names)):
        raise ValueError("final gene names are not unique")
    report["mapped_renamed"] = report["renamed"]
    report["renamed"] = sum(record["original_name"] != record["gene"] for record in keep_rows)
    report["final_genes"] = len(keep_rows)

    values = []
    for record in keep_rows:
        numeric = pd.to_numeric(pd.Series(record["values"]), errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite expression value in gene {record['gene']}")
        if (numeric < 0).any():
            raise ValueError(f"negative expression value in gene {record['gene']}")
        values.append(numeric.to_numpy(dtype=float))
    matrix = pd.DataFrame(values, index=final_names, columns=cell_ids).T
    matrix.insert(0, "cell_id", matrix.index.astype(str))
    matrix = matrix.reset_index(drop=True)
    return matrix, report


def convert_expression_table(
    expression_table: pd.DataFrame,
    gene_queries: pd.DataFrame,
    cell_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map Table S2 and return cell rows with ``log1p(RPKM)`` values."""

    matrix, report = _map_expression_rows(expression_table, gene_queries, cell_ids)
    genes = [column for column in matrix.columns if column != "cell_id"]
    matrix[genes] = np.log1p(matrix[genes].to_numpy(dtype=float))
    return matrix, report


def _group_records(records: list[dict[str, Any]], key: str):
    groups: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    for record in records:
        value = record[key]
        if value not in groups:
            groups[value] = []
            order.append(value)
        groups[value].append(record)
    for value in order:
        yield value, groups[value]


def classify_ms_c_group(family: str, values: dict[str, float]) -> tuple[dict[str, str], str]:
    """Classify one embryo-by-family group from raw F19F10.1 RPKM values."""

    if len(values) != 2:
        return {}, f"undetermined_{family}_group_size_or_threshold"
    ordered = sorted(values, key=lambda cell_id: (values[cell_id], cell_id))
    lo_cell, hi_cell = ordered[0], ordered[1]
    lo, hi = values[lo_cell], values[hi_cell]
    if hi < 25 or hi < 10 * (lo + 1):
        return {}, f"undetermined_{family}_group_size_or_threshold"
    return {lo_cell: f"{family}a", hi_cell: f"{family}p"}, ""


def _assign_ms_c_labels(
    selected: pd.DataFrame,
    raw_expression: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = selected.copy()
    has_ms_c = bool(selected["raw_label"].astype(str).str.startswith(("MSx", "Cx")).any())
    if has_ms_c:
        if "F19F10.1" not in raw_expression.columns:
            raise ValueError("Table S2 has no mapped F19F10.1 row for MS/C classification")
    if not has_ms_c:
        kept = selected.copy()
        kept["label"] = kept["raw_label"]
        return kept[["cell_id", "sample_id", "label", "stage"]].reset_index(drop=True), {
            "ms_c_decisions": [],
            "dropped_cells": [],
        }
    raw_values = raw_expression.set_index("cell_id")["F19F10.1"]
    labels = dict(zip(selected["cell_id"], selected["raw_label"]))
    decisions: list[dict[str, Any]] = []
    dropped_cells: list[dict[str, Any]] = []

    family_rows = selected.loc[selected["raw_label"].astype(str).str.startswith(("MSx", "Cx"))].copy()
    family_rows["family"] = family_rows["raw_label"].str[0].map({"M": "MS", "C": "C"})
    for (sample_id, family), group in family_rows.groupby(["sample_id", "family"], sort=False):
        cell_ids = group["cell_id"].tolist()
        values = {cell_id: float(raw_values.loc[cell_id]) for cell_id in cell_ids}
        classified, group_reason = classify_ms_c_group(family, values)
        for cell_id in cell_ids:
            value = values[cell_id]
            if cell_id in classified:
                label = classified[cell_id]
                labels[cell_id] = label
                result = label
                reason = ""
            else:
                result = None
                reason = group_reason
                dropped_cells.append(
                    {
                        "cell_id": cell_id,
                        "sample_id": sample_id,
                        "family": family,
                        "f19f10_1": value,
                        "reason": reason,
                    }
                )
            decisions.append(
                {
                    "cell_id": cell_id,
                    "sample_id": sample_id,
                    "family": family,
                    "f19f10_1": value,
                    "result": result,
                    "reason": reason,
                }
            )

    dropped_ids = {item["cell_id"] for item in dropped_cells}
    kept = selected.loc[~selected["cell_id"].isin(dropped_ids)].copy()
    kept["label"] = kept["cell_id"].map(labels)
    kept = kept[["cell_id", "sample_id", "label", "stage"]].reset_index(drop=True)
    return kept, {"ms_c_decisions": decisions, "dropped_cells": dropped_cells}


def convert_features(
    feature_table: pd.DataFrame,
    aliases: dict[str, set[str]],
    public_names: dict[str, str],
    expression_genes: set[str],
) -> pd.DataFrame:
    if not {"gene", "source"}.issubset(feature_table.columns):
        raise ValueError("feature table must contain gene and source columns")
    rows = []
    for row in feature_table[["gene", "source"]].to_dict(orient="records"):
        mapped = map_gene_name(row["gene"], aliases, public_names)
        if mapped is None:
            raise ValueError(f"feature gene {row['gene']!r} is not in the gene query table")
        gene = mapped[1]
        if gene not in expression_genes:
            raise ValueError(f"mapped feature gene {gene!r} is missing from expression")
        rows.append({"gene": gene})
    result = pd.DataFrame(rows, columns=["gene"])
    if result["gene"].duplicated().any():
        raise ValueError("mapped feature genes are not unique")
    return result


def convert_lr_pairs(
    lr_table: pd.DataFrame,
    aliases: dict[str, set[str]],
    public_names: dict[str, str],
    expression_genes: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"ligand", "receptor", "annotation"}
    missing = required.difference(lr_table.columns)
    if missing:
        raise ValueError(f"L-R table is missing columns: {', '.join(sorted(missing))}")
    extra_columns = [column for column in lr_table.columns if column not in {"ligand", "receptor", "weight"}]
    columns = ["ligand", "receptor", *extra_columns]
    output_rows = []
    report: dict[str, Any] = {
        "raw_rows": int(len(lr_table)),
        "non_lr": 0,
        "unmapped": 0,
        "dropped_rows": [],
    }
    for row_number, row in enumerate(lr_table.to_dict(orient="records")):
        annotation = row.get("annotation")
        annotation_text = "" if pd.isna(annotation) else str(annotation)
        if re.search(r"non-lr|non-ligand-receptor", annotation_text, flags=re.IGNORECASE):
            report["non_lr"] += 1
            report["dropped_rows"].append({"row": row_number, "reason": "non_lr_annotation"})
            continue
        ligand = map_gene_name(row["ligand"], aliases, public_names)
        receptor = map_gene_name(row["receptor"], aliases, public_names)
        if ligand is None or receptor is None:
            report["unmapped"] += 1
            report["dropped_rows"].append(
                {
                    "row": row_number,
                    "ligand": _display_identifier(row["ligand"]),
                    "receptor": _display_identifier(row["receptor"]),
                    "reason": "unmapped_ligand_or_receptor",
                }
            )
            continue
        output = {"ligand": ligand[1], "receptor": receptor[1]}
        output.update({column: row[column] for column in extra_columns})
        output_rows.append(output)
    report["final_rows"] = len(output_rows)
    report["mapped_but_missing_from_expression"] = [
        {
            "row": index,
            "ligand": row["ligand"],
            "receptor": row["receptor"],
        }
        for index, row in enumerate(output_rows)
        if row["ligand"] not in expression_genes or row["receptor"] not in expression_genes
    ]
    return pd.DataFrame(output_rows, columns=columns), report


def _normalize_contact_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().rstrip("'").strip()
    return value or None


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError(f"self-contact is not valid for identity {first}")
    return tuple(sorted((first, second)))


def convert_contacts(
    contact_tables: dict[int, pd.DataFrame],
    contact_overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert CShaper levels and M4 assumptions to contact tables."""

    known = set(IDENTITIES)
    levels: dict[tuple[str, str], int] = {}
    for level, frame in contact_tables.items():
        if frame.shape[1] < 2:
            raise ValueError(f"CShaper sheet for level {level} needs two columns")
        for first_raw, second_raw in frame.iloc[:, :2].itertuples(index=False, name=None):
            first = _normalize_contact_name(first_raw)
            second = _normalize_contact_name(second_raw)
            if first is None or second is None:
                continue
            pair = _canonical_pair(first, second)
            if first not in known or second not in known:
                continue
            levels[pair] = max(levels.get(pair, 0), int(level))

    required = set(CONTACT_COLUMNS)
    if not required.issubset(contact_overrides.columns):
        missing = required.difference(contact_overrides.columns)
        raise ValueError(f"contact overrides are missing columns: {', '.join(sorted(missing))}")
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for row in contact_overrides.to_dict(orient="records"):
        first = str(row["identity_a"]).strip()
        second = str(row["identity_b"]).strip()
        if first not in known or second not in known:
            raise ValueError(f"unknown identity in contact override: {first}, {second}")
        pair = _canonical_pair(first, second)
        if pair in overrides:
            raise ValueError(f"duplicate contact override for pair {pair[0]} and {pair[1]}")
        contact = pd.to_numeric(row["contact"], errors="coerce")
        if pd.isna(contact) or float(contact) not in {0.0, 1.0}:
            raise ValueError(f"contact override must be 0 or 1 for pair {pair[0]} and {pair[1]}")
        overrides[pair] = {"contact": int(contact), "source": f"override: {row['source']}"}

    all_pairs = sorted(combinations(sorted(IDENTITIES), 2))
    level_rows = [{"identity_a": first, "identity_b": second, "level": levels.get((first, second), 0)} for first, second in all_pairs]
    contact_rows = []
    for first, second in all_pairs:
        pair = (first, second)
        if pair in overrides:
            contact_rows.append(
                {
                    "identity_a": first,
                    "identity_b": second,
                    **overrides[pair],
                }
            )
        elif levels.get(pair, 0) == 4:
            contact_rows.append(
                {
                    "identity_a": first,
                    "identity_b": second,
                    "contact": 1,
                    "source": "cshaper_sheet4",
                }
            )
        elif levels.get(pair, 0) == 0:
            contact_rows.append(
                {
                    "identity_a": first,
                    "identity_b": second,
                    "contact": 0,
                    "source": "cshaper_absent",
                }
            )
    return pd.DataFrame(level_rows, columns=["identity_a", "identity_b", "level"]), pd.DataFrame(contact_rows, columns=["identity_a", "identity_b", "contact", "source"])


def convert_tables(
    *,
    sample_table: pd.DataFrame,
    expression_table: pd.DataFrame,
    gene_queries: pd.DataFrame,
    lr_table: pd.DataFrame,
    contact_tables: dict[int, pd.DataFrame],
    feature_table: pd.DataFrame,
    contact_overrides: pd.DataFrame,
) -> dict[str, Any]:
    """Run all deterministic M4 transformations on in-memory tables."""

    selected = select_samples(sample_table)
    selected_ids = selected["cell_id"].tolist()
    raw_expression, gene_report = _map_expression_rows(expression_table, gene_queries, selected_ids)
    cells, ms_c_report = _assign_ms_c_labels(selected, raw_expression)
    kept_ids = cells["cell_id"].tolist()
    genes = [column for column in raw_expression.columns if column != "cell_id"]
    expression = raw_expression.loc[raw_expression["cell_id"].isin(kept_ids)].copy()
    expression = expression.set_index("cell_id").loc[kept_ids].reset_index()
    expression[genes] = np.log1p(expression[genes].to_numpy(dtype=float))

    aliases, public_names = build_gene_alias_map(gene_queries)
    features = convert_features(feature_table, aliases, public_names, set(genes))
    lr_pairs, lr_report = convert_lr_pairs(lr_table, aliases, public_names, set(genes))
    contact_levels, contacts = convert_contacts(contact_tables, contact_overrides)

    report = {
        "selected_cell_count": len(selected_ids),
        "kept_cell_count": len(kept_ids),
        "dropped_cells": ms_c_report["dropped_cells"],
        "ms_c_decisions": ms_c_report["ms_c_decisions"],
        "gene_mapping": gene_report,
        "lr_pairs": lr_report,
        "contacts": {
            "cshaper_level_counts": {
                str(level): int((contact_levels["level"] == level).sum()) for level in range(5)
            },
            "output_contact_counts": {
                str(value): int((contacts["contact"] == value).sum()) for value in sorted(contacts["contact"].unique())
            },
            "overrides": contact_overrides.to_dict(orient="records"),
        },
    }
    return {
        "expression": expression[["cell_id", *genes]],
        "cells": cells,
        "features": features,
        "lr_pairs": lr_pairs,
        "contacts": contacts,
        "contact_levels": contact_levels,
        "report": report,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _source_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def write_conversion_outputs(
    result: dict[str, Any],
    output_dir: Path,
    raw_paths: dict[str, Path],
    script_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_names = ("expression", "cells", "features", "lr_pairs", "contacts", "contact_levels")
    derived_paths = {}
    for name in csv_names:
        path = output_dir / f"{name}.csv"
        _write_csv(result[name], path)
        derived_paths[name] = path

    repo_root = script_path.resolve().parents[2]
    commit, dirty = _git_provenance(repo_root)
    report = dict(result["report"])
    report["source_files_sha256"] = {
        name: {"path": _source_path(path, repo_root), "sha256": _sha256(path)} for name, path in raw_paths.items()
    }
    report["derived_files_sha256"] = {
        name: _sha256(path) for name, path in derived_paths.items()
    }
    report["conversion_script_commit"] = commit
    report["conversion_script_dirty"] = dirty
    report_path = output_dir / "conversion_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_dir = repo_root / "data" / "reference"
    feature_path = Path(__file__).resolve().parent / "features_tintori16.csv"
    override_path = Path(__file__).resolve().parent / "contact_overrides.csv"

    s1_path = reference_dir / "Table S1 Summary of All 219 Samples Sequenced Related to Experimental Procedures.xls"
    s2_path = reference_dir / "Table S2_RPKMs_Normalized Expression Levels for All Genes in All Samples Related to Figure 1.xls"
    query_path = reference_dir / "Gene queries of C elegans.xlsx"
    lr_path = reference_dir / "ligand_receptor_updated_C-elegans.csv"
    contact_path = reference_dir / "Supplementary Data 5_Identified Cell-Cell Contacts.xlsx"
    raw_paths = {
        "table_s1": s1_path,
        "table_s2": s2_path,
        "gene_queries": query_path,
        "ligand_receptor": lr_path,
        "contacts": contact_path,
    }
    sample_table = pd.read_excel(s1_path, sheet_name=0, header=0)
    expression_table = pd.read_excel(s2_path, sheet_name=0, header=0)
    gene_queries = pd.read_excel(query_path, sheet_name=0, header=0)
    lr_table = pd.read_csv(lr_path, skiprows=1, header=0)
    contact_tables = {
        level: pd.read_excel(contact_path, sheet_name=f"Sheet{level}", header=None, skiprows=2, usecols=[0, 1])
        for level in range(1, 5)
    }
    result = convert_tables(
        sample_table=sample_table,
        expression_table=expression_table,
        gene_queries=gene_queries,
        lr_table=lr_table,
        contact_tables=contact_tables,
        feature_table=pd.read_csv(feature_path),
        contact_overrides=pd.read_csv(override_path),
    )
    output_dir = reference_dir / "derived"
    report = write_conversion_outputs(result, output_dir, raw_paths, Path(__file__))
    print(json.dumps({
        "cells": len(result["cells"]),
        "genes": len(result["expression"].columns) - 1,
        "lr_pairs": len(result["lr_pairs"]),
        "contacts": len(result["contacts"]),
        "contact_levels": len(result["contact_levels"]),
        "conversion_script_commit": report["conversion_script_commit"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

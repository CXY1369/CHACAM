"""Convert Cole-Yanai early C. elegans data to M6 derived tables."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


STAGES = (15, 28, 51)
CPM_SCALE = 1_000_000.0
MISALIGNED_EMBRYOS = ("4c3", "4c5", "7c1", "8c1", "8c2")
MISALIGNED_CELL_COUNT = 25
EXPECTED_STAGE_COUNTS = {15: (5, 59), 28: (6, 132), 51: (5, 204)}
SPECIAL_IDENTITIES = {
    "MSxaap": ("MSaaap", "MSaapp", "MSpaap", "MSpapp"),
    "MSxpaa": ("MSapaa", "MSapap", "MSppaa", "MSppap"),
}
STATE_CORRECTIONS = {"Abpxpaa": "ABpxpaa", "P0": "P"}


def parse_s4_state(value: Any) -> tuple[tuple[str, ...], int | None]:
    """Parse one Table S4 identity cell before splitting it into identities."""

    text = str(value).strip()
    stage: int | None = None
    parenthetical = re.search(r"\s*\((\d+)-cell stage\)\s*$", text)
    if parenthetical:
        stage = int(parenthetical.group(1))
        text = text[: parenthetical.start()].strip()
    suffix = re.search(r"_(\d+)$", text)
    if suffix:
        if stage is not None:
            raise ValueError(f"Table S4 state has two stage annotations: {value!r}")
        stage = int(suffix.group(1))
        text = text[: suffix.start()].strip()
    identities = tuple(text.split())
    if not identities:
        raise ValueError(f"Table S4 state has no identities: {value!r}")
    return identities, stage


def parse_s5_state(value: Any) -> tuple[str, int | None]:
    """Normalize one Table S5 state name and return its optional stage."""

    text = str(value).strip()
    stage: int | None = None
    parenthetical = re.search(r"\s*\((\d+)-cell stage\)\s*$", text)
    if parenthetical:
        stage = int(parenthetical.group(1))
        text = text[: parenthetical.start()].strip()
    else:
        suffix = re.search(r"\s+(\d+)$", text)
        if suffix:
            stage = int(suffix.group(1))
            text = text[: suffix.start()].strip()
    if not text:
        raise ValueError(f"Table S5 state has no name: {value!r}")
    return text, stage


def _state_pattern_matches(pattern: str, identity: str) -> bool:
    if len(pattern) != len(identity):
        return False
    return all(expected == actual or (expected == "x" and actual in "aplr") for expected, actual in zip(pattern, identity))


def _state_correction(name: str) -> str:
    return STATE_CORRECTIONS.get(name, name)


def match_s5_state(value: Any, s4_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Match a Table S5 state to exactly one parsed Table S4 row."""

    name, stage = parse_s5_state(value)
    if name == "P0" and stage is None:
        stage = 0
    corrected_name = _state_correction(name)
    if corrected_name in SPECIAL_IDENTITIES:
        target = set(SPECIAL_IDENTITIES[corrected_name])
        candidates = [row for row in s4_rows if set(row["identities"]) == target]
    elif "x" in corrected_name:
        candidates = [
            row
            for row in s4_rows
            if len(row["identities"]) >= 2
            and all(_state_pattern_matches(corrected_name, identity) for identity in row["identities"])
        ]
    else:
        lookup_name = corrected_name
        candidates = [row for row in s4_rows if row["identities"][0] == lookup_name]

    if stage is not None:
        stage_candidates = [row for row in candidates if row.get("stage") == stage]
        if stage_candidates:
            candidates = stage_candidates
    if len(candidates) != 1:
        raise ValueError(
            f"Table S5 state {value!r} must match exactly one Table S4 row; "
            f"matched {len(candidates)}"
        )

    row = candidates[0]
    identities = tuple(row["identities"])
    label_kind = "fine" if len(identities) == 1 else "equivalence"
    label = identities[0] if label_kind == "fine" else corrected_name
    return {
        "state_id": str(row["state_id"]),
        "s4_state": " ".join(identities),
        "identities": identities,
        "label": label,
        "label_kind": label_kind,
        "normalized_name": corrected_name,
        "stage": stage,
    }


def validate_cell_counts(
    s4_rows: Sequence[Mapping[str, Any]], matched_s4_states: Iterable[str]
) -> dict[str, Any]:
    """Check Table S5 counts against Table S4, preserving the registered MS exception."""

    observed = Counter(str(state_id) for state_id in matched_s4_states)
    exceptions: list[dict[str, Any]] = []
    for row in s4_rows:
        state_id = str(row["state_id"])
        declared = int(row["declared_count"])
        actual = int(observed.get(state_id, 0))
        if actual == declared:
            continue
        if (
            state_id == "20"
            and tuple(row["identities"]) == ("MSa", "MSp")
            and declared == 12
            and actual == 6
        ):
            exceptions.append({"state_id": state_id, "declared": declared, "observed": actual})
            continue
        raise ValueError(
            f"Table S4 cell count mismatch for state {state_id}: declared {declared}, observed {actual}"
        )
    return {"exceptions": exceptions}


def normalize_embryo_id(value: Any) -> str:
    """Map Table S5 embryo identifiers to Table S1 identifiers."""

    text = str(value).strip()
    p0 = re.fullmatch(r"P0(\d+)", text)
    if p0:
        return f"1.{int(p0.group(1))}"
    cell = re.fullmatch(r"(\d+)c(\d*)", text)
    if cell:
        number, replicate = cell.groups()
        return f"{number}.{int(replicate)}" if replicate else number
    raise ValueError(f"invalid embryo ID {value!r}")


def validate_soft_titles(s5: pd.DataFrame, soft_titles: Mapping[str, str]) -> None:
    """Require every Table S5 GEO accession to have the same SOFT title."""

    for row in s5.to_dict(orient="records"):
        accession = str(row["GEO entry"])
        expected = str(row["sample name"])
        observed = soft_titles.get(accession)
        if observed != expected:
            raise ValueError(
                f"SOFT sample title mismatch for {accession}: Table S5 {expected!r}, SOFT {observed!r}"
            )


def filter_misaligned_cells(s5: pd.DataFrame, raw_counts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the closed transcript-count ratio interval and remove whole embryos."""

    out_of_range: list[str] = []
    details: list[dict[str, Any]] = []
    ratios: dict[str, float] = {}
    for row in s5.to_dict(orient="records"):
        cell_id = str(row["GEO entry"])
        if cell_id not in raw_counts.columns:
            raise ValueError(f"raw count matrix is missing {cell_id}")
        library_size = float(pd.to_numeric(raw_counts[cell_id], errors="raise").sum())
        if library_size <= 0:
            raise ValueError(f"raw count matrix has empty library for {cell_id}")
        ratio = float(row["Number of transcripts"]) / library_size
        ratios[cell_id] = ratio
        if ratio < 0.95 or ratio > 1.06:
            out_of_range.append(cell_id)
            details.append(
                {
                    "cell_id": cell_id,
                    "sample_name": str(row["sample name"]),
                    "embryo_id": str(row["Embryo ID"]),
                    "ratio": ratio,
                }
            )
    removed_embryos = list(dict.fromkeys(str(row["Embryo ID"]) for row in s5.to_dict(orient="records") if str(row["GEO entry"]) in out_of_range))
    kept = s5.loc[~s5["Embryo ID"].astype(str).isin(removed_embryos)].copy()
    report = {
        "out_of_range_cells": out_of_range,
        "out_of_range_details": details,
        "removed_embryos": removed_embryos,
        "ratios": ratios,
    }
    return kept.reset_index(drop=True), report


def to_log1p_cpm(
    raw_counts: pd.DataFrame,
    cell_ids: Sequence[str],
    *,
    library_sizes: pd.Series | None = None,
) -> pd.DataFrame:
    """Return cell rows in log1p(CPM), using all supplied raw rows for library size."""

    cell_ids = [str(cell_id) for cell_id in cell_ids]
    missing = [cell_id for cell_id in cell_ids if cell_id not in raw_counts.columns]
    if missing:
        raise ValueError(f"raw count matrix is missing cells: {', '.join(missing)}")
    numeric = raw_counts.loc[:, cell_ids].apply(pd.to_numeric, errors="raise")
    if (numeric < 0).any().any():
        raise ValueError("raw count matrix contains negative values")
    if library_sizes is None:
        library_sizes = numeric.sum(axis=0)
    library_sizes = pd.to_numeric(library_sizes.loc[cell_ids], errors="raise")
    if (library_sizes <= 0).any():
        raise ValueError("library sizes must be positive")
    result = np.log1p(numeric.div(library_sizes, axis="columns") * CPM_SCALE).T
    result.index = result.index.astype(str)
    result.index.name = "cell_id"
    return result


def _normalize_contact_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().rstrip("'").strip()
    return text or None


def convert_cole_contacts(
    contact_tables: Mapping[int, pd.DataFrame], fine_identities: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Convert CShaper levels for the fine identities in one developmental stage."""

    identities = sorted(set(str(identity) for identity in fine_identities))
    observed_names: set[str] = set()
    levels: dict[tuple[str, str], int] = {}
    for level, frame in contact_tables.items():
        if frame.shape[1] < 2:
            raise ValueError(f"CShaper sheet {level} needs two columns")
        for first_raw, second_raw in frame.iloc[:, :2].itertuples(index=False, name=None):
            first = _normalize_contact_name(first_raw)
            second = _normalize_contact_name(second_raw)
            if first is None or second is None or first == second:
                continue
            observed_names.update((first, second))
            pair = tuple(sorted((first, second)))
            levels[pair] = max(levels.get(pair, 0), int(level))

    missing = sorted(set(identities) - observed_names)
    level_rows = []
    contact_rows = []
    for first, second in combinations(identities, 2):
        level = int(levels.get((first, second), 0))
        level_rows.append({"identity_a": first, "identity_b": second, "level": level})
        if first in missing or second in missing:
            continue
        if level == 4:
            contact_rows.append(
                {"identity_a": first, "identity_b": second, "contact": 1, "source": "cshaper_sheet4"}
            )
        elif level == 0:
            contact_rows.append(
                {"identity_a": first, "identity_b": second, "contact": 0, "source": "cshaper_absent"}
            )
    return (
        pd.DataFrame(level_rows, columns=["identity_a", "identity_b", "level"]),
        pd.DataFrame(contact_rows, columns=["identity_a", "identity_b", "contact", "source"]),
        missing,
    )


def _docx_table(path: Path) -> list[list[str]]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables = root.findall(".//w:tbl", namespace)
    if len(tables) != 1:
        raise ValueError(f"expected one table in {path}, found {len(tables)}")
    rows: list[list[str]] = []
    for tr in tables[0].findall("./w:tr", namespace):
        cells = []
        for tc in tr.findall("./w:tc", namespace):
            cells.append("".join(text.text or "" for text in tc.findall(".//w:t", namespace)).strip())
        rows.append(cells)
    return rows


def read_s4(path: Path) -> list[dict[str, Any]]:
    rows = _docx_table(path)
    if not rows or rows[0][:3] != ["Cell-state", "Cell identities", "# cells"]:
        raise ValueError("unexpected Table S4 header")
    parsed = []
    for row in rows[1:]:
        identities, stage = parse_s4_state(row[1])
        parsed.append(
            {
                "state_id": str(row[0]),
                "identities": identities,
                "declared_count": int(row[2]),
                "stage": stage,
            }
        )
    return parsed


def read_s1(path: Path) -> dict[str, dict[str, Any]]:
    rows = _docx_table(path)
    if not rows or rows[0][:5] != ["Embryo ID", "# Cells Collected", "Intended Stage", "# Transcripts", "# Genes"]:
        raise ValueError("unexpected Table S1 header")
    result = {}
    for row in rows[1:]:
        embryo_id = str(row[0])
        if embryo_id in result:
            raise ValueError(f"duplicate Table S1 embryo {embryo_id}")
        result[embryo_id] = {
            "embryo_id": embryo_id,
            "stage": int(row[2]),
            "transcripts": int(row[3]),
            "cells": int(row[1]),
        }
    return result


def read_s5(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Table S5", header=0)
    required = {"GEO entry", "sample name", "Embryo ID", "Annotated State", "Number of transcripts"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Table S5 is missing columns: {', '.join(sorted(missing))}")
    return frame.copy()


def read_soft_titles(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    current_title: str | None = None
    with gzip_open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("!Sample_title ="):
                current_title = line.split("=", 1)[1].strip().strip("'")
            elif line.startswith("!Sample_geo_accession ="):
                accession = line.split("=", 1)[1].strip()
                if current_title is not None:
                    titles[accession] = current_title
    return titles


def gzip_open_text(path: Path):
    import gzip

    return gzip.open(path, "rt", encoding="utf-8", errors="replace")


def _load_tintori_module():
    path = Path(__file__).resolve().with_name("convert_tintori.py")
    spec = importlib.util.spec_from_file_location("chacam_convert_tintori", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_raw_expression(
    raw_counts: pd.DataFrame,
    gene_queries: pd.DataFrame,
    cell_ids: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    tintori = _load_tintori_module()
    aliases, public_names = tintori.build_gene_alias_map(gene_queries)
    cell_ids = [str(cell_id) for cell_id in cell_ids]
    library_sizes = raw_counts.loc[:, cell_ids].apply(pd.to_numeric, errors="raise").sum(axis=0)
    report: dict[str, Any] = {
        "raw_rows": int(len(raw_counts)),
        "mapped": 0,
        "unmapped": 0,
        "renamed": 0,
        "duplicate_id_dropped": 0,
        "dropped_rows": [],
    }
    mapped: list[dict[str, Any]] = []
    for row_number, (identifier, row) in enumerate(raw_counts.loc[:, cell_ids].iterrows()):
        mapped_name = tintori.map_gene_name(identifier, aliases, public_names)
        if mapped_name is None:
            report["unmapped"] += 1
            report["dropped_rows"].append(
                {"row": row_number, "original_identifier": str(identifier), "reason": "unmapped_or_ambiguous_name"}
            )
            continue
        wormbase_id, public_name = mapped_name
        original = str(identifier).strip()
        report["mapped"] += 1
        if original != public_name:
            report["renamed"] += 1
        mapped.append(
            {
                "row": row_number,
                "wormbase_id": wormbase_id,
                "original_name": original,
                "gene": public_name,
                "values": row.to_numpy(dtype=float),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in mapped:
        groups[record["wormbase_id"]].append(record)
    kept: list[dict[str, Any]] = []
    for wormbase_id, records in groups.items():
        if len(records) == 1:
            kept.append(records[0])
            continue
        preferred = [record for record in records if record["original_name"] == record["gene"]]
        if len(preferred) != 1:
            raise ValueError(f"two rows map to {wormbase_id} without one public-name row")
        kept.append(preferred[0])
        for record in records:
            if record is not preferred[0]:
                report["duplicate_id_dropped"] += 1
                report["dropped_rows"].append(
                    {"row": record["row"], "original_identifier": record["original_name"], "reason": "duplicate_wormbase_id"}
                )
    kept.sort(key=lambda record: record["row"])
    genes = [record["gene"] for record in kept]
    if len(genes) != len(set(genes)):
        raise ValueError("mapped gene names are not unique")
    matrix = pd.DataFrame([record["values"] for record in kept], index=genes, columns=cell_ids)
    report["final_genes"] = len(genes)
    report["mapped_renamed"] = report["renamed"]
    expression = to_log1p_cpm(matrix, cell_ids, library_sizes=library_sizes)
    return expression, report, library_sizes


def _read_lr_pairs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1, header=0)


def _read_contact_tables(path: Path) -> dict[int, pd.DataFrame]:
    return {
        level: pd.read_excel(path, sheet_name=f"Sheet{level}", header=None, skiprows=2, usecols=[0, 1])
        for level in range(1, 5)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _validate_embryo_mapping(s5: pd.DataFrame, s1: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    frame = s5.copy()
    frame["normalized_embryo_id"] = frame["Embryo ID"].map(normalize_embryo_id)
    counts = frame["normalized_embryo_id"].value_counts()
    expected_counts = counts.index.to_series().map(lambda value: s1.get(value, {}).get("cells", -1))
    if (counts > expected_counts).any():
        raise ValueError("Table S5 cell count exceeds the corresponding Table S1 row")
    missing = sorted(set(s1) - set(frame["normalized_embryo_id"]))
    extra = sorted(set(frame["normalized_embryo_id"]) - set(s1))
    if missing or extra:
        raise ValueError(f"Table S5/Table S1 embryo mismatch; missing={missing}, extra={extra}")
    frame["stage"] = frame["normalized_embryo_id"].map(lambda value: int(s1[value]["stage"]))
    return frame


def convert_inputs(
    *,
    s5: pd.DataFrame,
    s1: Mapping[str, Mapping[str, Any]],
    s4_rows: Sequence[Mapping[str, Any]],
    soft_titles: Mapping[str, str],
    raw_counts: pd.DataFrame,
    gene_queries: pd.DataFrame,
    lr_table: pd.DataFrame,
    contact_tables: Mapping[int, pd.DataFrame],
) -> dict[int, dict[str, Any]]:
    """Perform the deterministic C1 conversion in memory."""

    validate_soft_titles(s5, soft_titles)
    attached = _validate_embryo_mapping(s5, s1)
    matches = [match_s5_state(value, s4_rows) for value in attached["Annotated State"]]
    attached["label"] = [match["label"] for match in matches]
    attached["label_kind"] = [match["label_kind"] for match in matches]
    attached["s4_state"] = [match["s4_state"] for match in matches]
    attached["s4_state_id"] = [match["state_id"] for match in matches]
    count_report = validate_cell_counts(s4_rows, attached["s4_state_id"])
    filtered, misalignment = filter_misaligned_cells(attached, raw_counts)
    if len(misalignment["out_of_range_cells"]) != MISALIGNED_CELL_COUNT or tuple(misalignment["removed_embryos"]) != MISALIGNED_EMBRYOS:
        raise RuntimeError(
            "Cole-Yanai misalignment validation differs from §2.3: "
            f"cells={len(misalignment['out_of_range_cells'])}, embryos={misalignment['removed_embryos']}"
        )

    cell_ids = filtered["GEO entry"].astype(str).tolist()
    expression, gene_report, _library_sizes = _map_raw_expression(raw_counts, gene_queries, cell_ids)
    tintori = _load_tintori_module()
    aliases, public_names = tintori.build_gene_alias_map(gene_queries)
    expression_genes = set(expression.columns)
    lr_pairs, lr_report = tintori.convert_lr_pairs(lr_table, aliases, public_names, expression_genes)
    lr_report["with_both_genes_in_expression"] = int(
        (lr_pairs["ligand"].isin(expression_genes) & lr_pairs["receptor"].isin(expression_genes)).sum()
    ) if not lr_pairs.empty else 0
    lr_pairs = lr_pairs.loc[lr_pairs["ligand"].isin(expression_genes) & lr_pairs["receptor"].isin(expression_genes)].reset_index(drop=True)

    results: dict[int, dict[str, Any]] = {}
    for stage in STAGES:
        stage_cells = filtered.loc[filtered["stage"].eq(stage)].copy()
        expected_embryos, expected_cells = EXPECTED_STAGE_COUNTS[stage]
        if stage_cells["normalized_embryo_id"].nunique() != expected_embryos or len(stage_cells) != expected_cells:
            raise RuntimeError(
                f"Cole-Yanai stage {stage} count differs from §2.4: "
                f"embryos={stage_cells['normalized_embryo_id'].nunique()}, cells={len(stage_cells)}"
            )
        stage_ids = stage_cells["GEO entry"].astype(str).tolist()
        stage_expression = expression.loc[stage_ids].copy()
        fine_identities = stage_cells.loc[stage_cells["label_kind"].eq("fine"), "label"].astype(str).unique().tolist()
        contact_levels, contacts, missing_identities = convert_cole_contacts(contact_tables, fine_identities)
        cells = stage_cells[
            ["GEO entry", "Embryo ID", "sample name", "stage", "label", "label_kind", "s4_state"]
        ].rename(columns={"GEO entry": "cell_id", "Embryo ID": "sample_id", "sample name": "sample_name"})
        cells = cells[["cell_id", "sample_id", "sample_name", "stage", "label", "label_kind", "s4_state"]].reset_index(drop=True)
        stage_expression.insert(0, "cell_id", stage_expression.index)
        stage_expression = stage_expression.reset_index(drop=True)
        report = {
            "stage": stage,
            "selected_cell_count_before_filter": int(len(attached.loc[attached["stage"].eq(stage)])),
            "kept_cell_count": int(len(cells)),
            "embryo_count": int(cells["sample_id"].nunique()),
            "fine_cell_count": int((cells["label_kind"] == "fine").sum()),
            "equivalence_cell_count": int((cells["label_kind"] == "equivalence").sum()),
            "misalignment": misalignment,
            "cell_count_check": count_report,
            "gene_mapping": gene_report,
            "lr_pairs": lr_report,
            "cshaper_missing_identities": missing_identities,
            "notes": {
                "expression": "log1p(CPM) from all raw count rows; no features.csv is emitted",
                "comparability": "Cole-Yanai CPM is not compared in score or accuracy magnitude with Tintori RPKM",
            },
        }
        results[stage] = {
            "cells": cells,
            "expression": stage_expression,
            "lr_pairs": lr_pairs,
            "contacts": contacts,
            "contact_levels": contact_levels,
            "report": report,
        }
    return results


def write_stage_outputs(
    result: Mapping[str, Any],
    output_dir: Path,
    raw_paths: Mapping[str, Path],
    script_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cells", "expression", "lr_pairs", "contacts", "contact_levels"):
        _write_csv(result[name], output_dir / f"{name}.csv")
    report = dict(result["report"])
    repo_root = script_path.resolve().parents[2]
    commit, dirty = _git_provenance(repo_root)
    report["source_files_sha256"] = {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in raw_paths.items()
    }
    report["derived_files_sha256"] = {
        name: _sha256(output_dir / f"{name}.csv")
        for name in ("cells", "expression", "lr_pairs", "contacts", "contact_levels")
    }
    report["conversion_script_commit"] = commit
    report["conversion_script_dirty"] = dirty
    with (output_dir / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reference_dir = repo_root / "data" / "reference"
    raw_dir = reference_dir / "cole_yanai"
    raw_counts_path = raw_dir / "GSE83523_CE_blastomeres_rawcounts.txt.gz"
    soft_path = raw_dir / "GSE83523_family.soft.gz"
    s5_path = raw_dir / "Cell state assignments.xlsx"
    s4_path = raw_dir / "Inferred cell states.docx"
    s1_path = raw_dir / "Characteristics of the dataset.docx"
    query_path = reference_dir / "Gene queries of C elegans.xlsx"
    lr_path = reference_dir / "ligand_receptor_updated_C-elegans.csv"
    contact_path = reference_dir / "Supplementary Data 5_Identified Cell-Cell Contacts.xlsx"
    raw_paths = {
        "raw_counts": raw_counts_path,
        "soft": soft_path,
        "table_s5": s5_path,
        "table_s4": s4_path,
        "table_s1": s1_path,
        "gene_queries": query_path,
        "ligand_receptor": lr_path,
        "contacts": contact_path,
    }
    missing = [str(path) for path in raw_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Cole-Yanai conversion inputs: " + ", ".join(missing))

    s5 = read_s5(s5_path)
    s1 = read_s1(s1_path)
    s4_rows = read_s4(s4_path)
    soft_titles = read_soft_titles(soft_path)
    raw_counts = pd.read_csv(raw_counts_path, sep="\t", index_col=0)
    gene_queries = pd.read_excel(query_path, sheet_name=0, header=0)
    lr_table = _read_lr_pairs(lr_path)
    contact_tables = _read_contact_tables(contact_path)
    results = convert_inputs(
        s5=s5,
        s1=s1,
        s4_rows=s4_rows,
        soft_titles=soft_titles,
        raw_counts=raw_counts,
        gene_queries=gene_queries,
        lr_table=lr_table,
        contact_tables=contact_tables,
    )
    for stage, result in results.items():
        output_dir = reference_dir / "derived_cole_yanai" / f"stage{stage}"
        write_stage_outputs(result, output_dir, raw_paths, Path(__file__))
    print(json.dumps({stage: {"cells": len(result["cells"]), "genes": len(result["expression"].columns) - 1, "lr_pairs": len(result["lr_pairs"])} for stage, result in results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()

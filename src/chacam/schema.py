"""Input contracts and configuration validation for the framework baseline."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class AmbiguityGroup:
    name: str
    candidates: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True)
class InputPaths:
    expression: Path
    cells: Path
    features: Path
    lr_pairs: Path
    contacts: Path
    truth: Path | None
    expression_scale: str | None


@dataclass(frozen=True)
class ReferenceConfig:
    mode: str


@dataclass(frozen=True)
class AttentionConfig:
    normalization: str
    temperature: float | None


@dataclass(frozen=True)
class AssignmentConfig:
    mode: str


@dataclass(frozen=True)
class CCIConfig:
    transform: str
    direction: str
    lr_selection: str
    top_k: int


@dataclass(frozen=True)
class RunConfig:
    provenance: str
    seed: int | None
    inputs: InputPaths
    reference: ReferenceConfig
    attention: AttentionConfig
    assignment: AssignmentConfig
    cci: CCIConfig
    ambiguity_groups: tuple[AmbiguityGroup, ...]
    output_dir: Path
    config_path: Path


@dataclass
class Inputs:
    expression: pd.DataFrame
    cells: pd.DataFrame
    features: pd.DataFrame
    lr_pairs: pd.DataFrame
    contacts: pd.DataFrame
    ambiguity_groups: tuple[AmbiguityGroup, ...]
    used_feature_genes: list[str]
    missing_feature_genes: list[str]
    dropped_lr_pairs: list[dict[str, Any]]


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_mapping(parent: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    if key not in parent:
        raise ValueError(f"missing required configuration section {name}")
    return _as_mapping(parent[key], name)


def _required_value(parent: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in parent or parent[key] is None:
        raise ValueError(f"missing required configuration value {name}")
    return parent[key]


def _enum(value: Any, allowed: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {name} {value!r}; expected one of {choices}")
    return value


def _parse_ambiguity_groups(raw: Any) -> tuple[AmbiguityGroup, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list):
        raise ValueError("ambiguity_groups must be a list")
    groups: list[AmbiguityGroup] = []
    names: set[str] = set()
    for item in raw:
        item = _as_mapping(item, "ambiguity group")
        name = str(_required_value(item, "name", "ambiguity_groups.name"))
        if name in names:
            raise ValueError(f"duplicate ambiguity group {name}")
        names.add(name)
        candidates_raw = _required_value(item, "candidates", f"{name}.candidates")
        references_raw = _required_value(item, "references", f"{name}.references")
        if not isinstance(candidates_raw, list) or len(candidates_raw) != 2:
            raise ValueError(f"ambiguity group {name} must have exactly two candidates")
        if len(set(candidates_raw)) != len(candidates_raw):
            raise ValueError(f"ambiguity group {name} has duplicate candidates")
        if not isinstance(references_raw, list) or not references_raw:
            raise ValueError(f"ambiguity group {name} must have at least one reference")
        if len(set(references_raw)) != len(references_raw):
            raise ValueError(f"ambiguity group {name} has duplicate references")
        groups.append(
            AmbiguityGroup(
                name=name,
                candidates=tuple(str(value) for value in candidates_raw),
                references=tuple(str(value) for value in references_raw),
            )
        )
    return tuple(groups)


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _as_mapping(raw, "configuration")

    provenance = str(raw.get("provenance", "framework_baseline"))
    if provenance != "framework_baseline":
        raise ValueError("provenance must be framework_baseline")
    seed = raw.get("seed", 0)
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer or null")

    inputs_raw = _required_mapping(raw, "inputs", "inputs")
    input_paths: dict[str, Path | None] = {}
    for key in ("expression", "cells", "features", "lr_pairs", "contacts"):
        value = _required_value(inputs_raw, key, f"inputs.{key}")
        input_path = Path(str(value)).expanduser()
        input_paths[key] = (config_path.parent / input_path).resolve() if not input_path.is_absolute() else input_path
    truth_value = inputs_raw.get("truth")
    truth_path = None
    if truth_value is not None:
        truth_candidate = Path(str(truth_value)).expanduser()
        truth_path = (config_path.parent / truth_candidate).resolve() if not truth_candidate.is_absolute() else truth_candidate
    expression_scale = inputs_raw.get("expression_scale")
    if expression_scale is not None:
        expression_scale = _enum(expression_scale, {"counts", "cpm", "log", "log1p"}, "inputs.expression_scale")

    reference_raw = _required_mapping(raw, "reference", "reference")
    reference = ReferenceConfig(_enum(reference_raw.get("mode"), {"training_mean", "binary_marker"}, "reference.mode"))

    attention_raw = _required_mapping(raw, "attention", "attention")
    normalization = _enum(
        _required_value(attention_raw, "normalization", "attention.normalization"),
        {"l1", "softmax"},
        "attention.normalization",
    )
    temperature = attention_raw.get("temperature", 1.0)
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not np.isfinite(temperature):
            raise ValueError("attention.temperature must be finite")
        temperature = float(temperature)
    if normalization == "softmax" and (temperature is None or temperature <= 0):
        raise ValueError("attention.temperature must be positive for softmax")
    attention = AttentionConfig(normalization=normalization, temperature=temperature)

    assignment_raw = _required_mapping(raw, "assignment", "assignment")
    assignment = AssignmentConfig(_enum(assignment_raw.get("mode"), {"unique", "many_to_one"}, "assignment.mode"))

    cci_raw = _required_mapping(raw, "cci", "cci")
    transform = _enum(_required_value(cci_raw, "transform", "cci.transform"), {"none", "log1p"}, "cci.transform")
    direction = _enum(cci_raw.get("direction", "both"), {"both", "ref_to_cand", "cand_to_ref"}, "cci.direction")
    lr_selection = _enum(cci_raw.get("lr_selection", "given"), {"given", "train_topk"}, "cci.lr_selection")
    top_k = cci_raw.get("top_k", 50)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("cci.top_k must be a positive integer")
    cci = CCIConfig(transform=transform, direction=direction, lr_selection=lr_selection, top_k=top_k)

    ambiguity_groups = _parse_ambiguity_groups(raw.get("ambiguity_groups", []))
    ambiguity_names = {group.name for group in ambiguity_groups}
    for group in ambiguity_groups:
        for ref_identity in group.references:
            if ref_identity in ambiguity_names:
                raise ValueError(f"reference identity {ref_identity} cannot be an ambiguity group name")
    output_dir_raw = raw.get("output_dir", "runs")
    output_dir_path = Path(str(output_dir_raw)).expanduser()
    output_dir = (config_path.parent / output_dir_path).resolve() if not output_dir_path.is_absolute() else output_dir_path

    return RunConfig(
        provenance=provenance,
        seed=seed,
        inputs=InputPaths(
            expression=input_paths["expression"],
            cells=input_paths["cells"],
            features=input_paths["features"],
            lr_pairs=input_paths["lr_pairs"],
            contacts=input_paths["contacts"],
            truth=truth_path,
            expression_scale=expression_scale,
        ),
        reference=reference,
        attention=attention,
        assignment=assignment,
        cci=cci,
        ambiguity_groups=ambiguity_groups,
        output_dir=output_dir,
        config_path=config_path,
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(sorted(missing))}")


def _read_cell_table(path: Path, table_name: str, required: set[str]) -> pd.DataFrame:
    frame = _read_csv_with_unique_header(path, table_name)
    _require_columns(frame, required, table_name)
    if frame["cell_id"].isna().any():
        raise ValueError(f"{table_name} contains a missing cell_id")
    duplicate_ids = frame.loc[frame["cell_id"].duplicated(keep=False), "cell_id"].astype(str).unique().tolist()
    if duplicate_ids:
        raise ValueError(f"duplicate cell_id in {table_name}: {', '.join(duplicate_ids)}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    return frame


def _read_csv_with_unique_header(path: Path, table_name: str) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise ValueError(f"{table_name} is empty")
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate column name in {table_name}: {', '.join(duplicates)}")
    return pd.read_csv(path)


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def _validate_contacts(contacts: pd.DataFrame, groups: tuple[AmbiguityGroup, ...]) -> pd.DataFrame:
    contacts = contacts.copy()
    _require_columns(contacts, {"identity_a", "identity_b", "contact"}, "contacts.csv")
    contacts["identity_a"] = contacts["identity_a"].astype(str)
    contacts["identity_b"] = contacts["identity_b"].astype(str)
    contacts["contact"] = pd.to_numeric(contacts["contact"], errors="coerce")
    if contacts["contact"].isna().any() or (~np.isfinite(contacts["contact"])).any():
        raise ValueError("contacts.csv contact values must be finite numbers")
    if (contacts["contact"] < 0).any():
        bad = contacts.loc[contacts["contact"] < 0].iloc[0]
        raise ValueError(f"negative contact value for {bad['identity_a']} and {bad['identity_b']}")

    pair_values: dict[tuple[str, str], float] = {}
    pair_rows: dict[tuple[str, str], pd.Series] = {}
    for _, row in contacts.iterrows():
        first, second = str(row["identity_a"]), str(row["identity_b"])
        if first == second:
            raise ValueError(f"self-contact is not valid for identity {first}")
        key = _canonical_pair(first, second)
        value = float(row["contact"])
        if key in pair_values and pair_values[key] != value:
            raise ValueError(f"conflicting contact values for pair {key[0]} and {key[1]}")
        pair_values[key] = value
        pair_rows[key] = row

    normalized_rows = []
    for key, row in pair_rows.items():
        normalized = row.copy()
        normalized["identity_a"], normalized["identity_b"] = key
        normalized["contact"] = pair_values[key]
        normalized_rows.append(normalized)
    normalized_contacts = pd.DataFrame(normalized_rows, columns=contacts.columns).reset_index(drop=True)
    known_identities = set()
    for first, second in pair_values:
        known_identities.update((first, second))

    group_names = {group.name for group in groups}
    for group in groups:
        for reference in group.references:
            if reference in group_names:
                raise ValueError(f"reference identity {reference} cannot be an ambiguity group name")
            for identity in (*group.candidates, reference):
                if identity not in known_identities:
                    raise ValueError(f"identity {identity} from ambiguity group {group.name} is absent from contacts.csv")
            values = []
            for candidate in group.candidates:
                key = _canonical_pair(reference, candidate)
                if key not in pair_values:
                    raise ValueError(f"missing contact record for {reference} and {candidate} in group {group.name}")
                values.append(pair_values[key])
            positive = [value > 0 for value in values]
            if sum(positive) != 1:
                state = "both contact" if all(positive) else "both non-contact"
                raise ValueError(f"reference {reference} in group {group.name} has {state} with candidates {group.candidates[0]} and {group.candidates[1]}")
    return normalized_contacts


def load_inputs(config: RunConfig) -> Inputs:
    """Read the five input tables and enforce the declared input contract."""

    expression_raw = _read_cell_table(config.inputs.expression, "expression.csv", {"cell_id"})
    cells_raw = _read_cell_table(config.inputs.cells, "cells.csv", {"cell_id", "sample_id", "label"})
    expression_genes = [str(column) for column in expression_raw.columns if column != "cell_id"]
    if len(expression_genes) != len(set(expression_genes)):
        duplicates = sorted({gene for gene in expression_genes if expression_genes.count(gene) > 1})
        raise ValueError(f"duplicate expression gene columns: {', '.join(duplicates)}")
    if not expression_genes:
        raise ValueError("expression.csv must contain at least one gene column")
    expression_values = expression_raw[expression_genes].apply(pd.to_numeric, errors="coerce")
    if expression_values.isna().any().any():
        bad_columns = expression_values.columns[expression_values.isna().any()].tolist()
        raise ValueError(f"expression values contain NaN or non-numeric values in gene {bad_columns[0]}")
    if (~np.isfinite(expression_values.to_numpy(dtype=float))).any():
        bad_gene = expression_values.columns[~np.isfinite(expression_values.to_numpy(dtype=float)).all(axis=0)][0]
        raise ValueError(f"expression values are not finite in gene {bad_gene}")
    if (expression_values < 0).any().any():
        bad_gene = expression_values.columns[(expression_values < 0).any(axis=0)][0]
        raise ValueError(f"negative expression value in gene {bad_gene}")
    expression = expression_raw[["cell_id", *expression_genes]].copy()
    expression[expression_genes] = expression_values

    expression_ids = set(expression["cell_id"])
    cells_ids = set(cells_raw["cell_id"])
    if expression_ids != cells_ids:
        missing_in_cells = sorted(expression_ids - cells_ids)
        missing_in_expression = sorted(cells_ids - expression_ids)
        details = ", ".join(missing_in_cells + missing_in_expression)
        raise ValueError(f"expression.csv and cells.csv cell_id sets differ: {details}")
    if cells_raw["sample_id"].isna().any():
        raise ValueError("cells.csv contains a missing sample_id")
    if cells_raw["sample_id"].nunique() < 2:
        raise ValueError("sample_id must contain at least two samples")

    features = _read_csv_with_unique_header(config.inputs.features, "features.csv")
    _require_columns(features, {"gene"}, "features.csv")
    features = features.copy()
    features["gene"] = features["gene"].astype(str)
    feature_genes = list(dict.fromkeys(features["gene"].tolist()))
    used_feature_genes = [gene for gene in feature_genes if gene in expression_genes]
    missing_feature_genes = [gene for gene in feature_genes if gene not in expression_genes]
    if not used_feature_genes:
        listed = ", ".join(missing_feature_genes)
        raise ValueError(f"features.csv has no genes in expression.csv: {listed}")
    if config.reference.mode == "binary_marker" and "label" not in features.columns:
        raise ValueError("features.csv requires a label column for binary_marker")

    lr_pairs_raw = _read_csv_with_unique_header(config.inputs.lr_pairs, "lr_pairs.csv")
    _require_columns(lr_pairs_raw, {"ligand", "receptor"}, "lr_pairs.csv")
    lr_pairs = lr_pairs_raw.copy()
    lr_pairs["ligand"] = lr_pairs["ligand"].astype(str)
    lr_pairs["receptor"] = lr_pairs["receptor"].astype(str)
    for column in ("ligand", "receptor"):
        composite = lr_pairs[column].str.contains(r"[;,]", regex=True, na=False)
        if composite.any():
            value = lr_pairs.loc[composite, column].iloc[0]
            raise ValueError(f"multi-subunit complexes are not supported in first stage: {column}={value}")
    if "weight" not in lr_pairs.columns:
        lr_pairs["weight"] = 1.0
    lr_pairs["weight"] = pd.to_numeric(lr_pairs["weight"], errors="coerce")
    if lr_pairs["weight"].isna().any() or (~np.isfinite(lr_pairs["weight"])).any():
        raise ValueError("lr_pairs.csv weight values must be finite numbers")
    dropped_lr_pairs: list[dict[str, Any]] = []
    keep_rows = []
    for _, row in lr_pairs.iterrows():
        if row["ligand"] not in expression_genes or row["receptor"] not in expression_genes:
            dropped_lr_pairs.append(row.to_dict())
        else:
            keep_rows.append(row)
    lr_pairs = pd.DataFrame(keep_rows, columns=lr_pairs.columns).reset_index(drop=True)

    contacts_raw = _read_csv_with_unique_header(config.inputs.contacts, "contacts.csv")
    contacts = _validate_contacts(contacts_raw, config.ambiguity_groups)
    cells = cells_raw.set_index("cell_id")
    expression = expression.set_index("cell_id")
    return Inputs(
        expression=expression,
        cells=cells,
        features=features,
        lr_pairs=lr_pairs,
        contacts=contacts,
        ambiguity_groups=config.ambiguity_groups,
        used_feature_genes=used_feature_genes,
        missing_feature_genes=missing_feature_genes,
        dropped_lr_pairs=dropped_lr_pairs,
    )


def load_truth(path: str | Path) -> pd.DataFrame:
    """Read a truth table explicitly; it is intentionally separate from Inputs."""

    truth = _read_cell_table(Path(path), "truth.csv", {"cell_id", "true_label"})
    truth["true_label"] = truth["true_label"].astype(str)
    return truth.set_index("cell_id")

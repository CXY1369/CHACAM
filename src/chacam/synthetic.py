"""Deterministic, species-agnostic synthetic data for development and tests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticData:
    expression: pd.DataFrame
    cells: pd.DataFrame
    features: pd.DataFrame
    lr_pairs: pd.DataFrame
    contacts: pd.DataFrame
    truth: pd.DataFrame
    identities: tuple[str, ...]
    marker_genes: tuple[str, ...]
    contact_signal: bool
    signal_pairs: tuple[tuple[str, str], ...]
    decoy_pairs: tuple[tuple[str, str], ...]


def _contact_edges(identities: tuple[str, ...], n_extra_identities: int) -> set[frozenset[str]]:
    edges = {
        frozenset(("R1", "A1")),
        frozenset(("R2", "A2")),
        frozenset(("R1", "B1")),
        frozenset(("R1", "B2")),
        frozenset(("R1", "B3")),
        frozenset(("R2", "B3")),
        frozenset(("R2", "B4")),
        frozenset(("R2", "B5")),
        frozenset(("B1", "B2")),
        frozenset(("B3", "B4")),
        frozenset(("B5", "B6")),
    }
    for index in range(7, n_extra_identities, 2):
        if index + 1 <= n_extra_identities:
            edges.add(frozenset((f"B{index}", f"B{index + 1}")))
    return {edge for edge in edges if edge.issubset(identities)}


def make_synthetic(
    seed: int,
    n_samples: int,
    contact_signal: bool = True,
    marker_genes_per_identity: int = 3,
    n_extra_identities: int = 6,
    noise_rate: float = 2.0,
    marker_rate: float = 35.0,
    ligand_rate: float = 20.0,
    signal_rate: float = 20.0,
    decoy_rate: float = 20.0,
    n_decoy_pairs: int = 4,
) -> SyntheticData:
    """Generate samples where R1 contacts A1 and R2 contacts A2.

    Each sample has one cell for each fine identity.  A1 and A2 are observed
    as Ax, while their true labels remain in ``truth``.  Identity-specific
    ligand/receptor pairs receive receptor signal at contact partners when
    contact_signal is enabled; otherwise the same number of signal draws is
    assigned to random partners.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if marker_genes_per_identity < 1:
        raise ValueError("marker_genes_per_identity must be positive")
    if n_extra_identities < 6:
        raise ValueError("n_extra_identities must be at least six")
    if n_decoy_pairs < 0:
        raise ValueError("n_decoy_pairs must be nonnegative")
    rng = np.random.default_rng(seed)
    extra = tuple(f"B{index}" for index in range(1, n_extra_identities + 1))
    identities = ("A1", "A2", "R1", "R2", *extra)
    contact_edges = _contact_edges(identities, n_extra_identities)

    marker_by_identity: dict[str, list[str]] = {}
    marker_rows: list[dict[str, str]] = []
    for identity in identities:
        genes = [f"marker_{identity}_{index}" for index in range(1, marker_genes_per_identity + 1)]
        marker_by_identity[identity] = genes
        marker_rows.extend({"gene": gene, "label": identity} for gene in genes)
    signal_pairs = tuple((f"lig_{identity}", f"rec_{identity}") for identity in identities)
    decoy_pairs = tuple(
        (f"lig_decoy{index}", f"rec_decoy{index}")
        for index in range(1, n_decoy_pairs + 1)
    )
    lr_pairs = signal_pairs + decoy_pairs
    lr_genes = [gene for pair in lr_pairs for gene in pair]
    genes = [row["gene"] for row in marker_rows] + lr_genes
    contact_partners = {
        identity: sorted(
            other for other in identities if frozenset((identity, other)) in contact_edges
        )
        for identity in identities
    }

    expression_rows: list[dict[str, float | str]] = []
    cell_rows: list[dict[str, str]] = []
    truth_rows: list[dict[str, str]] = []
    for sample_number in range(1, n_samples + 1):
        sample_id = f"sample_{sample_number:03d}"
        cell_for_identity: dict[str, str] = {}
        for identity in identities:
            cell_id = f"{sample_id}_{identity}"
            cell_for_identity[identity] = cell_id
            observed = "Ax" if identity in {"A1", "A2"} else identity
            cell_rows.append({"cell_id": cell_id, "sample_id": sample_id, "label": observed})
            truth_rows.append({"cell_id": cell_id, "true_label": identity})
            row: dict[str, float | str] = {"cell_id": cell_id}
            row.update({gene: float(rng.poisson(noise_rate)) for gene in genes})
            for gene in marker_by_identity[identity]:
                row[gene] = float(rng.poisson(marker_rate))
            row[f"lig_{identity}"] += float(rng.poisson(ligand_rate))
            row["rec_decoy1"] += float(rng.poisson(decoy_rate)) if identity == "B1" else 0.0
            k_identity = len(contact_partners[identity])
            possible_partners = [other for other in identities if other != identity]
            random_partners = (
                list(rng.choice(possible_partners, size=k_identity, replace=False))
                if k_identity
                else []
            )
            signal_draws = rng.poisson(signal_rate, size=k_identity)
            partners_for_signal = contact_partners[identity] if contact_signal else random_partners
            for partner, signal in zip(partners_for_signal, signal_draws):
                row[f"rec_{partner}"] += float(signal)
            expression_rows.append(row)

    contacts_rows: list[dict[str, str | float]] = []
    for first, second in combinations(identities, 2):
        contacts_rows.append(
            {
                "identity_a": first,
                "identity_b": second,
                "contact": float(frozenset((first, second)) in contact_edges),
            }
        )

    return SyntheticData(
        expression=pd.DataFrame(expression_rows, columns=["cell_id", *genes]),
        cells=pd.DataFrame(cell_rows),
        features=pd.DataFrame(marker_rows),
        lr_pairs=pd.DataFrame(
            {
                "ligand": [pair[0] for pair in lr_pairs],
                "receptor": [pair[1] for pair in lr_pairs],
                "weight": [1.0] * len(lr_pairs),
            }
        ),
        contacts=pd.DataFrame(contacts_rows),
        truth=pd.DataFrame(truth_rows),
        identities=identities,
        marker_genes=tuple(row["gene"] for row in marker_rows),
        contact_signal=contact_signal,
        signal_pairs=signal_pairs,
        decoy_pairs=decoy_pairs,
    )


def write_synthetic(data: SyntheticData, out_dir: str | Path) -> None:
    """Write synthetic tables using the framework input filenames."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.expression.to_csv(output / "expression.csv", index=False)
    data.cells.to_csv(output / "cells.csv", index=False)
    data.features.to_csv(output / "features.csv", index=False)
    data.lr_pairs.to_csv(output / "lr_pairs.csv", index=False)
    data.contacts.to_csv(output / "contacts.csv", index=False)
    data.truth.to_csv(output / "truth.csv", index=False)

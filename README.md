# CHACAM

**C**CI-aided **H**ierarchical **C**ell **A**llocation **M**odel: a species-agnostic framework for assigning cell identities from expression data, using cell–cell interaction (CCI) evidence to separate identities that expression alone cannot tell apart.

> Status: research code under active development. The interfaces below may change.

## Overview

Some cell identities share nearly the same expression profile but differ in which neighbours they contact. CHACAM handles them in two stages:

1. **Expression-based assignment.** Each target cell is compared with reference identity profiles by cosine similarity. The similarity matrix is normalised over cells and over identities (L1 or softmax), and the product of the two gives an attention matrix `L`. Cells are then assigned greedily to the largest available `L` entry under per-identity capacities. Identities that cannot be separated by expression are declared as an *ambiguity group* (for example `Ax = {A1, A2}`) and assigned as one bucket.
2. **CCI triangulation.** For each ambiguity group, *reference identities* with a known contact pattern (each reference contacts exactly one of the two candidates) score ligand–receptor (L–R) communication with the two bucket cells. Each reference votes for the cell with the higher score as its contact partner. The bucket is resolved only when all available references agree; otherwise it is left unresolved with an explicit reason.

Evaluation is leave-one-sample-out: for each sample, reference profiles and (optionally) the L–R set are fitted on the other samples only, so nothing from the held-out sample leaks into its own prediction.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/CXY1369/CHACAM.git
cd CHACAM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Quick start

The synthetic example generates a small dataset (10 samples, one cell per identity per sample, with `A1` and `A2` observed only as the ambiguity group `Ax`) and runs the full pipeline:

```bash
python examples/synthetic/run_demo.py /tmp/chacam_demo
```

The first argument is a working directory (defaults to `examples/synthetic/`); the second, optional argument picks a config file from `examples/synthetic/` (`config.yaml` uses the supplied L–R list, `config_loso.yaml` selects L–R pairs per fold with `train_topk`). Expected output (abridged):

```text
status:
resolved      18
assigned      80
unresolved     2
metrics:
  stage1_accuracy: 1.0
  resolution_rate: 0.9
  reference_agreement: 0.9
  fine_accuracy: 1.0
  named_accuracy: 1.0
```

From Python:

```python
import chacam

result = chacam.run("path/to/config.yaml")
print(result.predictions.head())
print(result.output_dir)
```

## Inputs

All inputs are CSV files. Paths in the config are resolved relative to the config file.

| File | Required columns | Notes |
| --- | --- | --- |
| `expression.csv` | `cell_id`, one column per gene | Non-negative, finite values. |
| `cells.csv` | `cell_id`, `sample_id`, `label` | Same `cell_id` set as `expression.csv`; at least two samples. Cells of an ambiguity group carry the group name as `label`. |
| `features.csv` | `gene` (+ `label` for `binary_marker`) | Genes used for the expression stage; genes absent from `expression.csv` are dropped and recorded. |
| `lr_pairs.csv` | `ligand`, `receptor` (+ optional `weight`, default 1) | Single-gene ligands and receptors only; pairs with genes absent from `expression.csv` are dropped and recorded. |
| `contacts.csv` | `identity_a`, `identity_b`, `contact` | Symmetric, non-negative identity-level contact map; `> 0` means contact. |
| `truth.csv` (optional) | `cell_id`, `true_label` | Only read for evaluation, never by the predictor. |

`unassigned` is a reserved label name.

## Configuration

A run is described by one YAML file. The synthetic example ([examples/synthetic/config.yaml](examples/synthetic/config.yaml)):

```yaml
provenance: framework_baseline
seed: 0
inputs:
  expression: data/expression.csv
  cells: data/cells.csv
  features: data/features.csv
  lr_pairs: data/lr_pairs.csv
  contacts: data/contacts.csv
  truth: data/truth.csv          # optional
  expression_scale: counts       # counts | cpm | log | log1p
reference:
  mode: training_mean            # training_mean | binary_marker
attention:
  normalization: l1              # l1 | softmax
  temperature: 1.0               # used by softmax
assignment:
  mode: unique                   # unique | many_to_one
cci:
  transform: log1p               # none | log1p
  direction: both                # both | ref_to_cand | cand_to_ref
  lr_selection: given            # given | train_topk
  top_k: 50
ambiguity_groups:
  - name: Ax
    candidates: [A1, A2]         # exactly two
    references: [R1, R2]         # each contacts exactly one candidate
output_dir: runs/synthetic_demo
```

Key options:

- `reference.mode`: `training_mean` uses the mean training expression per label; `binary_marker` uses the marker table in `features.csv`.
- `assignment.mode`: `unique` gives each identity capacity 1 (an ambiguity group gets one slot per candidate); `many_to_one` removes the capacity limit.
- `cci.lr_selection`: `given` uses every pair in `lr_pairs.csv`; `train_topk` keeps the `top_k` pairs with the largest positive difference between mean contact and mean non-contact scores, computed on training samples only.
- `cci.direction`: which L–R direction counts between a reference cell and a bucket cell.

## Outputs

Each run writes to `output_dir`:

| File | Contents |
| --- | --- |
| `predictions.csv` | Per cell: `stage1_label`, `final_label`, `status` (`assigned`, `resolved`, `unresolved`, `unassigned`) and `unresolved_reason`. |
| `metrics.json` | Pooled and per-fold metrics, each with numerator and denominator. |
| `assignment_trace.csv` | Order of greedy assignments, with `L` values and tie flags. |
| `cci_evidence.csv` | Per reference cell, bucket cell and L–R pair contribution used in triangulation. |
| `attention/<sample>.npz` | Similarity `S`, normalised `P_col` / `P_row` and `L` for each fold. |
| `topk/<sample>.csv` | L–R table used in each fold, with selection effect sizes for `train_topk`. |
| `manifest.json` | Resolved config, package versions, input sizes, dropped genes and L–R pairs, per-fold details. |

Unresolved reasons: `unresolved_bucket_size` (the bucket does not hold exactly two cells), `unresolved_no_lr` (no L–R pairs available), `unresolved_no_reference` (no reference cells in the sample), `unresolved_tie`, and `unresolved_reference_conflict` (references disagree).

## Metrics

- `stage1_accuracy`: stage-1 label equals the observed label in `cells.csv`.
- `resolution_rate`: share of ambiguity-bucket cells resolved by triangulation.
- `reference_agreement`: share of eligible buckets (at least two references present, two bucket cells) where all references agreed.
- `fine_accuracy` (needs truth): accuracy among resolved cells.
- `named_accuracy` (needs truth): accuracy among all cells given a fine label (`assigned` or `resolved`).

## Repository layout

```text
src/chacam/
  schema.py         config and input validation
  reference.py      reference profiles (training_mean, binary_marker)
  attention.py      cosine similarity and L1 / softmax normalisation
  assignment.py     capacity-aware greedy assignment
  cci.py            L–R scoring and training-fold Top-K selection
  triangulation.py  CCI triangulation for ambiguity groups
  evaluation.py     fold-wise and pooled metrics
  pipeline.py       leave-one-sample-out run and output writer
  synthetic.py      synthetic data generator
examples/synthetic/ demo configs and script
tests/              pytest suite
```

## License

[MIT](LICENSE) © Xingyu Chen

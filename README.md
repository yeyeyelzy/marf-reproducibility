# MARC: Multi-Scale Adaptive Relation Calibration

This repository is the anonymized implementation package for the MARC paper.
It contains only the MARC/TCEM model family used in the paper, the H/E/R
factorial ablation switches, controlled mechanism variants, and scripts for
private MIMIC asset preparation and evaluation.

It deliberately does **not** contain MIMIC data, derived patient assets,
model checkpoints, predictions, experiment logs, or source code for baseline
reproductions.  Those materials either require controlled access or belong to
their respective official implementations.

## Repository scope

Included:

- the MARC prediction path and the TCEM backbone obtained by disabling H, E,
  and R;
- H (CCS hierarchy), E (state-aware edge calibration), and R (training-only
  class-aware relation calibration);
- full `2^3` factorial and controlled H/E/R configuration generators;
- MIMIC-III and MIMIC-IV private preprocessing entry points, a training-split
  graph builder,
  CCS hierarchy builder, training/evaluation scripts, and mechanism analyses;
- tests that use only synthetic arrays.

Not included:

- MIMIC-III/IV records, visit sequences, code maps produced from patients,
  processed tensors, semantic caches, graph assets, checkpoints, predictions,
  or tables used in the manuscript;
- baseline source code.  Official implementations and local adaptation
  boundaries are documented in [docs/BASELINES.md](docs/BASELINES.md).

## Quick start

Create the server environment as described in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), install this package, and run the
synthetic test suite:

```bash
python -m pip install -e ".[test]"
pytest -q
```

The commands below use paths outside the repository for all controlled data
and outputs.  Replace every placeholder before running them:

```bash
# Build private MIMIC-III tensors and the formal training-split graph.
python -m scripts.preprocess_mimic --dataset mimic3 \
  --mimic-root /secure/raw/mimiciii \
  --destination /secure/mimic3_assets

# Build a private CCS hierarchy from a private code map and public CCS file.
python -m scripts.build_ccs_hierarchy \
  --code-map /secure/assets/code_map.json \
  --ccs-csv /secure/public-knowledge/CCS_MULTI_DX_TOOL_2015.csv \
  --output /secure/assets/ccs_hierarchy.npz

# Train one run.  Checkpoints and logs are written outside this repository.
python -m scripts.train \
  --config configs/experiments/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets \
  --output-dir /secure/results/marc_mimic3/run_01 \
  --device cuda

# Select the validation-AUROC checkpoint and evaluate the private test split.
python -m scripts.evaluate \
  --config configs/experiments/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets \
  --checkpoint /secure/results/marc_mimic3/run_01/best_valid_auc.pt \
  --split test --output-dir /secure/results/marc_mimic3/run_01 \
  --device cuda
```

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the data contract,
five-run protocol, ablations, and mechanism exports.  The package must not be
used to redistribute controlled clinical data.

## Release boundary

This repository is intended for anonymous review.  Before making it public,
run `python -m scripts.check_release --root .`; verify the staged file list;
and confirm that no private input or output directory has been copied into the
repository.  The final anonymous URL should be the anonymous mirror URL, not
the account-owned GitHub URL.

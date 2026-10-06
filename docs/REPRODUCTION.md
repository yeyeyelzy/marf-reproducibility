# Reproduction protocol

## 1. Build private assets

Use authorized MIMIC access and place all raw and derived files outside this
repository.  Build the training-split adjacency and hierarchy asset before
model training.  For MIMIC-III, the authorized raw directory must contain
`ADMISSIONS.csv[.gz]` and `DIAGNOSES_ICD.csv[.gz]`:

```bash
python -m scripts.preprocess_mimic --dataset mimic3 \
  --mimic-root /secure/raw/mimiciii \
  --destination /secure/mimic3_assets
```

The formal defaults are 6,000 training patients, 1,000 test patients, the
remaining eligible patients for validation, randomization seed 6669, and graph
threshold 0.01.  The command writes only to the user-selected private output
directory.  It does not package or upload raw tables, patient identifiers, or
derived assets.

For MIMIC-IV, the release includes a corresponding entry point:

```bash
python -m scripts.preprocess_mimic --dataset mimic4 \
  --mimic-root /secure/raw/mimiciv \
  --gem-path /secure/knowledge/2018_I10gem.txt \
  --destination /secure/mimic4_assets
```

`--gem-path` accepts either CDC's fixed-width `2018_I10gem.txt` file or a
two-column CSV headed `ICD10,ICD9`.  The MIMIC-IV root must contain the
authorized `hosp/patients`, `hosp/admissions`, and `hosp/diagnoses_icd`
tables in CSV or CSV.GZ form.

Both builders retain the graph scope used in the formal TCEM/MARC runs: the
graph is computed from all visits of training-split patients, including their
target visits, while validation and test patients never contribute to the
graph.  Historical model tensors exclude each patient's target visit.  See
[DATA_FORMAT.md](DATA_FORMAT.md) for the exact support, pruning, and
normalization chain.

The formal TCEM splitter first places one supporting patient for every code
and the longest-sequence patient in the training set, then shuffles the
remaining patient IDs.  Its historical implementation converted patient IDs
through Python sets, whose iteration order is not a portable serialization
format.  Each preprocessing run therefore writes a private
`split_ids.json`.  To reproduce an already reported split exactly, retain that
file outside the repository and pass it back with `--split-manifest`.  The
manifest contains controlled patient identifiers and must never be committed.

Then create a private hierarchy asset.  Keep the tracked configuration
unchanged and pass the private file with `--ontology-path`; this also binds its
SHA-256 value into the selected checkpoint and records it in the private run
manifest.

## 2. Main MARC runs

For each independent run, use the common patient-level train/validation/test
split, train at most 40 epochs, select the checkpoint by validation AUROC, and
evaluate the selected checkpoint once on the test split.  Run five independent
replicates under the same randomization protocol.  Keep checkpoints, metrics,
and predictions in a private output root.

```bash
python -m scripts.train --config /secure/configs/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets \
  --ontology-path /secure/mimic3_assets/ccs_hierarchy.npz \
  --output-dir /secure/results/marc/run_01 \
  --device cuda
python -m scripts.evaluate --config /secure/configs/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets \
  --ontology-path /secure/mimic3_assets/ccs_hierarchy.npz \
  --checkpoint /secure/results/marc/run_01/best_valid_auc.pt \
  --split test --output-dir /secure/results/marc/run_01 --device cuda
```

`train.py` uses Adam with weight decay 0 and the paper learning-rate sequence:
0.01 (epoch 1), 0.001 (epoch 2), 0.0001 (epochs 3--19), and 0.00001
thereafter.  Early stopping uses validation AUROC, patience 5, and minimum
improvement `1e-4`.  The reported runs use split-specific time weights; the
mode is stored in both the YAML and checkpoint.  Evaluation writes metrics and
an NPZ containing patient order, labels, probabilities, 0.5-threshold
predictions, and patient embeddings to the private output directory.

## 3. Factorial and controlled mechanisms

Materialize the eight H/E/R configurations, train each with the same protocol,
and aggregate the five private results:

```bash
python -m scripts.run_factorial --base-config /secure/configs/marc_mimic3.yaml \
  --output-dir /secure/configs/mimic3_factorial
python -m scripts.run_mechanism_controls \
  --base-config /secure/configs/marc_mimic3.yaml \
  --output-dir /secure/configs/mimic3_mechanisms
```

The analysis scripts export only to a user-provided private directory:

```bash
python -m analysis.r_geometry --help
python -m analysis.e_gate_statistics --help
python -m analysis.h_attention_statistics --help
python -m analysis.summarize_runs --help
```

Do not commit their outputs.  The release package uses synthetic tests only.

## 4. Release checks

Before a public or anonymous upload, run:

```bash
python -m scripts.check_release --root .
git status --short
git ls-files
```

Review the output manually.  In particular, confirm that no controlled data,
derived asset, checkpoint, absolute server path, account name, institutional
domain, or manuscript draft has entered the release tree.

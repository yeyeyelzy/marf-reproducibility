# Reproduction protocol

## 1. Build private assets

Use authorized MIMIC access and place all raw and derived files outside this
repository.  Build a training-only adjacency and hierarchy asset before model
training.  For MIMIC-IV, the release includes an entry point:

```bash
python -m scripts.preprocess_mimic4 \
  --mimic-root /secure/raw/mimiciv \
  --gem-csv /secure/knowledge/2018_I10gem.txt \
  --destination /secure/mimic4_assets
```

`--gem-csv` accepts either CDC's fixed-width `2018_I10gem.txt` file or a
two-column CSV headed `ICD10,ICD9`.  The MIMIC-IV root must contain the
authorized `hosp/patients`, `hosp/admissions`, and `hosp/diagnoses_icd`
tables in CSV or CSV.GZ form.

Then create a private hierarchy asset and replace the placeholder
`ontology_path` in the dataset configuration.

## 2. Main MARC runs

For each independent run, use the common patient-level train/validation/test
split, train at most 40 epochs, select the checkpoint by validation AUROC, and
evaluate the selected checkpoint once on the test split.  Run five independent
replicates under the same randomization protocol.  Keep checkpoints, metrics,
and predictions in a private output root.

```bash
python -m scripts.train --config /secure/configs/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets --output-dir /secure/results/marc/run_01 \
  --device cuda
python -m scripts.evaluate --config /secure/configs/marc_mimic3.yaml \
  --data-root /secure/mimic3_assets \
  --checkpoint /secure/results/marc/run_01/best_valid_auc.pt \
  --split test --output-dir /secure/results/marc/run_01 --device cuda
```

`train.py` uses Adam with weight decay 0 and the paper learning-rate sequence:
0.01 (epoch 1), 0.001 (epoch 2), 0.0001 (epochs 3--19), and 0.00001
thereafter.  Early stopping uses validation AUROC, patience 5, and minimum
improvement `1e-4`.

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

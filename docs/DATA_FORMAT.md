# Private asset contract

All inputs below are private, generated outside this repository, and excluded
by `.gitignore`.  `data_root` passed to a training command has this layout:

```text
data_root/
├── code_adj.npz
├── code_map.json           # private cohort-derived diagnosis vocabulary
├── split_ids.json          # private exact patient split manifest
├── train/
│   ├── code_x.npz          # sparse [N, T, C]
│   ├── code_y.npz          # optional private sparse target codes [N, C]
│   ├── divided.npz         # sparse [N, T, C, 3]
│   ├── neighbors.npz       # sparse [N, T, C]
│   ├── visit_lens.npz      # dense key: lens, shape [N]
│   ├── hf_y.npz            # dense key: hf_y, shape [N]
│   ├── time_weight.npz     # sparse [N, C, 1]
│   └── code_x_timestamp.npz  # optional dense key: time
├── valid/
└── test/
```

Sparse assets use exactly three arrays: `idx`, `values`, and `shape`.  The
loader validates the expected shapes before training.  The held-out final
visit is not part of `code_x`; it supplies `code_y` and the heart-failure
label used by the prediction task.

`split_ids.json` is written to make the exact patient split reusable without
depending on Python set iteration order.  Both JSON files are controlled,
cohort-derived artifacts and must remain outside the source repository.

`code_adj.npz` is the training-split graph passed to graph propagation.  To
match the retained TCEM preprocessing used for the reported experiments, the
MIMIC-III and MIMIC-IV builders count coded pairs from every admission of
training-split patients, including those patients' held-out target admission.
No validation- or test-patient admission contributes to the graph.  The code
vocabulary is defined over the eligible cohort, as in the retained protocol;
only the training split contributes graph counts or weights.

For each included admission, the builder counts unique unordered code pairs
in both directions, leaves self loops absent, prunes entries whose preliminary
row share is below 0.01 only when the raw row sum exceeds 100, and then
row-normalizes the retained nonnegative weights.  Raw pair support is
symmetric, while thresholding by source-row share and final row normalization
can make the saved directed support and weights asymmetric.  E only scales
positive entries of this saved matrix, so it does not create new edges and
cannot increase a nonnegative original weight.

F1 is calculated using the fixed probability threshold 0.5.  AUROC, AUPRC,
and Brier score use the unthresholded probabilities.

For hierarchy calibration, construct a separate private `ccs_hierarchy.npz`
with `ancestor_ids` and `ancestor_mask` using `scripts.build_ccs_hierarchy`.
The asset holds code and concept IDs only; it must remain private because its
code map is derived from the controlled cohort.

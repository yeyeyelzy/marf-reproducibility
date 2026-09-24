# Private asset contract

All inputs below are private, generated outside this repository, and excluded
by `.gitignore`.  `data_root` passed to a training command has this layout:

```text
data_root/
├── code_adj.npz
├── train/
│   ├── code_x.npz          # sparse [N, T, C]
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
visit provides only the heart-failure label; it is not part of `code_x`.

`code_adj.npz` is the training-only graph passed to graph propagation.  The
MIMIC-IV asset builder constructs it from coded visits of training-split
patients, counts unique within-visit code pairs, leaves self loops absent,
prunes low row shares, and finally row-normalizes the nonnegative adjacency.
The raw support is symmetric, while the final row-normalized matrix can be
asymmetric.  E only scales positive entries of this saved matrix, so it does
not create new edges and cannot increase a nonnegative original weight.

F1 is calculated using the fixed probability threshold 0.5.  AUROC, AUPRC,
and Brier score use the unthresholded probabilities.

For hierarchy calibration, construct a separate private `ccs_hierarchy.npz`
with `ancestor_ids` and `ancestor_mask` using `scripts.build_ccs_hierarchy`.
The asset holds code and concept IDs only; it must remain private because its
code map is derived from the controlled cohort.

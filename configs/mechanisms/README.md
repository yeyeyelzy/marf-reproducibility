# Controlled mechanism configurations

Generate the six controlled configurations from full MARC:

```bash
python -m scripts.run_mechanism_controls \
  --base-config configs/experiments/marc_mimic3.yaml \
  --output-dir /secure/configs/mimic3_mechanisms
```

The controls are `h_uniform`, `h_no_residual`, `e_no_state`,
`r_stop_gradient`, `r_label_only`, and `r_no_mask`.  Each changes only the
named design factor; all common model and training settings are inherited from
the supplied base configuration.

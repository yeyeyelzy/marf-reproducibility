# Factorial configurations

Generate the eight paper configurations from a completed full-MARC base
configuration:

```bash
python -m scripts.run_factorial \
  --base-config configs/experiments/marc_mimic3.yaml \
  --output-dir /secure/configs/mimic3_factorial
```

The generated name `aHER.yaml` uses `1` for an enabled module and `0` for a
disabled module.  For example, `a011` has H disabled and E/R enabled.  The
generator removes the hierarchy asset path whenever H is disabled.

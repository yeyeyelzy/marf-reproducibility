# Server environment

The training environment is defined by the execution server rather than by a
developer workstation.  The tracked `environment.yml` records the Python and
non-CUDA package family used for the release.  Install a PyTorch build that is
compatible with the server's NVIDIA driver and CUDA runtime, then record the
exact `python`, `torch`, and CUDA versions in the private experiment manifest.

```bash
conda env create -f environment.yml
conda activate marc-server
# Install the server-compatible PyTorch build here.
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pytest -q
```

The source repository does not bundle a CUDA wheel, virtual environment, or
hardware-specific binary.  This prevents an accidental mismatch between a
published environment file and the actual GPU server.

# Server environment

The environment file is fixed from the formal execution server, not from a
developer workstation.  The server inspection on 2026-09-22 reported:

| Component | Verified value |
| --- | --- |
| Python | 3.8.16 |
| PyTorch | 1.8.1+cu111 |
| PyTorch CUDA build | 11.1 |
| torchvision / torchaudio | 0.9.1+cu111 / 0.8.1 |
| NumPy / pandas / scikit-learn | 1.24.4 / 2.0.2 / 1.2.2 |
| GPU | NVIDIA GeForce RTX 3090 |
| NVIDIA driver | 570.211.01 |

The NVIDIA driver reports CUDA 12.8 compatibility, but the formal PyTorch
wheel is compiled for CUDA 11.1.  `environment.yml` deliberately installs the
matching pip wheels from the official PyTorch wheel index; it does not replace
them with a newer CUDA build.

```bash
conda env create -f environment.yml
conda activate marc-server
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pytest -q
```

The repository does not bundle a virtual environment or CUDA binary.  A GPU
server must have a driver compatible with the CUDA 11.1 wheel.

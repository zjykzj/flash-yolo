# Docker dev environment

> A ready-to-run container reproducing this repo's verified environment (TensorRT included), so
> all four pipelines (train / eval / export / infer) run on a clean machine.
> Related files: root `Dockerfile` + `.dockerignore`, `docker/run.sh` (run entry),
> `docker/constraints.txt` (version snapshot); quick reference in `docker/README.md`.

## 1. Image contents and host requirements

| Item | Value |
|---|---|
| Base image | `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` (Python 3.12 + torch/torchvision) |
| Python / torch / CUDA | 3.12.15 · torch 2.13.0+cu130 · CUDA 13.0 |
| TensorRT | 11.4.0.106 (real package from the NVIDIA index, included in the image by default) |
| Other deps | `requirements.txt` + `requirements-dev.txt` (with matplotlib), versions pinned by `docker/constraints.txt` |
| apt | `libgl1` / `libglib2.0-0` (opencv-python import deps) + git/curl |
| Host | Docker + **nvidia-container-toolkit** + NVIDIA driver **>= 580** (CUDA 13), `--gpus all` at runtime |

### 1.1 Core version list

The versions below are the exact list from the current `py312` conda environment; the Docker
image uses the same torch/CUDA/TensorRT and pip dependency versions, while the Python / pip patch
versions follow the base image (3.12 series).

| Software | Version | Notes |
|---|---|---|
| Python | **3.12.15** | conda env `py312`; Docker uses the 3.12 series |
| pip | **26.2.1** | conda env; Docker follows the base image |
| torch | **2.13.0+cu130** | provided by the base image |
| torchvision | **0.28.0+cu130** | provided by the base image |
| CUDA | **13.0** | needs driver >= 580 |
| TensorRT | **11.4.0.106** | `tensorrt-cu13` / `-libs` / `-bindings`; optional |
| numpy | **2.5.3** | |
| opencv-python | **4.13.0.92** | |
| PyYAML | **6.0.3** | |
| safetensors | **0.8.0** | |
| onnx | **1.23.1** | |
| onnxruntime | **1.30.0** | CPU build |
| pycocotools | **2.0.11** | |
| pytest | **9.1.1** | dev-time |
| matplotlib | **3.11.2** | dev-time / optional |
| ultralytics | **8.4.173** | dev-time, AGPL-3.0 |

> `docker/constraints.txt` pins only pip packages; Python / torch / torchvision / CUDA /
> TensorRT are decided by the base image or the Dockerfile. For a CPU-only reproduction, swap
> torch for the matching CPU wheel, keep the rest, and skip TensorRT.

The image bundles a source snapshot (self-contained: inference / demo / CI); bind mount over it
for development.

## 2. Build

```bash
docker build -t flash-yolo:dev .

# non-Aliyun network: override the https index only (keep PIP_TRUSTED_HOST default; don't pass empty)
docker build -t flash-yolo:dev . --build-arg PIP_INDEX_URL=https://pypi.org/simple

# self-hosted http mirror: pass both, and PIP_TRUSTED_HOST must be the real hostname
docker build -t flash-yolo:dev . \
    --build-arg PIP_INDEX_URL=http://mirror.example.com/pypi/simple \
    --build-arg PIP_TRUSTED_HOST=mirror.example.com
```

## 3. Run (`docker/run.sh`)

```bash
docker/run.sh                              # interactive shell (repo mounted at /workspace)
docker/run.sh pytest tests/                # run any command inside the container
DATA_DIR=/path/to/datasets docker/run.sh   # datasets mounted at /data
IMAGE=flash-yolo:dev docker/run.sh         # pick the image name
```

What `run.sh` always passes: `--gpus all` (remove on GPU-less machines), `--shm-size=8g`
(DataLoader workers need it), `--user $(id -u):$(id -g)` (artifact ownership under bind mount),
the repo mounted at `/workspace`, `DATA_DIR` mounted at `/data`.

## 4. Constraints and notes

- **TensorRT engines are bound to the build machine's GPU architecture + TRT version**: using
  `--engine trt` in the container requires rebuilding the engine
  (`scripts/export.py --trt [--fp16]`); a `.engine` built on the host may not work in the
  container.
- For temporary apt/pip installs, drop the `--user` line in `run.sh` (run as root).
- No docker compose (one interactive service).
- **AutoDL container instances cannot build/run Docker** (no docker CLI, insufficient system
  disk) — the checkpoints below must run on another machine.

## 5. First-build checkpoints on a new machine

1. base tag exists: `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` (fall back to `-devel` if
   missing);
2. base Python is really 3.12 (`docker run --rm flash-yolo:dev python -V`);
3. TensorRT install from the NVIDIA index succeeds (on failure use the curl fallback noted in
   the Dockerfile);
4. apt package names vary with the base distribution (fix as errors come).

After a successful build, verify in order: `pytest tests/` → single-image inference →
`export.py --trt` (build the engine inside the container) → one short training run
(`--data <small dataset> --epochs 1`).

# Flash-YOLO

> ⚡ **A super-lightweight detection & segmentation framework — 31% fewer GFLOPs than YOLO26n
> for detection (18% for segmentation), millisecond GPU/CPU inference, and a complete independent
> pipeline: train (single-GPU / DDP) · evaluate · export · infer (image / video / stream).
> Raw dataset in, TensorRT engine out.**
>
> Core runtime is just PyTorch + NumPy — no monolithic abstractions, no ultralytics dependency.

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.7.0-blue.svg" alt="Version 0.7.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <img src="assets/traffic_det.jpg" width="49%" alt="flash-yolo detections — street traffic">
  <img src="assets/baseball_det.jpg" width="49%" alt="flash-yolo detections — baseball">
  <br>
  <em>Detections by the 100-epoch Flash-YOLO checkpoint — COCO val2017 samples, 640 input</em>
</p>

COCO val2017, official pycocotools (`scripts/eval.py`), 640 input. E2E = NMS-free default path;
NMS = one-to-many head + NMS. Official rows are Objects365 pretraining + COCO finetuning;
"from scratch" rows use the built-in baseline recipe for 100 epochs — not like for like.

| Model | Training | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n · official weights | O365 + COCO | **40.27** | 55.80 | **40.89** | 56.87 | 2.57M | 5.48 |
| YOLO26n · this repo | 100 ep from scratch | 36.22 | 51.32 | 37.15 | 52.67 | 2.57M | 5.48 |
| Flash-YOLO · this repo | 100 ep from scratch | 33.77 | 48.63 | 34.72 | 50.16 | 1.97M | 3.80 |
| **Δ (Flash-YOLO vs YOLO26n, same budget)** | | **−2.45 (−7%)** | **−2.69 (−5%)** | | | **−0.60M (−23%)** | **−1.68 (−31%)** |

<p align="center">
  <img src="assets/traffic_seg.jpg" width="49%" alt="flash-yolo-seg instance segmentation — street traffic">
  <img src="assets/baseball_seg.jpg" width="49%" alt="flash-yolo-seg instance segmentation — baseball">
  <br>
  <em>Instance segmentation by the 100-epoch Flash-YOLO-seg checkpoint — same COCO val2017 samples, 640 input</em>
</p>

| Model | Training | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg · official weights | O365 + COCO | **39.67** | 55.48 | **33.67** | 52.76 | 3.13M | 9.3 |
| YOLO26n-seg · this repo | 100 ep from scratch | 35.69 | 51.03 | 30.52 | 48.61 | 3.13M | 9.3 |
| Flash-YOLO-seg · this repo | 100 ep from scratch | 33.11 | 47.96 | 28.24 | 45.34 | 2.53M | 7.6 |
| **Δ (Flash-YOLO-seg vs YOLO26n-seg, same budget)** | | **−2.58 (−7.2%)** | **−3.07 (−6.0%)** | **−2.28 (−7.5%)** | **−3.27 (−6.7%)** | **−0.60M (−19%)** | **−1.7 (−18%)** |

- Official YOLO26n-seg training-time validator records 39.75 / 33.86 (Box / Mask mAP50-95);
  this pipeline reports 39.67 / 33.67 — a ~0.002 metric-implementation difference, not a model
  difference.
- Flash-YOLO-seg and YOLO26n-seg rows use the same recipe and budget (100 ep, batch 16, nbs 64,
  fp32); the 20-epoch ablation is in [docs/flash-yolo-seg-spec.md](docs/flash-yolo-seg-spec.md).
- Full latency (ORT-CPU / TensorRT fp32/fp16, b1/b8), deployment artifacts and provenance:
  [docs/weights.md](docs/weights.md).

## Supported Models

| Architecture | Task | Scale | Params | GFLOPs | Notes |
|---|---|---|---:|---:|---|
| YOLO26 | detection | n / s / m / l / x | 2.57M (n) | 5.48 (n) | official n checkpoint + from-scratch n validated |
| YOLO26-seg | segmentation | n / s / m / l / x | 3.13M (n) | 9.3 (n) | Box + Mask, COCO polygons / YOLO-seg txt |
| Flash-YOLO | detection | flash | 1.97M | 3.80 | this repo's lightweight design |
| Flash-YOLO-seg | segmentation | flash | 2.53M | 7.6 | same backbone/neck, Segment26 head |
| YOLOv3-tiny / -SiLU | detection | tiny | 8.85M | 5.56 @416 | darknet-compatible; official `.weights` for the original variant |

YOLOv3-tiny reference: official darknet weights evaluate at 17.16 / 35.90 (mAP50-95 / mAP50)
@640 through this pipeline. From-scratch 300-epoch runs at 416 reach **16.15 / 29.93** with the
current default recipe (SiLU + `--obj-iou-target`); the v0.6.0 recipe
(`--muon-w 0.528 --sgd-w 0.674`) reaches 16.52 / 30.44. See [docs/weights.md](docs/weights.md).

> YOLOv3-tiny's canonical input is **416** (darknet; the GFLOPs column is measured there).
> `config/train.yaml` keeps `imgsz: 640` for every architecture, so the v3 protocol needs
> `--imgsz 416` explicitly; from-scratch at 640 would also want re-clustered anchors.

## Quick Start

```bash
pip install -r requirements.txt
python model/summary.py --model flash-yolo   # architecture sanity check, no data needed
```

Datasets are described by a small yaml. Fill in the single machine-specific `path:` and
`--data coco` resolves to the local descriptor:

```bash
cp config/datasets/coco.yaml config/datasets/local/coco.yaml   # then edit `path:`
```

See [docs/dataset.md](docs/dataset.md) for YOLO-format descriptors and segmentation labels,
and [docs/docker.md](docs/docker.md) for the container environment.

## Usage

### Train

```bash
# detection — the repo's own lightweight architecture
python scripts/train.py --data coco --model flash-yolo --name flash-coco-e100 --batch 64 --workers 16
# -> runs/train/train-flash-coco-e100/  (best/last safetensors · results.csv · diag/ · samples/)

# segmentation
python scripts/train.py --data coco --model yolo26-seg --name seg-coco --batch 64 --workers 16
python scripts/train.py --data coco --model flash-yolo-seg --name flash-seg --batch 32 --workers 8

# official COCO-stage recipe (needs Objects365 segmentation initialization)
python scripts/train.py --data coco --recipe yolo26-seg-coco-ft --weights <objv1-seg.safetensors>

# multi-GPU DDP (training only; --batch is the GLOBAL batch, --workers is per GPU)
python scripts/train.py --data coco --name ddp-coco --device 0,1 --batch 64 --workers 8
# or: torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco --batch 64

# resume works across GPU counts / devices (1<->2 GPUs, CPU<->GPU)
python scripts/train.py --data coco --device 0,1 --resume runs/train/train-ddp-coco/resume.pt
```

From-scratch / finetune / resume, DDP details and all recipe options: [docs/train.md](docs/train.md).

### Evaluate

```bash
python scripts/eval.py --weights runs/train/train-flash-coco-e100/weights/best.safetensors --data coco
# E2E by default; add --nms for the one-to-many + NMS path.
# Segmentation checkpoints automatically produce Box + Mask tables.
```

`--split`, `--limit`, `--imgsz` and metric details: [docs/val.md](docs/val.md).

### Export

```bash
python scripts/export.py --weights runs/train/train-flash-coco-e100/weights/best.safetensors --trt
# -> runs/export/*.onnx + *.engine  (add --fp16 for half precision; segmentation exports two outputs)
```

ONNX / TensorRT options and output contracts: [docs/export.md](docs/export.md).

### Inference

```bash
python scripts/infer.py --weights runs/train/train-flash-coco-e100/weights/best.safetensors \
    --image assets/traffic.jpg
# --engine onnx|trt switches runtime; --image accepts a file or a directory.
# Annotated images + YOLO-txt labels are written under runs/predict/.

# Video / camera / RTSP stream -> annotated video under runs/predict/ (no per-frame labels)
# --video takes your own file / stream URL / camera index; there is no bundled demo video.
python scripts/infer.py --weights runs/train/train-flash-coco-e100/weights/best.safetensors \
    --video /path/to/video.mp4
python scripts/infer.py --weights runs/train/train-flash-coco-e100/weights/best.safetensors \
    --video 0 --max-frames 300   # camera index / live stream
```

Full engine and output notes: [docs/infer.md](docs/infer.md).

## Documentation

The full manual starts at [docs/README.md](docs/README.md): train / val / export / infer / dataset /
docker / FAQ, released weights and latency ([docs/weights.md](docs/weights.md)), and the
per-architecture specs (YOLO26, Flash-YOLO / -seg, YOLOv3-tiny).

## Citation

If you use the YOLO26 architecture or its training recipe in your work, please cite the YOLO26
paper. The YOLOv3-tiny variant follows the original YOLOv3 design (darknet cfg / `.weights`) —
cite the YOLOv3 paper when referring to that architecture.

```bibtex
@article{jocher2026yolo26,
  title   = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author  = {Jocher, Glenn and Qiu, Jing and Liu, Mengyu and Lyu, Shuai and Akyon, Fatih Cagatay and Kalfaoglu, Muhammet Esat},
  journal = {arXiv preprint arXiv:2606.03748},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.03748},
  url     = {https://arxiv.org/abs/2606.03748}
}
```

```bibtex
@article{redmon2018yolov3,
  title   = {YOLOv3: An Incremental Improvement},
  author  = {Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv preprint arXiv:1804.02767},
  year    = {2018},
  doi     = {10.48550/arXiv.1804.02767},
  url     = {https://arxiv.org/abs/1804.02767}
}
```

## License

The code is licensed under **Apache-2.0** — see [LICENSE](LICENSE) for details.

- **Official YOLO26 weights** (AGPL-3.0) are not distributed here; download them with
  `scripts/download_weights.py` for personal use.
- **From-scratch detection checkpoints** are published as safetensors on
  [Releases](https://github.com/zjykzj/flash-yolo/releases); the exact list, profiles and the
  segmentation status (local-only for now) are in [docs/weights.md](docs/weights.md).

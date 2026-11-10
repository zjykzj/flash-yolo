# Flash-YOLO Documentation

The root [README](../../README.md) covers the project overview, installation and quick start; this
directory is the **product manual**: usage and contracts of the four pipelines (train / val /
export / infer), dataset and model specs, and verification conventions.

## Which page do I need?

| I want to… | Read |
|---|---|
| Bring in a COCO / YOLO dataset | [datasets/README.md](datasets/README.md) |
| Pick or learn about a model | [models/README.md](models/README.md) |
| Train (from scratch / finetune / resume / DDP / pose) | [pipeline/train.md](pipeline/train.md) |
| Evaluate (pycocotools / E2E / NMS / Box+Mask / Pose) | [pipeline/eval.md](pipeline/eval.md) |
| Export ONNX / TensorRT | [pipeline/export.md](pipeline/export.md) |
| Infer on images / video / streams | [pipeline/infer.md](pipeline/infer.md) |
| Container environment | [docker.md](docker.md) |
| FAQ | [faq.md](faq.md) |
| Alignment with the official implementation / test gates | [development/alignment.md](development/alignment.md) |

## Layout

```text
docs/en/
├── datasets/
│   ├── README.md      descriptor schema, --data resolution, shared conventions
│   ├── coco.md        COCO json (RLE / crowd / missing images)
│   └── yolo.md        YOLO txt / YOLO-seg polygons
├── models/
│   ├── README.md      model matrix, weight metadata, shared structure conventions
│   ├── yolo26.md
│   ├── yolo26-seg.md
│   ├── yolo26-pose.md
│   ├── flash-yolo.md
│   ├── flash-yolo-seg.md
│   └── yolov3-tiny.md original + SiLU variant
├── pipeline/
│   ├── README.md      shared CLI / run dir / engine conventions
│   ├── train.md
│   ├── eval.md
│   ├── export.md
│   └── infer.md
├── development/
│   └── alignment.md   official-weight alignment, training internals parity, tests
├── docker.md
└── faq.md
```

## Metric conventions

- **Formal metrics** = `scripts/eval.py` + pycocotools, COCO val2017, 640 input (YOLOv3-tiny is
  natively 416 but also evaluates at 640).
- **E2E** = the NMS-free default path (o2o head); **NMS** = one-to-many head + NMS.
- **Pose** uses pycocotools keypoints (OKS, maxDets=20) on the **2,346 COCO-pose val2017 images
  with valid keypoints** (not the 5,000-image detection split).
- **Official YOLO26 weights** are Objects365 pretraining + COCO finetuning; from-scratch numbers
  are not like-for-like with the official ones (the pose COCO stage runs 400 epochs and
  initializes from O365 detection weights).

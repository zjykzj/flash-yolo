# Flash-YOLO (detection)

> This repo's own lightweight architecture (the model is the project's namesake). Goal: at 640
> input, **GFLOPs strictly below YOLO26n (5.48)**, accuracy above YOLOv3-tiny's converged level,
> and the change comes from a new structural idea, not hyperparameter tuning.
> Structure: `config/models/flash-yolo.yaml`; segmentation variant:
> [flash-yolo-seg.md](flash-yolo-seg.md).

## 1. Design proposition

- **YOLOv3-tiny's cheap spatial downsampling × YOLO26's CSP capacity allocation**; the head
  (Detect / E2E), loss and training pipeline are fully reused from YOLO26 — changes land only in
  backbone / neck.
- **Six stride-2 3×3 convs → PoolConv** (`MaxPool2d(2,2) → 1×1 Conv`): the channel transform
  happens after the half-resolution, saving ~9× FLOPs vs a stride-2 3×3 conv at equal channels.
- **stem C3k2 → DW 3×3 + 1×1 projection**: the P1/P2 stages have no detection head, so a whole
  CSP block's capacity is wasted; one depthwise 3×3 restores spatial mixing at nearly zero cost.
- The diff list vs YOLO26n has exactly two items: six downsamples → PoolConv, stem → DW 3×3 +
  1×1; C3k2 / SPPF / C2PSA / Detect and the three detection layers stay untouched.
- Structural conclusions: the DW stem pays off at equal compute; neck channel cuts and LiteBlock
  both lose to the final structure.

## 2. Final structure

25 yaml lines (one more than YOLO26n: the stem split into DW 3×3 + 1×1), a 262-layer module
tree:

- Size: **1,974,248 params (1.97M, −23% vs YOLO26n) · 3.80 GFLOPs @640 (−31%)**.
- Detection layers P3/P4/P5 and the Detect head stay untouched; weight keys map one-to-one with
  the YOLO26 prefixes — zero special cases in build/engine/training.
- Speed: close to YOLO26n at b1 (per-run overhead dominates), clearly faster at b8; full latency
  in §5.

## 3. Weight download

Download links point to the **v1.0.0** release assets.

| Weights | Training | Input | Download |
|---|---|---|---|
| `flash-yolo-e300-640` | 300 ep from scratch | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/flash-yolo-e300-640.safetensors) |

- Weight metadata carries `arch=flash-yolo` / `scale=flash` / `nc` / `imgsz` / `names`, so
  eval/infer/export resolve automatically.

## 4. Formal metrics (COCO val2017 @640)

| Training | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| From scratch 300 ep | **35.14** | **50.15** | **36.09** | **51.56** | 1.97M | 3.80 |

## 5. Inference latency

> Protocol: **model-only** (no pre/postprocessing), 640 input; b1 = single image, b8 = per-image
> average of batch 8. Engines warm up at init; numbers are steady-state repeats. Latency is the
> same across different weights of one architecture.

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.31 ms | 0.210 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 1.27 ms | 0.123 ms |
| Xeon 8470Q (25-core quota) | ONNX Runtime 1.30 · fp32 | 13.2 ms | 5.84 ms |

- ORT conditions: intra_op=16 / inter_op=1; b1 fixed graph, b8 dynamic graph; best-case min over
  300 samples.
- TRT conditions: b1 fixed engine, b8 batch-8 engine; 30 single shots + 100 pipelined runs.
- b1 hugs the launch-overhead floor; b8 reflects the compute difference. ORT absolute values are
  affected by neighbor load on the host.

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 3.07 ms | 1.25 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 3.32 ms | 0.535 ms |

- b1 is the best-case min of three rounds (single shots drift noticeably under laptop GPU
  dynamic clocks); b8 is stable.

## 6. Usage

```bash
# structure self-check (no data needed)
python model/summary.py --model flash-yolo

# train
python scripts/train.py --data coco --model flash-yolo --name flash-coco-e300 --batch 32 --workers 8

# evaluate (E2E / NMS)
python scripts/eval.py --weights flash-yolo-e300-640.safetensors --data coco
python scripts/eval.py --weights flash-yolo-e300-640.safetensors --data coco --nms

# export / infer
python scripts/export.py --weights flash-yolo-e300-640.safetensors --trt
python scripts/infer.py --weights flash-yolo-e300-640.safetensors \
    --image assets/traffic.jpg
```

## 7. Notes

- `flash-yolo` has only the single `flash` scale — don't pass `--scale`.
- Structural changes land only in backbone/neck; the new operators (PoolConv/DW stem) are
  registered in `model/basic/` and `model/build.py`.
- This is the product of structural trade-offs: accuracy below YOLO26n is expected, not a
  training bug.

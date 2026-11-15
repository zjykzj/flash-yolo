# Flash-YOLO-seg (instance segmentation)

> The `flash-yolo` instance segmentation variant (`arch_task = segment`). backbone/neck are
> **identical line-by-line** with the detection variant — only the last line swaps `Detect` for
> `Segment26`, the same relation as `yolo26 → yolo26-seg`.
> Structure: `config/models/flash-yolo-seg.yaml`; shared structure:
> [flash-yolo.md](flash-yolo.md), pose variant:
> [flash-yolo-pose.md](flash-yolo-pose.md).

## 1. Naming and scales

- Family name first, task suffix last: `flash-yolo-seg` (not `flash-yolo-seg-n`).
- Single `flash` scale: channels are materialized, there is no n/s/m/l/x dimension; `--scale`
  does not apply.
- The architecture is inferred from the weights filename (`flash-yolo-seg.safetensors`); the
  long-name regex wins, so the `flash-yolo` prefix never mis-swallows it.

## 2. Structure

25 yaml lines; the only diff vs `flash-yolo.yaml` is the last line:

```yaml
- [[17, 20, 23], 1, Segment26, [nc, 32, 64]]
```

`Segment26` = `Detect` + the `cv4` mask-coefficient branch + the `Proto26` prototype branch + the
training-time semseg auxiliary head (one shared implementation with yolo26-seg). The three
detection layers stay P3/P4/P5 (idx 17/20/23).

- `npr = 64`: `yolo26-seg.yaml` writes 256, which at n (`width=0.25`) becomes 64 through
  `make_divisible(256*0.25,8)`; the flash scale uses `width=1.0`, so the effective value 64 is
  written directly (writing 256 would yield the wrong 4.68M / 47.8 GFLOPs structure).
- `nm = 32` is a task constant, not scaled.

| Metric | flash-yolo | flash-yolo-seg | yolo26n-seg |
|---|---:|---:|---:|
| Params | 1,974,248 | **2,528,248** | 3,126,280 |
| GFLOPs @640 | 3.80 | **7.6** | 9.3 |
| Head params | 309,656 | 863,656 | 863,656 |

- backbone/neck are key-identical (test `test_backbone_neck_identical_to_detect`).
- The gap vs yolo26n-seg comes only from backbone/neck (−598,032 params, −1.7 GFLOPs); the
  segmentation heads are key-identical.
- GFLOPs roughly double vs the detection variant because of the Proto26 branch (same for the
  official seg).

## 3. Reused pipeline

| Stage | Reuse |
|---|---|
| Forward contract | E2E `(B,300,6+nm)` + proto `(B,32,160,160)`; raw `(B,4+nc+nm,N)` + proto |
| Train mode | dual-branch dict (`one2many`/`one2one`; the o2o proto/semantic are detached) |
| Loss | `build_loss` dispatches to `ComputeLossSeg` (detection terms/normalization/ProgLoss reused + mask BCE) |
| Validation | in-training Box+Mask, best fitness = box + mask mAP50-95; formal numbers via `scripts/eval.py` |
| Weights | metadata writes `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz` |
| Engines | pt / onnx / trt dual-output (output0 detections+coefficients, output1 prototypes) |

Besides `ARCHS`, the loss dispatch table and the weights-name regex, nothing new was added.

## 4. Weight download

This repo's weights download through the script: `python scripts/download_weights.py
--self <name>` (release details stay inside the script; `--self all` fetches every weight).

| Weights | Training | Input | Download |
|---|---|---|---|
| `flash-yolo-seg-e300-640` | 300 ep from scratch | 640 | `--self flash-yolo-seg-e300-640` |

- Weight metadata carries `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz`, so
  eval/infer/export resolve automatically.

## 5. Formal metrics (COCO val2017 @640)

300 epochs from scratch (default recipe, batch 32, nbs 64, fp32, mask_ratio 4, seed 0):

| Training | Params | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| From scratch 300 ep | 2,528,248 | 7.6 | 50.74 | **35.05** | 48.14 | **29.98** |

## 6. Inference latency

> Protocol: **model-only** (no mask decode / pre/postprocessing), 640 input; b1 = single image,
> b8 = per-image average of batch 8. Engines warm up at init. b1 is single-shot (with H2D and
> output readback), b8 is pipelined (per image).

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4090 | TensorRT 11.4 · fp32 | 2.34 ms | 0.402 ms |
| RTX 4090 | TensorRT 11.4 · fp16 | 3.10 ms | 0.179 ms |

- TRT conditions: b1 fixed engine, b8 batch-8 engine; 30 single shots + 100 pipelined runs
  (RTX 4090 / RTX 4060 Laptop share the same protocol).
- RTX 4090 rows: TensorRT 11.4.0.106 / driver 580.76.05 / 24 GB; b1 includes H2D and
  readback, b8 is 100 back-to-back pipelined runs.
- **fp16 is slower than fp32 at b1 on the 4090** (3.10 vs 2.34 ms; dominated by the
  32×160×160 proto readback and half→float conversion); on the 4060 fp16 is slightly faster
  at b1 (3.77 vs 4.07 ms). b8 is where fp16 wins on both platforms (4090 0.179 vs 0.402 ms;
  4060 0.702 vs 1.85 ms).
- b1 hugs the launch-overhead floor, b8 reflects the compute difference.

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 4.07 ms | 1.85 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 3.77 ms | 0.702 ms |
| Ultra 9 185H (22 threads) | ONNX Runtime 1.29 · fp32 | 26.6 ms | — |

- b1 is the best-case min of three rounds (single shots drift noticeably under laptop GPU
  dynamic clocks); b8 is stable.
- ORT conditions: intra_op=16 / inter_op=1; fixed batch-1 graph; best-case min over 300 samples
  (three rounds, min across rounds).

> Mask decode and IoU are currently mostly on the CPU; deployed end-to-end latency is
> significantly higher than model-only and needs a separate measurement.

## 7. Usage

```bash
python model/summary.py --model flash-yolo-seg
# -> 311 layers, 2,528,248 parameters, 7.6 GFLOPs

python scripts/train.py --data coco --model flash-yolo-seg --name flash-seg \
    --batch 32 --workers 8 --mask-ratio 4
python scripts/eval.py --weights flash-yolo-seg-e300-640.safetensors --data coco
python scripts/export.py --weights flash-yolo-seg-e300-640.safetensors --trt
python scripts/infer.py --weights flash-yolo-seg-e300-640.safetensors \
    --image assets/traffic.jpg
```

## 8. Notes

- Single `flash` scale — don't pass `--scale`.
- Segmentation training is dominated by the loader and validation; the architecture difference is
  not visible in throughput.
- Cross-device alignment (CUDA vs CPU oneDNN) shows 1e-3~4e-3 raw-output differences and mask
  boundaries may flip; compare within one runtime or go through TensorRT.

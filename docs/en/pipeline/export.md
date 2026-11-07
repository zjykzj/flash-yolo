# Export & deployment

> Entry point: `python scripts/export.py --weights <w.safetensors> [...]`.
> One chain: safetensors → onnx → (optional) TensorRT engine. The three runtimes share one
> `predict()` contract.

## 1. pt → onnx

```bash
python scripts/export.py --weights weights/yolo26n.safetensors              # default E2E fused graph
python scripts/export.py --weights weights/yolo26n.safetensors --raw        # raw head output
python scripts/export.py --weights weights/yolo26n.safetensors --dynamic    # dynamic batch
python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny

# yoloe prompt-free: E2E graph with the conf=0 static proposal path (official convention)
python scripts/export.py --weights weights/yoloe-26n-seg-pf.safetensors
# yoloe text mode: the prompts are fused into the cls conv — a single-input, class-locked ONNX
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car"
# yoloe text mode Non-E2E: dense o2m output; pass --nms at infer/eval time
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --nms
# yoloe visual mode: refer image + bbox/mask JSON -> vpe fused into the cls conv (class-locked)
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --visual-prompt prompt.json
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | — | `.safetensors` |
| `--out` | `runs/export/<stem>.onnx` | output path |
| `--raw` | off | export the raw head `(B, 4+nc, N)` (yolo26 only; errors for v3) |
| `--nms` | off | export the dense o2m output (yoloe-26 text only; pass `--nms` at infer/eval) |
| `--dynamic` | off | dynamic batch axis; default fixed batch 1 |
| `--classes` | — | yoloe-26 text mode: class names to lock into the exported model (combine with `--visual-prompt`) |
| `--visual-prompt` | — | yoloe-26 visual mode: refer image + bbox/mask JSON (vpe fused into the cls conv) |
| `--model` / `--scale` | inferred from filename | architecture / scale |
| `--nc` | metadata > model yaml | rebuild the head for non-COCO models |
| `--imgsz` | metadata > 640 | export input size |
| `--opset` | script default | ONNX opset |
| `--trt` | off | also build a TensorRT engine |
| `--fp16` | off | half-precision onnx + strongly-typed engine (requires `--trt`) |

**Exported graph conventions**

| Path | Output | Notes |
|---|---|---|
| yolo26 default (E2E) | `(B, 300, 6)` = [x1,y1,x2,y2,conf,cls] | decode + two-stage top-k in-graph |
| yolo26 `--raw` | `(B, 4+nc, N)` | decode/postprocess outside |
| yolo26-seg / flash-yolo-seg | `output0 (B, 300, 6+nm)` + `output1 (B, nm, H/4, W/4)` | coefficients appended; prototypes a separate branch |
| seg `--raw` | `output0 (B, 4+nc+nm, N)` + `output1` | masks decoded by `process_mask` |
| yolov3-tiny | `(B, NA, 5+nc)` | NA = 3·ΣH·W; obj×cls and NMS outside |
| yoloe pf | `output0 (B, 1000, 6+nm)` + `output1 (B, nm, H/4, W/4)` | E2E, **max_det=1000**, conf=0 static proposal path |
| yoloe class-locked (`--classes` / `--visual-prompt` / mixed) | `output0 (B, 300, 6+nm)` + `output1` | single input: the prompt embeddings are fused into the cls conv (text: BN folded, reprta removed; visual: the same fuse path); the exported model rejects new prompts |
| yoloe class-locked `--nms` | `output0 (B, 4+V+nm, N)` + `output1` | dense output is already pixel xyxy + sigmoid; NMS runs outside (`nms_per_image`) |

- Training artifacts carry metadata, so `--scale` / `--nc` can be omitted. Exported artifacts
  also carry deploy metadata: ONNX writes `metadata_props` (arch/scale/nc/imgsz/names/
  prompt_mode/end2end) and TRT writes a `<engine>.json` sidecar — so a class-locked yoloe
  graph can run `infer/eval --engine onnx|trt` without the original `.safetensors`. Older
  exports without metadata still need `--model` / `--scale`.
- yoloe visual-mode export runs the refer image through the backbone, rasterizes bboxes/masks
  at 1/8, computes SAVPE vpe and feeds the same `fuse_text` path; the exported graph is
  class-locked, so deployment needs no refer image (metadata `prompt_mode: text`, names = the
  prompt class names).
- yoloe text-mode export runs the full prompt chain (tokenize → MobileCLIP2-B → reprta+L2) —
  `mobileclip2_b.safetensors` must sit next to the model weights — then fuses the result into
  the cls conv, so the onnx has no text-encoder dependency. Fuse equivalence is verified:
  fused pt vs unfused 1.8e-6, onnx vs fused 6e-5, zero class flips above conf=0.25.
- **TorchScript exporter**: since torch 2.9 `torch.onnx.export` defaults to the dynamo exporter,
  whose `external_data=True` splits even ~10MB models into `.onnx.data`. This script pins
  `dynamo=False` (weights embedded, single file, `dynamic_axes` native). Fallback: dynamo +
  `external_data=False`.

## 2. onnx → TensorRT

```bash
python scripts/export.py --weights weights/yolo26n.safetensors --trt           # fp32
python scripts/export.py --weights weights/yolo26n.safetensors --trt --fp16    # + fp16
```

- `--fp16` requires `--trt`: it first exports `<stem>.fp16.onnx` via `model.half()`, then builds
  `<stem>.fp16.engine` with a **strongly-typed** network (fp32/fp16 artifacts coexist).
- TRT 11 removed `BuilderFlag.FP16` and has no precision-constraint flags; precision is decided
  entirely by tensor types. Handing an fp32 graph to the fp16 build raises ValueError directly.
- **Engines are bound to the build machine's GPU architecture + TRT version**: a different
  machine/version must rebuild.
- TensorRT is an optional dependency, not in `requirements.txt`; this repo builds and measures
  against TRT 11.4 / onnxruntime 1.30.

### fp16 measurements (RTX 5090, E2E inference segment)

| Model | b1 fp32 | b1 fp16 | b8 fp32 | b8 fp16 |
|---|---|---|---|---|
| YOLO26n @640 | 1.38 ms | 1.22 ms | 0.248 ms/img | **0.139 ms/img** |
| Flash-YOLO @640 | 1.32 ms | 1.23 ms | 0.213 ms/img | **0.126 ms/img** |

- b1 hugs the launch-overhead floor; fp16 gains live in batch (b8 −41~44%).
- No measurable accuracy loss; per-box comparisons must match by IoU, not bit-compare by score
  order (near-tie boxes swap).

## 3. The three runtimes

| Engine | Device | Notes |
|---|---|---|
| `pt` | GPU / CPU | reference implementation |
| `onnx` | CPU (ORT) | input size decided by the graph shape; explicit conflicts error out |
| `trt` | GPU | needs a prebuilt engine; half outputs converted back to fp32 at the engine layer |

Both `scripts/infer.py` and `scripts/eval.py` accept `--engine pt|onnx|trt` with the same
pre/postprocessing.

- **YOLOE works the same way**: pf / text (including `--nms`) can all export `--trt`; the text
  prompts are fused into the graph before engine construction, so no text encoder is needed at
  engine time. fp16 E2E top-k ties may flip classes — validate on detections above conf=0.25.

## 4. Acceptance thresholds

| Cross-check | Threshold | Test |
|---|---|---|
| pt ↔ onnx, raw | < 5e-4 | `tests/test_export_parity.py` |
| pt ↔ onnx, E2E detections | < 1e-3 | same |
| pt ↔ TensorRT fp32 | ≤ 2e-3 | `tests/test_trt.py` |
| torch-half ↔ TensorRT fp16 (raw) | bit-identical | `tests/test_trt.py` |

The E2E graph promotes cls with `.float()` in the head, so the half-precision engine's E2E output
stays fp32; raw / v3 outputs are half-precision (`TRTEngine._forward` converts everything back to
float32).

## 5. Release artifact conventions

Each weight on Releases usually ships 9 companion files (this repo's release archive
convention):

| File | Convention |
|---|---|
| `<stem>.onnx` | fp32 · batch 1 (yolo26 family = in-graph E2E top-k; v3 = decoded output) |
| `<stem>.fp16.onnx` / `<stem>.fp16.engine` | half-precision graph + strongly-typed build |
| `<stem>.engine` | fp32 · b1 |
| `<stem>.dyn.onnx` | dynamic-batch graph (for ORT b8) |
| `<stem>.b8.fp32.onnx` / `.b8.fp32.engine` | raw batch-8 graph + engine |
| `<stem>.b8.fp16.onnx` / `.b8.fp16.engine` | same · fp16 |

- **Engines are bound to the build machine** (GPU architecture + TRT version) — rebuild on a new
  machine: `scripts/export.py --weights <w> --trt [--fp16]`.
- b8 raw graphs need `--dynamic` export plus a manual build; the repo's export path pins b1.
- Weight filenames and download links follow each model page's convention; metadata travels in
  the safetensors header — see [../models/README.md](../models/README.md).

## 6. Gotchas

- `utils/anchors.py::make_anchors` must **build the grid in fp32 and cast** — a half-precision
  `torch.arange` is rejected by the TorchScript ONNX exporter. Regression gate
  `test_export_parity.py::test_make_anchors_half_dtype_is_export_safe` — don't revert it.
- Engines with batch > 1 need a `--dynamic` export plus a manual build; the `TRTEngine.predict()`
  contract is batch 1.
- Edge toolchains prefer `--dynamic` or fixed shapes; **never hand-edit onnx input names**
  (I/O binding matches by name).
- Don't install TensorRT from the PyPI name-squatting shell package; follow the Dockerfile's
  NVIDIA-index route, or use the Docker image directly.

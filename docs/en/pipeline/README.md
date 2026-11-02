# Pipeline shared conventions

Four pipelines: **train → eval → export → infer**. Each page documents only its own arguments;
this page lists the conventions shared by all of them.

## 1. Command shape

```text
python scripts/<train|eval|export|infer>.py [--weights ...] [--data ...] [options]
```

- Every script prints a resolved-parameter preview as the first console line:
  `scripts/xxx.py: key=value, …`.
- Scripts with a run directory (train / eval / infer) write logs to `<run_dir>/run.log`; scripts
  without one (`bench_io` / `convert_weights` / `download_weights`) fall back to `logs/`.
- `--data` accepts only a descriptor name or a `.yaml` path, never a bare directory (see
  [../datasets/README.md](../datasets/README.md)).

## 2. Weights and metadata

- Weight formats: `.safetensors` (primary), `.onnx`, `.engine` (TensorRT).
- Training artifacts and converters write optional metadata into the safetensors header:
  `arch` / `scale` / `nc` / `imgsz` / `names` (anchor models also `anchors`) + provenance
  `version/repo/commit/describe`.
- Resolution order: **CLI explicit > metadata > filename/default**; when nothing can be
  inferred, pass `--model/--scale` explicitly.
- The onnx input size is decided by the graph shape; an explicit conflicting `--imgsz` errors out
  instead of being ignored.

## 3. Engines

| Engine | Device | Notes |
|---|---|---|
| `pt` | GPU / CPU | reference implementation; alignment/acceptance run on it |
| `onnx` | CPU (ORT) | input size decided by the graph shape |
| `trt` | GPU | needs a prebuilt engine; bound to the build machine's GPU + TRT version |

The three runtimes share one set of pre/postprocessing and the `Detections` contract (with
optional masks).

## 4. Threshold conventions

- **eval default conf = 0.001** (aligned with the official evaluation convention).
- **infer default conf = 0.25** (cleaner output images).
- Don't mix the two; detection counts on inference images cannot be used to estimate mAP.

## 5. Run directories

- train: `runs/train/train-<name>/` (weights, results.csv, diag, samples, meta.json, run.log).
- eval: `runs/val/valN/` (results and run.log).
- infer: `runs/predict/predictN/` (visualizations/video + labels/).
- Existing directories get an incremented suffix; logs follow the run directory.

## 6. Environment and tests

- Dependencies: `pip install -r requirements.txt`; development / official-weight conversion:
  `pip install -r requirements-dev.txt`.
- Tests: `pytest tests/ -q`; env-gated suites:
  `FLASH_YOLO_SMOKE` / `FLASH_YOLO_REPRO` / `FLASH_YOLO_DDP`.
- Container: see [../docker.md](../docker.md).

## 7. Pages

- [train.md](train.md) — from scratch / finetune / resume, recipes, DDP, reproducibility
- [eval.md](eval.md) — pycocotools, E2E/NMS, Box+Mask
- [export.md](export.md) — ONNX / TensorRT / fp16
- [infer.md](infer.md) — images / directories / video / streams

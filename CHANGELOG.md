# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **TensorRT fp16 export (`scripts/export.py --trt --fp16`)**: fp16 deployment engines are built the
  TensorRT 11 way — a half-precision ONNX (`model.half()`, written as `<stem>.fp16.onnx`) parsed into
  a strongly-typed network, since TRT 11 removed the `FP16` builder flag and has no precision-constraint
  flags left. The fp32 and fp16 artifacts coexist (`<stem>.fp16.engine`), an fp32 graph handed to an
  fp16 build is rejected instead of silently producing an fp32 engine, engine outputs are normalized
  back to fp32, and `TRTEngine` now reports its precision in the model line. Anchor grids are
  generated in fp32 and cast (the TorchScript ONNX exporter rejects half-dtype `arange`, which
  blocked the whole half-export path; fp32 numerics are unchanged).

### Changed

- **README restructured around the four workflows** (train · evaluate · export · infer): it now opens
  with a flash-yolo-vs-YOLO26n comparison table (mAP / params / GFLOPs / latency, with per-column
  deltas) and a claim list, Quick Start splits into four one-command subsections, Evaluation and
  Verification get dedicated sections, the reproduction numbers are consolidated in a single Results
  section, and the TensorRT material states the tested engine versions (TensorRT 11.4 /
  onnxruntime 1.30) and its cross-version compatibility story.
- **Demo assets refreshed**: the README hero images are now COCO val2017 samples
  (`assets/traffic.jpg` / `assets/baseball.jpg`, image ids recorded in `assets/README.md`) with
  rendered flash-yolo detections beside them; the previous ultralytics-ecosystem demo images
  (bus.jpg / zidane.jpg) are removed, and the export-parity test now uses `traffic.jpg` as its
  real-image fixture.

## [0.5.0] - 2026-10-10

### Added

- **Multi-scale training + step LR schedule (with a `yolov3-tiny-darknet` recipe)**: `multi_scale: r`
  jitters each training batch's input size within imgsz×(1±r), quantized to stride multiples
  (ultralytics semantics; the pixel-space targets are rescaled with the canvas), and
  `lr_schedule: step` adds the darknet step decay (`lr_step_fracs` × `lr_step_gamma`, default
  80% / 90% ×0.1) beside the existing linear/cosine curves (also via CLI: `--multi-scale`,
  `--lr-schedule`). The `yolov3-tiny-darknet` recipe bundles both for the yolov3-tiny
  residual-gap experiment, with single-variable toggles (`--multi-scale 0` / `--lr-schedule linear`).
- **Flash-YOLO — the framework's own lightweight architecture**: a pooled-downsampling redesign
  of the yolo26 meta-architecture (`config/models/flash-yolo.yaml`, plus three screening variants),
  keeping the Detect head, E2E semantics and the full training pipeline. It replaces every stride-2
  3×3 downsampling convolution with a new `PoolConv` operator (max-pool + 1×1, the yolov3-tiny
  downsampling philosophy) and cuts the consumer-less P1/P2 stem to a channel projection: @640 that
  budgets **3.8 GFLOPs / 1.97M params** (−31% / −23% vs yolo26n) and it beats yolo26n at batch-1
  inference on CPU and CUDA-graph measurement (ORT-CPU +45%, torch-CPU +11%, CUDA-graph +11%; raw
  eager is a Python-dispatch-bound tie). `scripts/search_arch.py` searches the design space (in-place
  operator/channel surgery with per-layer GFLOPs/params tables and index-remapping for layer
  insertion), `LiteBlock` joins `model/basic` for block-level neck variants, and a new **`--cfg`**
  flag trains unregistered variant yamls (recorded in meta/resume) for screening runs.
- **TensorRT inference — the three-engine matrix is complete**: `utils/engine.py` gains `TRTEngine`
  + `build_trt_engine` beside the PyTorch and ONNX-Runtime-CPU engines — all sharing the same
  `predict`/`predict_timed` contract. The `tensorrt` dependency is optional (lazily imported, with
  an actionable error) and a built `.engine` is bound to the build machine's GPU + TRT version,
  stated plainly in the docs. `scripts/export.py --trt` chains safetensors → onnx → `.engine` in
  one command, `eval.py`/`infer.py` take `--engine trt`, and the yolo26-only guards on
  `--nms`/`--raw` now key off the `YOLO26_FAMILY` set so flash-yolo gets them too. Batch-1 fp32
  latency on this machine: flash-yolo **1.35 ms** vs yolo26n 1.52 ms (+12.6% throughput) vs
  yolov3-tiny@416 0.40 ms — TRT narrows the eager-to-deployment gap for every model (yolo26n
  5.97 → 1.52 ms). A parity test pins the fp32 engine's raw outputs to PyTorch within 2e-3.

### Changed

- **onnxruntime sessions use a bounded thread pool**: `OnnxEngine` now sets `intra_op_num_threads`
  to ≤16 and `inter_op_num_threads` to 1 instead of leaving ORT's core-count default — on
  container hosts that over-report CPU counts (this box: 208 reported vs a 25-core cgroup quota)
  the default pool oversubscribed and ran the engine 15-20× slow (120-196 ms vs 10-14 ms per 640²
  image, batch 1). Inference numbers are unchanged; only deployed speed.
- **flash-yolo adopts the DW-stem design**: the round-1 20-epoch screening (control = from-scratch
  YOLO26n) promoted S2 — the depthwise-3×3 stem variant: +0.67 mAP at equal GFLOPs, batch-1 latency
  unchanged (TRT 1.33 vs 1.32 ms) — into `config/models/flash-yolo.yaml`; the pre-promotion
  structure is kept as `flash-yolo-s1.yaml`, and s1-s4 together remain the `--cfg` screening corpus.
- **README training documentation**: a "Training Results" section with the from-scratch command +
  metric table for YOLO26n (100 ep) and YOLOv3-tiny (100 / 300 ep @416, plus the official weights
  through the same pipeline at matched input sizes), and a "Flash-YOLO" section documenting the
  design premise and the round-1 20-epoch screening table.

### Fixed

- **TensorRT batch-dim strip**: `TRTEngine._forward` now strips the batch dimension from the
  engine output, honoring the shared engine contract (`PtEngine` / `OnnxEngine` both strip) — the
  engine had only been exercised through raw `_forward` parity, so `predict_timed` (and with it
  `eval.py` / `infer.py --engine trt`) crashed on the un-stripped `(1, …)` output with an
  IndexError; `tests/test_trt.py` now pins the stripped shape.

## [0.4.0] - 2026-10-09

### Changed

- **Entry-point scripts now open with a resolved-parameter preview line**: every script logs
  `scripts/<name>.py: key=value, …` (same style as `model/summary.py: cfg=…`) to console and file
  as its first line, showing effective values (inferred architecture, filled-in defaults). The
  run.log header keeps the file-only `run dir:` + `argv:` lines for provenance.

### Added

- **Weights metadata — model parameters that travel with the checkpoint**: every training save
  (`best` / `last` / `best_raw` / `epochNNN.safetensors`) and both official-weight converters now
  write an optional JSON metadata block into the safetensors header (`arch`, `scale`, `nc`, `imgsz`,
  class `names`, plus the yolov3-tiny `anchors`) — header only, not a single tensor is touched, so
  weight alignment and converter byte checks are unaffected. Loading resolves through
  **CLI > metadata > filename/default**: `best.safetensors` from a custom run now evaluates or
  infers without hand-passing `--model/--scale/--nc`, and re-clustered yolov3-tiny anchors are
  restored from the file even against a stock model yaml. Metadata is strictly optional — weights
  without it (older conversions, any `.onnx`/`.pt`) behave exactly as before, and unknown keys are
  ignored.
- **`--imgsz` for eval / inference / export**: the input size is resolved end to end
  (CLI > weights metadata > 640) instead of being pinned to 640; the ONNX engine reads the fixed
  input size from the graph itself and rejects a conflicting `--imgsz` (re-export instead of
  silently mis-sizing), and the o2m NMS decode derives its grid shapes from the input size.
- **YOLOv3-tiny model support**: a darknet-faithful architecture (LeakyReLU convolution blocks,
  max-pool downsampling, route/upsample neck, anchor-based two-scale head with the official COCO
  anchors kept as-is, including the original mask quirk) defined in `config/models/yolov3-tiny.yaml`
  and assembled by the generalized `model/build.py` factory (`build_model("yolov3-tiny")` /
  `build_yolov3_tiny()`; `python model/summary.py --model yolov3-tiny`). Anchors are the official
  darknet values kept as-is (fixed pixel units — the weights self-compensate for input scale, so
  they must not be scaled with `imgsz`) and live as non-persistent buffers, so checkpoints keep a
  converter-friendly key set.
- **YOLOv3-tiny training**: `train/loss_v3.py` (BCE objectness + one-hot BCE classes + CIoU box,
  per-level best-anchor matching with a 0.7 ignore band) runs through the existing Trainer and
  validator — the default recipe drives it unchanged. `TrainConfig.model` (plus `train.py --model`)
  selects the architecture and `obj_gain` joins the loss weights; trainer/validator column sets,
  epoch rows, diagnostics and the val decode path now derive from the loss' `item_keys` and a
  per-head `postprocess_val` contract, so yolo26 console/CSV output is byte-identical to before.
- **`scripts/compute_anchors.py`**: evaluates the current yolov3-tiny anchor set against a dataset's
  train split with the YOLOv5-style shape-ratio coverage check (`anchor_t` 4.0) and re-clusters with
  darknet-style k-means (IoU distance, seeded/deterministic), printing a paste-ready `anchors:`
  fragment with a three-way verdict (acceptable / swap recommended / optional). The tool is
  model-agnostic: `--model` picks which model yaml's `anchors:` section is treated as the current
  set (default yolov3-tiny) and `--levels/--n` control the output grouping, so future anchor-based
  models reuse it unchanged. Custom-dataset from-scratch training should run it first; the official
  darknet weights must keep the official anchor set.
- **YOLOv3-tiny weights and inference pipeline**: `download_weights.py --model yolov3-tiny` fetches
  the official darknet `.weights` and `convert_weights.py` gains a darknet parser (cfg layer order
  incl. the P5-head interleave, 4/5-int header autodetect, exact file-length check). The engines
  accept `model=` and decode + NMS the v3 head output (`utils/postprocess.v3_detections`);
  `infer.py`/`eval.py`/`export.py` take `--model` with filename inference (`resolve_arch_scale`) and
  yolo26-only guards on `--nms`/`--raw`. `export.py --model yolov3-tiny` writes the single decoded
  output `(1, NA, 5+nc)`.

### Fixed

- **In-training validation no longer grinds on early checkpoints (yolov3-tiny)**: with an
  untrained model every anchor/class candidate clears the validation confidence floor
  (2535 anchors × 80 classes), and the per-class greedy NMS processed ~200k candidates per image
  — measured 7.2 s/image, i.e. a 5000-image validation pass grinding for ~10 hours with the GPU
  idle. `nms_per_image` is now a class-aware greedy over globally score-sorted candidates with an
  early exit at `max_det`: because kept boxes emerge in score order, the first `max_det` kept
  boxes are exactly the previous "full per-class NMS, then global top-`max_det`" result — outputs
  are unchanged (bitwise identical on 300 official-weight COCO val images and on the degenerate
  all-candidates case, modulo ordering within equal-score blocks) while the pathological case
  drops to ~12 ms/image. A full 300-image eval at `--imgsz 416` reproduces the previous metrics
  exactly (0.4139 mAP50 / 0.2210 mAP50-95).
- **Progress lines no longer wrap (and smear) on narrow terminals**: dataset-scan descriptions
  used the full annotation path — 180+ characters with the counter suffix — so on any terminal
  narrower than the line the redraw (`\r`) could no longer return to the start of the physical row
  and every frame left stacked artifacts. The scan line now shows just the file (directory) name,
  and every `ProgressBar` clamps its rendered line to the terminal width (right-truncating the
  description with a word-boundary-aware ellipsis; the bar/count/speed/elapsed suffix is always
  kept), so a redraw can never wrap.

## [0.3.0] - 2026-10-08

### Added

- **Inference and export for models with a non-COCO class count**: `scripts/infer.py --data
  <descriptor>` builds the head with the descriptor's `nc` and renders its label names (without it,
  the default is COCO/80), and `scripts/export.py --nc N` exports a graph with the right class count
  (`PtEngine` gained an `nc` parameter). A checkpoint trained on a custom dataset previously failed
  the strict weight load with a size mismatch and no way out; that mismatch is still the explicit
  error, now with a way to proceed.
- **`scripts/make_coco_subset.py`**: extracts a small COCO subset (seeded sampling of N train/val
  images, hard-linked by default) and emits ready-to-use coco/yolo descriptors — the coco one is
  copied into `config/datasets/local/` so `--data coco-tiny` works out of the box. It prints the
  `dataflow-cv convert coco2yolo` commands (the CLI converts annotations only; its empty `images/`
  is replaced by symlinks to the subset) and a 2,000/1,000-image subset turns a full startup +
  train + eval cycle into minutes instead of hours.
- **YOLO txt datasets (`data/yolo.py`) and the descriptor-driven factory (`data/build.py`)**: images
  come from a directory or a `.txt` image list; labels are sibling `<stem>.txt` files
  (`cls xc yc w h`, normalized) with a numeric-stem fallback (`1.txt` ↔ `000001.jpg`, and
  dataflow-cv's COCO-image-id naming). Missing/empty label files count as backgrounds, a wrong labels
  dir fails loudly instead of training on 100 % background, and standalone evaluation works too:
  `YoloDataset.gt_source()` materializes a COCO GT dict from the image sizes recorded during the eval
  loop and `CocoEvaluator` accepts a json path, a dict or a lazy callable. `data/scan.py` and
  `data/loader.py` carry the shared scan types and training machinery (cv2 fork contract, shared
  corrupt counter, augmentation entry) so the two formats cannot drift.
- **Dataset descriptors (`config/datasets/spec.py`)**: datasets are described by an
  ultralytics-style yaml — `format` (`coco` | `yolo`), `path` (dataset root, resolved against the
  yaml's own directory), `names` (list or `{index: name}` mapping; index = class id, length = nc)
  and per-role blocks (`train:` / `val:` / `test:` with `images`, plus `ann` for coco or optional
  `labels` for yolo, mirrored from the `images` segment when omitted). `config.datasets.load_dataset`
  resolves a name or a `.yaml` path into a validated `DatasetSpec` with absolute role paths and
  actionable errors (empty `path:` with relative role paths, unknown roles/format, a directory
  passed instead of a descriptor, ...). Name lookup prefers `config/datasets/local/<name>.yaml`
  (gitignored — the place where machine paths live) over the shipped template;
  `config/datasets/coco.yaml` becomes that template and keeps `path: ""`, and `load_names` keeps
  working off the same files.
- **`scripts/eval.py` reports the validation split the way training does**: the standalone
  evaluator runs the same dataset scan — a static `val:   Scanning ... (19 MB) ...` line, a
  throttled progress bar carrying `5000 images, 48 backgrounds, 0 missing`, and the shared
  `└ ... instances · categories · parse + scan` continuation now rendered by
  `data/coco.py::scan_summary` — so training and evaluation logs read the same way. Engine
  summaries are rendered per backend: `PtEngine` keeps the module-tree wording
  (`260 layers · 2,572,280 params`, same source as the training log) while `OnnxEngine` reports
  deployment facts (`ONNX 9.3 MiB · in (1, 3, 640, 640) · out (1, 300, 6)`) — the two parameter
  counts are not the same quantity (the exported graph has Conv+BN folded: 2,408,932
  initializers), and an ONNX graph has no module hierarchy to print.
  `utils.engine.device_label` / `resolve_device` are shared by the scripts so the environment
  line no longer needs a built engine (`_device_name(engine)` is gone).

- **Dataset scan statistics + progress bar (`data/coco.py::scan_split`)**: the three separate
  passes over the training split (file-existence check, `iscrowd` filtering, per-image label
  construction) are now one loop that also reports progress and counts. The console shows
  `118287 images, 1021 backgrounds, 0 missing` (backgrounds = images left with no box after crowd
  removal; missing = listed in the annotations but not on disk) on a throttled `ProgressBar`
  (`min_interval=0.25 s`, last frame always drawn — 118k unconditional redraws would cost seconds
  of stdout I/O), followed by a continuation line
  `└ 849949 instances · 80 categories · crowd 10052 excluded · parse 12.6s + scan 6.6s`.
  `ProgressBar` gains backward-compatible `unit`/`start`/`min_interval`/`pct` parameters plus
  `fmt_rate` (`18.0Kit/s`); with default arguments the rendered string is byte-identical to before.
- **Lazy "corrupt" accounting**: unreadable images no longer abort training. `load_image` returns
  a blank sample and bumps a fork-shared counter (`multiprocessing.Value(lock=False)` — inherited
  by the DataLoader workers, read once per epoch by the parent, no `resource_tracker` process);
  the trainer warns once per epoch and records the total in `meta.json`. Train samples become
  blank-canvas backgrounds (no fake boxes), val samples keep their GT (an honest miss). No
  per-image decode at scan time, so startup is not slowed down.

### Changed

- **`--data` takes a dataset descriptor and the model's `nc` follows it** (breaking): `train.py`,
  `eval.py`, `bench_io.py` and `compare_official.py` now take `--data <name|.yaml>` (name lookup:
  `config/datasets/local/<name>.yaml` first, then the shipped templates); passing a directory is a
  migration error. Descriptor roles (`train`/`val`/`test`) replace split names — `TrainConfig.data_dir`
  and `train_split` (with the `--train-split` flag) are gone, and `eval --split` selects a role
  (default `val`). The trainer builds the model with `nc = len(descriptor names)` — cross-checked
  against the COCO json categories — so a non-COCO class count no longer requires editing the model
  yaml; `meta.json` records the descriptor, format and root in place of `dir`/`train_split`. The COCO
  loader no longer derives paths from `(data_dir, split)` itself: `data/build.py` resolves every path
  from the descriptor, and `data/dataset.py` is deleted (its contents moved to `data/coco.py` and
  `data/loader.py`).
- **Config package reorganized: one home per kind of thing** (values in yaml, mechanisms in py).
  Model structures move to `config/models/` (`yolo26.yaml`), dataset label names to
  `config/datasets/<name>.yaml` read through the new `config.datasets.load_names()` (the COCO
  80-name table left the code), and the inference/eval constants module was renamed
  `defaults.py` -> `inference.py`. The training-config mechanism (`TrainConfig` + yaml/recipe/CLI
  merge) moved from `config/train.py` to `config/train_config.py`, removing the name collision
  with `scripts/train.py`; `TRAIN_CONFIG_PATH` lives next to it now. The dead `YOLO26_SCALES` table
  (duplicated `yolo26.yaml`'s scales, zero consumers) is deleted, and `config/__init__.py` no
  longer wildcard-re-exports (submodules are imported explicitly). `resume.pt` files written
  before this release pin the old module path and can no longer be resumed — continue from
  `weights/*.safetensors` instead; new checkpoints store the config as a plain dict, so future
  module renames cannot break them.
- **`scripts/eval.py` header follows the training layout**: it used to print only after the
  dataset *and* the engine had been built silently (~1.5 s of blank terminal), with a print order
  that did not match the build order, and its single `val:` line mixed dataset facts with
  evaluation parameters. It is now env -> model -> dataset -> eval parameters -> details, with the
  parameters on their own line (`eval:  split val2017 · images 8/5000 · conf 0.001 · iou 0.7 ·
  max_det 300`). The dataset construction is deliberately no longer wrapped in
  `redirect_prints`: that wrapper would have swallowed the new status line and progress bar,
  which write straight to stdout by design.
- **Training startup block now prints in build order**: environment + hyperparameters (immediately,
  before the model is built) -> model table -> dataset scan -> training components ->
  `Starting training`. Previously every line was deferred to `Trainer._print_startup()`, which ran
  only after `Trainer.__init__` had silently built the model and the dataset, leaving ~20 s of blank
  terminal (the 448 MiB `instances_train2017.json` parse alone is ~12.6 s). The JSON parse phase is
  announced by a single static line (`train: Scanning <ann.json> (448 MB) ...`): `json.load` is a
  blocking C call that never releases the GIL, so a spinner/clock could only be animated by forking
  a helper process — deliberately not done. The `--resume` block moved before the dataset build so
  its line stays inside the model section.

### Fixed

- **Classification bias prior follows the training image size**: `Detect.bias_init` hard-coded the
  640 inside `log(5 / nc / (imgsz / stride)^2)` (as the official implementation does), so a
  from-scratch run at any other `--imgsz` started with a per-cell classification prior off by
  `(640 / imgsz)^2`. The size is now threaded through (`YOLO26(..., imgsz=)` / `build_yolo26`,
  `bias_init(imgsz)`), and the trainer and `bench_io` build with `cfg.imgsz`; the default stays
  640, so standard runs and the official-weight alignment are unchanged.
- **`scripts/infer.py` can run non-`n` scales**: the script had no `--scale` and always built the
  model with `PtEngine`'s default `"n"`, so `--weights yolo26s.safetensors` died at the strict
  load with a shape mismatch. The scale is now derived from the weights filename
  (`model.weights.scale_from_weights`: `yolo26s.safetensors` -> `s`), `--scale` overrides it, and
  a name that carries no scale raises a clear CLI error instead of being guessed. The header is
  split into the shared env / model / input+params layout — it previously hard-coded `YOLO26n`
  and merged environment, model and mode into a single line — and the weights load now happens
  after `run.log` is attached, so load failures land in the run log instead of vanishing.
- **Missing-image warning no longer disappears**: `data/dataset.py` used a bare `print` (invisible
  in `run.log`, emitted before the banner); it is now an aggregated `logger.warning` naming the
  first missing file, and the count is carried into the scan line and `meta.json`.

## [0.2.0] - 2026-10-08

### Added

- **`scripts/bench_io.py` — training I/O throughput benchmark (workers / threads / batch picker)**:
  two-stage auto-narrowing — (1) a loader-only sweep (workers × threads, CPU-only, no GPU) keeps
  the top-K configs, (2) end-to-end runs (batch × top K) time the full training step (fwd + loss
  + assign + bwd + clip + opt + EMA) and rank them. Both stages are needed: **a faster loader does
  not mean faster training** — measured here (RTX 5090 / 25 cores / batch 64) loader-only reaches
  496 img/s while end-to-end only ~230 img/s, and raising workers 16 -> 20 made end-to-end *slower*
  (20 single-threaded workers saturate the 25 cores and starve the main process's collate / H2D
  copy / kernel launches). Candidates step by 8 up to the CPU budget, read from the cgroup quota
  (the affinity mask claims 208 cores, the real quota is 25); a batch that does not divide `nbs`
  is flagged (the accumulated wd factor would not be 1.0). **The tool only advises — it never
  edits the defaults**: `config/train.yaml` keeps the minimal portable values (batch 16 /
  workers 8).
- **Worker thread capping (`data/dataset.py::worker_init_fn`)**: DataLoader spawns processes, and
  each worker defaults to a full OpenCV/torch thread pool (= core count), so N workers × cores
  threads oversubscribe the machine. Measured at batch 64: 16 workers with library-default threads
  302 img/s -> 403 img/s with 1 thread (+34%); 20 workers × 1 thread reaches 496 img/s (+64%).
- **Training artifacts (default-on; `config/train.yaml` fields + CLI)**: a run no longer keeps
  only best/last/resume — (a) `save_period` + `keep_periodic`: `weights/epochNNN.safetensors`
  every N epochs (last K kept), so any mid-run point can be forked for a new experiment;
  (b) `diag_interval`: one `diag/train_diag.csv` row per N optimizer steps carrying the **pre-clip
  gradient norm** (taken from `clip_grad_norm_`'s return value — zero extra cost), per-group lr
  and the loss breakdown; this is what exposed that the clip is permanently saturated
  (||g|| ~ 2000-6000 at init against a threshold of 10, so the effective step is proportional to
  lr); (c) `aug_samples`: an augmented-sample grid (`samples/epochNNN.png`, GT boxes drawn) saved
  for the first epoch / the first epoch after close_mosaic / the last epoch, for eyeballing the
  augmentation; (d) `meta.json`: git commit/dirty flag, torch/cuda/GPU, full config snapshot, data
  statistics and argv; (e) `best_raw.safetensors` next to every new best (an EMA/raw pair for
  later analysis); (f) `stop_after`: run only the first N epochs while lr/close_mosaic still
  follow the full `epochs` schedule, so A/B screens stay epoch-aligned with a full run (shrinking
  `epochs` would change the schedule and break comparability); (g) linear warmup momentum ramp
  (`warmup_momentum` 0.8 -> `momentum`), matching the official training loop.
- **Validation losses (results.csv only)**: every validation round also computes the val loss
  with the same `ComputeLoss` on the same EMA model, from the *same* backbone/neck forward
  (`YOLO26.forward_feats`; eval-mode BN, no running-stat pollution — metrics stay bit-identical
  to the previous validator, 200-image baseline diff 0.0). The five items (box/cls/l1/o2m/o2o,
  same definitions as the train row) are written to `results.csv` as
  `val_box, val_cls, val_l1, val_o2m, val_o2o` (empty on non-validation epochs) and are
  deliberately **not printed** — matching the official behavior (ultralytics' console shows no
  val loss either; its values live in `val/*_loss` CSV columns). Cost ≈ +10 s per full val2017
  round (~2% of a 100-epoch run). Val letterbox size now follows the configured `imgsz` (was
  hardcoded 640, which mis-sized the loss anchors for non-640 runs); the train row's loss
  columns use a width-aware formatter (fixed-point as before, scientific fallback for
  degenerate values, so a diverging run cannot break column alignment); the gated smoke asserts
  the new CSV columns.
- **Training pipeline (M3)**: complete from-scratch COCO training — `train/` package with
  TAL+STAL assigner, dual-head loss (CIoU + BCE + box-L1) with ProgLoss schedule (o2m 0.8→0.1),
  MuSGD optimizer (Muon+Newton-Schulz orthogonalization blended with Nesterov SGD), EMA,
  warmup + linear LR, resume checkpoints (safetensors weights + resume.pt state), per-epoch
  COCO validation and `scripts/train.py` (finetune via `--weights`, resume via `--resume`);
  Detect head train mode (o2o branch trained on detached features); official yolo26n
  COCO-stage hyperparameters in `config/train.yaml` (nbs=128 keeps lr semantics independent of
  physical batch size); training defaults to fp32 (half-precision from-scratch training NaNs).

### Changed

- **Run logs now live in the run directory**: `train` / `eval` / `infer` write
  `runs/<kind>/<name>/run.log` instead of a timestamped file under `logs/`, so a run directory is
  self-contained — copy, archive, compare or delete it as one unit — and its log can be found
  without guessing from a timestamp (which never carried the run name). A resumed run appends to
  the same `run.log` instead of starting a second file, and `--resume <dir>` now reuses that
  directory rather than creating a stray empty `runs/train/trainN/` on the way in. The file format
  (timestamped plain text, 10 MB x 5 rotation) and the console output are unchanged, and `logs/`
  stays as the fallback for scripts without a run directory (`bench_io`, `convert_weights`, ...).
  `attach_file_log()` is idempotent, replaces any previous file handler and removes the fallback
  file when it is still empty; it also writes a two-line header (run dir + argv) so the log is
  self-describing.

- **The eval console table now matches the training-side layout, and finally shows P/R and the size
  buckets**: `scripts/eval.py` prints 11-wide right-aligned columns (the same shape as the
  in-training val row) and adds two columns the evaluator computed but never displayed —
  `P` / `R` (pycocotools exposes no such stats slot, so they are read off the max-F1 point of the
  101-point recall grid, the same convention the training-side FastMetrics uses; the two pipelines
  now have comparable P/R — at 100 epochs FastMetrics 0.5997/0.4473 vs pycocotools 0.6009/0.4519)
  — plus `AR@100`. Per-class rows are printed by default (80 rows, as ultralytics does for
  standalone validation) with `--summary-only` to collapse them to the all row; the table is
  followed by the `mAP_small/medium/large` and `mAP50/75` lines that were previously computed and
  thrown away. `runs/val/valN/metrics.txt` now stores the full table instead of just the all row.
  Two reporting bugs went with it: the per-500-image `[N/total] elapsed` line was printed through
  `logger.info` while the progress bar owned the terminal line, so it collided with the bar's `\r`
  row (it now goes to the log file only, the same `log_file_only` path the trainer uses); and the
  speed line listed pre/in/post-processing without the image load or the end-to-end total, so its
  numbers could not be reconciled with the reported throughput — it now reads
  `12.5 ms/image = load 2.5 + preprocess 1.5 + inference 7.9 + postprocess 0.1 + other 0.6 -> 79.9 img/s`
  (the residual is computed so the sum is exact) plus a `model-only inference` line. The per-class
  table also stopped shifting: class names were right-aligned in an 11-wide field, so the 10 COCO
  names longer than that ("baseball glove" 14, "traffic light" 13, ...) pushed their whole row's
  numbers to the right — the name column is now left-aligned and 15 wide, with the numeric columns
  sized to their headers (total table width 88 columns, was 94).

- **No machine-specific paths in the config**: `config/train.yaml` leaves `data_dir` empty — the
  dataset root must be given with `--data` (omitting it is a hard error instead of a guessed
  path), and `scripts/bench_io.py` makes `--data` required. The shipped config keeps only what
  works in any environment (batch 16 / workers 8, matching ultralytics `default.yaml`);
  environment-specific values go on the command line.
- **Training artifacts are on by default**: `save_period` 0 -> 20 and `aug_samples` 0 -> 8
  (`keep_periodic` 3 and `diag_interval` 50 unchanged). These artifacts exist for post-hoc
  analysis and tuning, so defaulting them off defeated their purpose; a 100-epoch run now ships
  with periodic checkpoints, gradient diagnostics, augment samples and `meta.json` for ~45 MB.
  Opt out with `--save-period 0 --aug-samples 0`.
- **Five augment geometry/colour mismatches aligned with the official pipeline (100% of samples)** —
  cross-checked against `ultralytics/data/augment.py` + `data/base.py` (8.4.173) and verified
  pixel-identical against the official `Mosaic`/`RandomPerspective` given the same images and
  mosaic centre: mosaic canvas 1,638,400 px with 0 differing pixels, affine output 409,600 px with
  0, labels 0 difference, identical box count after filtering.
  - **Mosaic tiles**: each tile is resized preserving its aspect ratio to long side = imgsz
    (official `load_image` rect_mode) and pasted at its natural size anchored at the random centre
    (xc, yc) (`Mosaic._mosaic4`). Ours stretched tiles to S×S, applied an extra per-tile
    r~U(0.5,1.5) scale and pasted them into quadrants — which broke the aspect ratio (4:3 images
    stretched ×1.33 vertically while val/inference letterbox preserves it, so the train and test
    shape distributions disagreed) and piled on extra scale jitter (effective scale std 0.325 ->
    0.44). The mixup partner image goes through the same path.
  - **RandomHSV back in HSV space**: the official applies an **additive** hue shift
    `(x + r*180) % 180`, multiplicative sat/val LUTs and `lut_sat[0] = 0` (pure white keeps its
    colour). Ours ran in HLS space and multiplied `s_gain` into **L** and `v_gain` into **S**.
  - **`bgr` semantics**: the official flag is the *probability of returning BGR* (source images
    are BGR; with the default 0, `random.uniform(0,1) > 0` is always true -> always RGB), i.e. a
    probability-p RGB/BGR channel swap. Ours was a per-channel gain/bias colour jitter (a near
    no-op).
  - **Box filtering**: after the affine, boxes are filtered exactly like the official
    `box_candidates` (`w>2`, `h>2`, area retention > 0.10, aspect < 100; detection `area_thr=0.10`);
    after the mosaic paste, `_cat_labels` (clip to 2S, drop zero-area boxes). We used to drop only
    sub-1px boxes, feeding heavily-clipped slivers in as positives.
  - **Affine translation** is now in output-size units, `(0.5 ± translate) * S`; it used the canvas
    width (2S on the mosaic path), doubling the shift. Shear is the official `S` matrix (x and y,
    applied after the rotation) and the matrix is composed as `M = T @ S @ R @ C`.
  - **`copy_paste` is a no-op by default** (`copy_paste_mode: "off"`): the official
    `CopyPaste.__call__` starts with `if len(labels["instances"].segments) == 0: return`, so it
    never runs on detection-only labels. Our box-level paste covered backgrounds and kept the
    labels of the objects it overwrote (a label-noise source); it survives behind
    `copy_paste_mode: "box"`. The recipe's `copy_paste: 0.075` is therefore inert (the startup
    block says so).
  Measured effect: boxes kept per image 9.5 -> 15.4 (the old path discarded ~40% of the
  annotations), box aspect ratio p95 8.9 -> 5.0 with max 223 -> 27.8, box area p90 0.094 -> 0.073.
  A 20-epoch screen under the same recipe and schedule reaches mAP@[.5:.95] 0.2865 at epoch 20
  against 0.2435 for the baseline (+17.7%; both the in-training metric, 0.2896 under pycocotools),
  and +44% relative at epoch 7.
- **Three training-loop parity fixes vs the official implementation (EMA / BN momentum / scale
  aug)** — (a) **EMA decay**: now the official `decay * (1 - exp(-steps/tau))` ramp; the previous
  `min(decay, (1+s)/(tau+s))` formula never reaches `decay` (at 100 epochs the weight was 0.989,
  a ~90-step window ≈ the raw model, where the official saturates at ~0.9999 / ~10k-step
  window). (b) **BN momentum 0.03** (official yaml-built models use 0.03 on all 114 BatchNorm
  layers; we had PyTorch's 0.1 default) — running-stats update rate only, the training forward
  (batch stats) is unaffected; inference numeric alignment is untouched (eps stays 0.001).
  (c) **Affine scale gain** now `U(1 - s, 1 + s)` (`0.5 → [0.5, 1.5]`), matching the official
  `random_perspective`; ours was `[s, 1/s]` (`[0.5, 2.0]`), a stronger upsample-side
  augmentation than the reference.
- **`dfl_loss` renamed to `l1_loss`** in the epoch table header, `results.csv` column and val
  row (reg_max=1 has no DFL — the term is a plain L1; this matches the official ultralytics
  loss names for YOLO26). The config field stays `dfl_gain` (same name as `hyp.dfl` in the
  official implementation).
- **Training throughput ~5x**: TAL assigner + loss rewritten as batched `(B,N,M)` masked ops
  (the per-image Python loop was ~105k kernel launches and thousands of device syncs per step)
  and MuSGD elementwise updates switched to `_foreach_*` with a single device sync per step;
  measured batch 64 / 640² on RTX 5090: step 1250→250 ms, 51→~250 img/s (245-epoch COCO ETA
  ~6.5 d → ~1.4 d). Loss forward values and optimizer updates are bit-identical to the previous
  implementation; gradients no longer flow through the label assignment graph (official
  `@torch.no_grad()` + detached-input semantics).
- **Training throughput, round 2 (~1.4x)**: channels_last (NHWC, `channels_last: true` in
  train.yaml, `--channels-last/--no-channels-last`) + MuSGD Newton-Schulz batched across
  same-shape parameter groups + `_foreach_*` EMA updates; measured end-to-end 408.8→287.8 ms/step
  at batch 64 (156→222 img/s on RTX 5090). `save_weights` normalizes tensors to contiguous
  (NHWC parameter strides are rejected by safetensors) and C2PSA/Detect use `reshape` instead
  of `view` for channels_last compatibility; inference/export stay NCHW, bit-identical parity
  untouched.
- **Faster in-training validation**: validation forwards are batched (16 images) instead of one
  image at a time — steady 89.5→146 img/s (~56→~34 s per val2017 epoch); assigner chunk trimming
  now needs a single host transfer (≤8 device syncs/step → 1) and preds NaN diagnostics are
  sampled every 10 steps (loss finiteness still checked every step).
- **Training recipes split into files — the base config is now the general 100-epoch recipe**:
  `config/train.yaml` holds the ultralytics-aligned defaults (100 epochs) and doubles as the
  built-in `default` recipe; named recipes live in `config/recipes/<name>.yaml`
  (`--recipe <name>`, or a direct `.yaml` path), each with a provenance header stating the
  source, the required initialization, and which published values are not implemented. Shipped:
  `yolo26-coco-ft.yaml` (published COCO stage per scale — 245/70/80/60/40 epochs for n/s/m/l/x,
  lr0 0.0054/0.00038, **requires Objects365-pretrained init**) and `yolo26-o365-pt.yaml`
  (published Objects365 stage, 150 epochs, unverified here — needs the O365 dataset).
- **Training display aligned with ultralytics semantics**: the epoch progress row now shows
  losses as the running mean within the epoch (ultralytics `tloss`; the live value converges to
  the frozen epoch mean at epoch end) and speed as the cumulative average (`n/elapsed`, tqdm
  semantics) instead of 10-batch sliding windows; the memory column now reports the process peak
  (`max_memory_reserved`, header renamed `GPU_peak`) instead of the jumpy current reserved value.
- **Faster in-training metrics (~4x)**: `FastMetrics.compute()` vectorized — one IoU matrix per
  (image, class) reused across all 10 IoU thresholds, all thresholds matched in a single
  detection pass, short-circuit when a block's max IoU is below the lowest threshold, IoU skipped
  for (image, class) pairs without GT (counted as FP only), vectorized 101-point AP interpolation
  and a single `bincount` for per-class GT counts. Outputs are bit-identical to the previous
  implementation (49-case baseline, max diff 0.0), so the metric semantics are unchanged.

### Fixed

- **Training deadlocked at the first epoch boundary — `cv2.setNumThreads(1)` was being called
  inside forked DataLoader workers** — OpenCV's pthreads thread pool is not fork-safe: once the
  parent process has executed any parallel cv2 op the pool exists, and a forked child calling
  `setNumThreads` waits on worker threads that do not exist in the child, blocking forever in
  `futex`. Training touches cv2 in the parent twice per epoch (validation's `cv2.imread`, the
  augmentation sample grid's `imwrite`), so the *first* fork (epoch 1, before any cv2 use) was
  clean while every later fork deadlocked: a COCO run (batch 64 / 16 workers) finished epoch 1 in
  9m44s, validated normally and saved its checkpoints, then never produced the first batch of
  epoch 2 — 16/16 workers parked in `futex_wait_queue_me`, GPU 0%, no output, no traceback; only
  Ctrl-C ended it. Whether a run survived was a race on whether the parent happened to touch cv2
  before forking. The cap now happens at import time in `data/dataset.py` (parent side, before any
  cv2 op and any fork), so workers inherit a 1-thread pool instead of reconfiguring one; the
  parent loses nothing (imread + letterbox: 407 img/s at 1 thread vs 396 at 25 — JPEG decode is
  single-threaded anyway). Measured trigger matrix: parent pool > 1 thread **and** child calling
  `setNumThreads` is the only deadlocking combination; a child that does not call it is safe
  regardless of the parent's pool. Regression gates:
  `tests/test_dataset.py::test_worker_thread_contract` and
  `::test_worker_init_survives_fork_after_parent_cv2_use` (forks a child once the parent has a
  4-thread pool — both fail in ~20s against the old code). `scripts/bench_io.py` now reads the
  library default via `cv2.getNumberOfCPUs()`: the import-time cap made `getNumThreads()` return
  1, which would have silently collapsed its `--threads 0` sweep into `--threads 1`.
- **Training-time mAP was under-reported ~10x (FastMetrics accumulated the PR curve in
  per-image order instead of global score order)** — AP was accumulated over detections
  concatenated image-by-image; COCOeval semantics require all detections of a class sorted
  globally by score (`_prepare`) before the precision/recall sweep. With interleaved scores
  across images the precision envelope collapses: on one fixed set of detections (1000 val
  images) mAP50 was 0.0183 before and 0.1973 after the fix, matching pycocotools' 0.197; on
  the full val2017 the fixed FastMetrics matches `scripts/eval.py` to 3 decimals (5.71% vs
  5.74%). P/R/AR were unaffected (they were already computed in global score order). Impact:
  the in-training log / results.csv / best-selection mAP was systematically 9–14x too low —
  one full run displayed mAP50 0.62% at epoch 2 where the true value was 5.74%. Regression
  gate: `tests/test_metrics.py::test_global_score_order_across_images`.
- **Added the official gradient clipping (global norm, max_norm=10)** — the official
  optimizer_step clips unconditionally every step (AMP order: unscale → clip → step); ours
  had no clipping. A/B on the tiny regime (1000 images, mosaic off, batch 64, 12 epochs):
  without clipping the run NaNs at epoch 7 — the o2o classification head (`one2one_cv3`)
  enters a BN-masked compounding growth (0.68 → 7e7 → 5.8e18 while the train losses stayed
  normal for 4 epochs); with clipping it is stable for 12/12 epochs (0.37 → 0.99, bounded),
  and the official implementation on the same data/recipe is stable 12/12. A single-variable
  A/B also showed the MuSGD blend constants (0.528/0.674 vs the official 0.2/1.0) cannot
  prevent the divergence (NaNs at epoch 4). A full-data probe shows ||g|| ~800-28000 from
  initialization (every step > 10), and at full lr the unclipped run explodes within 15
  steps — the clip is load-bearing for lr=0.01 from scratch, not just a safety net.
- **Training loss normalization now matches the official implementation (ultralytics E2ELoss)** —
  from-scratch runs never converged because two of the three loss terms were mis-normalized
  versus the reference the recipe is calibrated against (diagnosed from a 12-epoch run: every cls
  head bias still at its init value, val mAP 0.0000, BN `running_var` up to 1e9, all losses
  rising):
  - **cls** divided the BCE by the raw element count (B·N·nc ≈ 4.3e7) instead of the official
    soft-target sum `Σtarget_scores` (~10² per batch), diluting the classification gradient by
    ~10⁶ — the head received effectively zero gradient (predictions stayed at the bias-init
    sigmoid ≈ 1e-5, so every detection was filtered and mAP was 0 by construction). Now
    `ΣBCE / Σt`, the same denominator the box and L1 terms use;
  - **L1** summed `|Δltrb|` in grid units; the official reg_max=1 branch normalizes ltrb by
    `stride/imgsz`, takes the mean over the 4 sides and soft-weights by t (grid units made the
    term 10¹–10²× too strong relative to CIoU — L1 was 94% of the total loss, official balance
    is ≈1:1);
  - label-assignment soft targets now follow the official TAL normalization
    (`align/align_max × CIoU_max`, value ⊂ (0, 1] — previously "sums to 1 per GT"), and the o2o
    second-stage top-k selects by alignment (was CIoU).
  `tests/test_loss_parity.py` pins all three items and the ProgLoss-weighted total of both
  branches against the installed ultralytics E2ELoss on fixed inputs (rel < 1e-4).
- **Removed the extra `cls_w` factor on the o2o classification loss** — the published-recipe
  doc's "algorithm constant" 2.74 had no counterpart in the official implementation (ultralytics
  E2ELoss applies no branch-specific cls multiplier; effective o2o:o2m cls weight there is 9×,
  ours was 24.7×). With it, once ProgLoss pushes the o2o weight to 0.9 the o2o cls P3 head
  enters a BN-masked compounding growth (loss stays healthy while the head's weights grow
  geometrically 1.8 → 1e10, then explode; reproduced and A/B-tested by resuming from the
  epoch-1 checkpoint: with `cls_w=2.74` it blew up within ~110 batches, with `cls_w=1.0` the
  same schedule ran stable). The field is removed from `TrainConfig` / `train.yaml` / CLI.

## [0.1.0] - 2026-10-05

### Added

- **Initial release (M1+M2)**: faithful YOLO26 detection reproduction — architecture with
  verified weight alignment (2,572,280 params, strict load, bit-identical to official output),
  inference (.pt/.onnx, E2E NMS-free & NMS paths), pt→onnx export, COCO evaluation
  (40.27 / 40.89 vs official 40.1 / 40.9), ultralytics-style logging, progress bar and
  runs/ result layout.
- **Export `--dynamic` flag**: dynamic batch opt-in, default fixed batch=1 (edge toolchains
  prefer fixed shapes); ONNX weights embedded in a single file — `dynamo=False` pins the
  TorchScript exporter (torch 2.13's dynamo default split weights into `.onnx.data`).

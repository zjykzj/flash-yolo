# FAQ

> Cross-page "why / how" questions are answered here; per-page details:
> [datasets](datasets/README.md) · [models](models/README.md) ·
> [train](pipeline/train.md) · [eval](pipeline/eval.md) · [export](pipeline/export.md) ·
> [infer](pipeline/infer.md) · [docker](docker.md).

## 1. Numbers and conventions

### Where does the official YOLO26n 40.1 come from? Why can't from-scratch training reach it?

The official numbers are **not from-scratch training**: every official checkpoint is Objects365v1
pretraining (150 epochs) + COCO finetuning (n: 245 epochs, batch 128). The official docs state:
"No YOLO26 checkpoint was trained on COCO from random weights". This repo's same-budget
from-scratch comparison: YOLO26n 300 ep 37.82 / Flash-YOLO 35.14 (E2E, val2017) — **from-scratch
numbers side by side with official numbers are not like-for-like**.

### So what does "reproduction" reproduce?

It reproduces **verifiability**, not the official numbers: official weights loaded into this
repo's pipeline = 40.27 / 40.89 (E2E / NMS), the ~0.1 gap vs the official 40.1 / 40.9 is
metric-implementation noise; the same weights run through this repo's evaluation pipeline with
the ultralytics implementation give bit-identical numbers. The training internals are aligned
too: dual-implementation cross-checks at the same weights and the same real batch show total-loss
difference <1% and gradient-norm difference <2%; the augmentation pipeline is 0-diff per pixel
against the official one.

### Flash-YOLO is 2.68 points below YOLO26n — why is it worth using?

What you buy is compute: −31% GFLOPs (3.80 vs 5.48), −23% params, b8 latency −14% (TensorRT
fp32) / −26% (ORT-CPU b8). The accuracy-per-compute ratio is this repo's core lightweighting
metric; the structural conclusions are in [models/flash-yolo.md](models/flash-yolo.md).

### b1 is nearly tied in the latency table while b8 clearly favors Flash-YOLO — why?

batch 1 is the **floor** dominated by per-run overhead (startup/binding/launch): TRT fp32 1.32 vs
1.38 ms (−4%), the two architectures are barely distinguishable; batch 8 reflects the compute
difference: 0.213 vs 0.248 ms/img (−14%, the same magnitude as the 1.44 GFLOP ratio).
**GFLOPs ≠ latency** — pick by the target batch's convention.

### Why chase numeric alignment down to 0.1 / per-pixel?

A 0.1 gap could come from metric-implementation noise (acceptable) or a systematic deviation
from an implementation error (unacceptable). Two real lessons: BN eps with PyTorch's default
1e-5 (official: 0.001) diverges activations from layer 0; decoding xywh as xyxy on the o2m path
yields ~632-scale errors. The alignment tests are the gates against such "0.1-lower illusions".

### Why doesn't the official YOLOv3-tiny 33.1 match?

Different conventions: the official 33.1 is **test-dev @416 AP50**; the same weights @640 on
val2017 through pycocotools score 35.90 AP50 / 17.16 mAP50-95 in this pipeline. The two v3
variants: `yolov3-tiny` (darknet line-by-line isomorphic) only matches the official `.weights`;
`yolov3-tiny-silu` (only difference LeakyReLU→SiLU) is part of the recommended training config
(with `--obj-iou-target`). Key sets / param counts / GFLOPs are fully identical.

### Why does pose val have only 2,346 images?

The COCO-pose protocol only evaluates images with **valid keypoints** (`num_keypoints > 0`);
`person_keypoints_val2017.json` contains 5,000 images, but 2,654 of them have no pose GT.
`scripts/eval.py` filters automatically; do not mix this number with the 5,000-image detection
split.

### Why can't the official pose numbers be compared with from-scratch training?

The official `yolo26n-pose` is **Objects365 detection pretraining + 400 COCO-pose epochs**;
from-scratch training works but is not like-for-like. Also, the official recipe's `amp=true`
relies on the O365 initialization; from-scratch runs should use `--no-amp`.

## 2. Training

### Can from-scratch training use AMP?

No — measured explosions: fp16 NaNs within epoch 1, bf16 explodes on the first optimizer step
(BN eps=0.001 amplifies near-constant mosaic gray-border regions in cascade + the fp16 65504
ceiling). **From-scratch training defaults to fp32**; **finetuning** may try `--amp`.

### Training loss looks fine, but weights explode / NaN after a few epochs?

Gradient clipping `max_norm=10` is part of the official training loop and must run
unconditionally every step. Without it the o2o classification head enters "BN-masked compound
growth" — loss stays normal the whole time while non-bias weights compound per step to 1e10+,
then NaN. Debugging this class of issue: **watch the |W|max of non-bias weights, not just loss**.

### The first epoch's val mAP is 0 — is that normal?

Yes. The cls initial bias makes sigmoid ≈ 1e-5, below the conf threshold, so all predictions are
filtered; it recovers after a training step or two. Don't mistake it for a pipeline failure.

### Why is `val_rle` always 0 in pose training logs?

The validation forward does not return `kpts_sigma` (the sigma branch only runs in training
mode), so the validation RLE term short-circuits to zero; the training `rle_loss` keeps updating.
This matches the official training-time validator's `val/rle_loss=0`.

### Training hangs: the process is alive, GPU at 0%, no error?

A known scenario, already fixed in the framework: OpenCV's thread pool is not fork-safe — once
the parent process has touched cv2, calling `cv2.setNumThreads()` inside a worker blocks forever
(typical symptom: epoch 1 fine, everything hangs from epoch 2). The contract is **thread counts
are set by the parent process when `data/loader.py` is imported; workers only inherit** — don't
break it when editing the data pipeline.

### How long does training take / how to size GPU memory / tune batch?

Reference magnitudes: YOLO26n 300 ep, batch 32 @640, one RTX 5090 ≈63 h; Flash-YOLO same budget
≈62.7 h. Physical batch is decided by GPU memory — **gradient semantics and lr are guaranteed by
nbs (default 64); changing batch doesn't require touching lr**. Don't guess throughput — measure
with `scripts/bench_io.py` (see [pipeline/train.md](pipeline/train.md) §6).

### Is multi-GPU / distributed training supported?

Yes (**training only**). `--device 0,1` spawns one worker per GPU automatically; torchrun works
too:

```bash
python scripts/train.py --data coco --device 0,1 --batch 32 --workers 8
```

`--batch` is the **global** batch (must divide evenly), `--workers` is per GPU. eval / export /
infer stay single-process. Multi-GPU is not bit-identical with single-GPU (no SyncBatchNorm);
repeated runs at the same GPU count are bit-reproducible.

### Can resume switch from single-GPU to multi-GPU, or across devices (CPU↔GPU)?

Yes. `resume.pt` records `world_size` / `device`; restore first lands on CPU and then copies to
the target device. Changing GPU count/device warns that the trajectory is no longer
bit-identical. Same GPU count + same device stays bit-equivalent. Changing
`--batch`/`--nbs`/`--imgsz`/`--amp` also resumes but warns.

### What to change for my own dataset?

Write a descriptor yaml ([datasets/README.md](datasets/README.md) §1) + `--data <descriptor>`:
nc/names travel with the descriptor (the model head is built by nc), index = class id. The
custom-dataset checklist is in [datasets/README.md](datasets/README.md) §5.

### Switching yolov3-tiny's input size — do anchors need recomputation?

**Official weights: no** — anchors are darknet fixed pixel values, self-compensated by the
network under multi-scale training; linear scaling with imgsz is wrong (measured on 500 images:
mAP50 640 original 0.412 → linear scaling 0.154). **From-scratch training** at a new input size
does need re-clustering: `python scripts/compute_anchors.py --data <descriptor>`.

## 3. Data

### The scan warns that train2017 is missing 1 image?

The official train2017 is indeed missing `000000391895.jpg`. The loader filters by file existence
at init and warns — expected behavior.

### Must YOLO label filenames match the image names?

Two-level fallback: `<stem>.txt` first; on miss, `str(int(stem)).txt` — compatible with
"COCO-image-id naming" (`000000397133.jpg` → `397133.txt`) and `1.txt ↔ 000001.jpg`. Zero hits
across the whole directory **errors out** (all-background guard). See
[datasets/yolo.md](datasets/yolo.md).

### How are crowd annotations handled? Why do the two formats' val numbers differ?

Training drops crowd in both formats (statistics equal item by item); val keeps crowd in COCO
(official convention) while YOLO has no crowd concept — the val difference comes only from this
(full coco-tiny 0.4080 vs 0.4024).

### What happens with corrupt images?

No per-image decode during scanning (lazy); `load_image` failures are counted and warned,
training continues: train returns a blank image + **empty labels** (no fake boxes taught), val
returns a blank image + **kept GT** (honestly recorded as a miss).

### Can `--limit` subsets be compared across formats?

Not directly: COCO takes the first N in json order, YOLO in filename order — two different image
sets. Cross-format comparisons need the full set or aligned subsets.

## 4. Evaluation

### Why do eval and infer have different conf defaults?

`eval.py` uses 0.001 (aligned with the official evaluation convention), `infer.py` uses 0.25
(cleaner output images). Don't estimate mAP from detection counts on inference images.

### Evaluation of early/untrained models is extremely slow (seconds per image in NMS)?

Reason: at conf 0.001 an untrained model passes almost all of 2535 anchors × 80 classes;
per-class full NMS measured 7.2 s/image. NMS is now **class-aware + early-stopping** (returns
once max_det is filled), ~12 ms/image, canonically equivalent to the per-class full version.

### How far apart are E2E and NMS? Which should I look at?

Same weights usually differ by 0.6~1 points (official 40.89 vs 40.27) — both are reported. E2E is
the default path (one less deployment step; NMS-free is YOLO26's selling point); NMS goes through
the o2m head. See [pipeline/eval.md](pipeline/eval.md) §4.

## 5. Export and deployment

### Why did a `.onnx.data` file appear next to my exported onnx?

Since torch 2.9 `torch.onnx.export` defaults to the dynamo exporter, whose `external_data=True`
splits even ~10MB models in two — the `.onnx` alone is unusable. `scripts/export.py` pins
`dynamo=False` (weights embedded, single file); if torch someday removes the legacy path, the
fallback is dynamo + `external_data=False`.

### Can a TensorRT engine move to another machine?

No: the engine is bound to the **build machine's GPU architecture + TensorRT version**. A
different machine/version must rerun `scripts/export.py --trt [--fp16]` on the target machine.

### How to enable fp16 on TRT 11? Why is there no FP16 flag?

TRT 11 removed `BuilderFlag.FP16` and has no precision-constraint flags — precision is decided
entirely by the network's tensor types. `--fp16` first exports a half-precision onnx via
`model.half()`, then builds `<stem>.fp16.engine` with a strongly-typed network (coexisting with
the fp32 engine). The gain is in batch: b8 −41~44% per image; no measurable accuracy loss.

### Can onnxruntime run the half-precision onnx?

CPU ORT has no fp16 kernels and cannot run half-precision graphs; the half-precision acceptance
anchor is the **torch half model** (measured: the TRT fp16 engine's raw output is bit-identical
with it). ORT-CPU's role is the fp32 reference convention.

### pip install TensorRT hangs / fails?

The same-name `tensorrt` package on PyPI is a shell (it hangs); follow the Dockerfile's NVIDIA
direct-index route, or use this repo's Docker image directly (TRT 11.4 included). See
[docker.md](docker.md).

## 6. Reproduction and environment

### Two runs with the same seed differ?

Check four things: ① is `deterministic` on (default on); ② are `workers`/`batch` the same (they
are part of the reproduction conditions); ③ did someone add bare global `np.random`/`random`
calls; ④ is this a resume scenario. **Bit-identical across machines is impossible** (different
kernels) — only statistical agreement is promised. See [pipeline/train.md](pipeline/train.md) §5.

### Given a weight file, how do I know which code version and settings trained it?

Weight metadata carries `arch/scale/nc/imgsz/names` + provenance `version/repo/commit/describe`.
Three-level read fallback: **CLI explicit > metadata > filename/default**; a missing key =
unknown (never guessed). `version` is the release line, `commit` the exact coordinate — the two
need not agree, and `describe` spells out the gap.

### The container reports 208 cores via nproc — how many workers should I use?

A common container CPU lie: the affinity mask reports the host core count while the real quota is
a cgroup limit (measured 25 cores on this host). Pick workers by **measurement**:
`scripts/bench_io.py` (two stages: loader-only grid → end-to-end) — **a fast loader ≠ fast
training**, and more workers is not better.

### Under WSL2, elapsed times / speeds occasionally look wrong (negative)?

WSL2 clock sync makes `time.time()` jump. This project uses `time.monotonic()` everywhere; when
editing timing code, don't use `time.time()`.

### Docker build fails — what to do?

Follow the checkpoints in [docker.md](docker.md) §5 in order: base tag, Python version, the
TensorRT index install, apt package names. After a successful build the suggested verification
order: `pytest tests/` → single-image inference → `export.py --trt` → a short training run.

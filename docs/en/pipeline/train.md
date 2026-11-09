# Train

> Entry point: `python scripts/train.py --data <name|.yaml> [options]`.
> Data: [../datasets/README.md](../datasets/README.md), models:
> [../models/README.md](../models/README.md), formal evaluation: [eval.md](eval.md).

## 1. Five ways to run

```bash
# from scratch (default; the built-in default recipe, 100 epochs)
python scripts/train.py --data coco --name my-run --batch 32 --workers 8

# finetune (initialize from existing weights; --recipe can switch the finetune recipe)
python scripts/train.py --data coco --weights weights/yolo26n.safetensors --name ft

# resume (restores optimizer / EMA / epoch / schedule / RNG)
python scripts/train.py --data coco --resume runs/train/train-my-run/resume.pt

# switch architecture
python scripts/train.py --data coco --model flash-yolo --name flash-coco
python scripts/train.py --data coco --model yolo26-seg --name seg-coco --mask-ratio 4
python scripts/train.py --data coco --model yolov3-tiny --imgsz 416 --name v3tiny
python scripts/train.py --data coco --model yolov3-tiny-silu --imgsz 416 \
    --obj-iou-target --name v3tiny-silu

# YOLOE-26 PE linear probing (only the cv3 tails train; full finetune / from scratch not wired)
python scripts/train.py --data coco --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors --name yoloe-pe-coco
```

## 2. Recipes

Priority: `config/train.yaml` base values → recipe file overrides → CLI overrides.

| Recipe | Stage | Prerequisite | Epochs (n/s/m/l/x) |
|---|---|---|---|
| *(none)* | from scratch / finetune on own data | — | 100 |
| `yolo26-coco-ft` | official COCO stage (finetune) | **needs O365-pretrained weight init** | 245/70/80/60/40 |
| `yolo26-seg-coco-ft` | official seg COCO stage | needs `yolo26*-objv1-seg` (365 classes) + COCO polygons | 245/50/80/60/40 |
| `yolo26-o365-pt` | official Objects365 pretraining (detection) | needs Objects365 (unverified) | 150 |
| `yolo26-seg-o365-pt` | official Objects365 pretraining (segmentation) | needs Objects365v1 seg (unverified) | 150 |

```bash
python scripts/train.py --data coco --recipe yolo26-coco-ft \
    --weights weights/yolo26n-objv1-150.safetensors

python scripts/train.py --data coco --recipe yolo26-seg-coco-ft \
    --weights weights/yolo26n-objv1-seg.safetensors
```

- **From-scratch training uses only `default`**. `yolo26-coco-ft` halves the base lr and decays
  steeper — it is tuned for an "already converged" starting point; running it from scratch pins
  the model in a slow gear (measured ~43% lower at the same epoch).
- `yolo26-seg-coco-ft` explicitly sets `mask_ratio: 1`, `overlap_mask: true`,
  `copy_paste_mode: flip`; the semseg auxiliary loss is always enabled.
- The segmentation init is 365-class O365 weights; `--weights` loads by key+shape intersection
  (class-dependent heads rebuilt).

## 3. Common arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | — (required) | descriptor |
| `--name` | empty → `runs/train/train` | run-dir suffix; incremented if it exists |
| `--recipe` | built-in default | recipe name or `.yaml` path |
| `--model` | `yolo26` | `yolo26` / `yolo26-seg` / `flash-yolo` / `flash-yolo-seg` / `yolov3-tiny` / `yolov3-tiny-silu` / `yoloe-26` |
| `--scale` | `n` | yolo26-family scale; fixed for other architectures |
| `--pe` | false | YOLOE-26 text-mode PE linear probing (requires a text `--weights`; only the cv3 tails train) |
| `--text-encoder` | next to weights | MobileCLIP2-B `.safetensors` path used by `--pe` |
| `--mask-ratio` | 4 | segmentation GT downsample factor (`imgsz/mask_ratio`) |
| `--overlap-mask` | true | instance index map vs per-instance binary masks |
| `--cfg` | — | model yaml override |
| `--epochs` | 100 | total epochs |
| `--batch` | 16 | **physical** batch (global batch under DDP) |
| `--nbs` | 64 | nominal batch; `accum = max(round(nbs/batch),1)` |
| `--imgsz` | 640 | training input (v3 usually 416) |
| `--workers` | 8 | loader processes (machine-dependent) |
| `--seed` | 0 | global seed |
| `--deterministic` | true | bit-reproducible kernel switch |
| `--weights` | — | finetune initialization |
| `--resume` | — | `resume.pt` or a run directory |
| `--device` | auto | `cuda:0` / `cpu` / `0,1` (DDP) |
| `--amp` | false | fp16+GradScaler; **NaNs in from-scratch training, finetune only** |
| `--optimizer` | `musgd` | `musgd` (Muon+SGD) or `sgd` (darknet convention) |
| `--obj-iou-target` | false | v3 IoU-aware objectness |
| `--close-mosaic` | 10 | last N epochs without mosaic/mixup/cutmix/copy_paste (auto `min(N, epochs//5)`) |
| `--multi-scale` | 0.0 | input-size jitter ±r |
| `--copy-paste-mode` | `off` | `off` / `box` / `flip` (official segmentation CopyPaste) |
| `--freeze` | 0 | freeze the first N layers |
| `--single-cls` | false | single-class training |
| `--plots` | true | results/PR/confusion-matrix at the end (needs matplotlib; in `requirements-dev.txt`, installed on demand at runtime) |
| `--channels-last` | true | NHWC training |
| `--stop-after` | 0 | run only the first N epochs (schedule still uses `epochs`) |
| `--val-epochs` | 1 | 0 = never validate |
| `--limit` | 0 | training subset (first N images) |

Every other hyperparameter has a same-name CLI override — `--help` lists them all; defaults are
`config/train.yaml`.

## 4. Run artifacts (`runs/train/train-<name>/`)

| Artifact | Content |
|---|---|
| `weights/best.safetensors` | EMA weights picked by in-training metrics (formal numbers need a separate eval) |
| `weights/last.safetensors` · `best_raw.safetensors` | last epoch / non-EMA best |
| `weights/epochNNN.safetensors` | saved every `save_period` epochs (branching mid-run experiments) |
| `results.csv` | per-epoch training losses + in-training metrics + val losses |
| `diag/train_diag.csv` | every `diag_interval` steps: pre-clip gradient norm, per-group lr, loss breakdown |
| `samples/*.png` | augmentation sample grids |
| `results.png` / `PR_curve.png` / `confusion_matrix*.png` | end-of-training report |
| `meta.json` | environment + full config snapshot |
| `run.log` | console + file log |
| `resume.pt` | resume state |

Weight metadata: [../models/README.md](../models/README.md).

## 5. Reproducibility

Three levels:

| Level | Meaning | Guarantee |
|---|---|---|
| **Stable** | no divergence/NaN in any configuration | fp32 default + gradient clipping max_norm=10 |
| **Bit-identical** | same commit + same seed + same hardware/driver + same workers/batch → bit-identical weights | on by default (`deterministic: true`) |
| **Statistical** | across hardware/drivers → curves agree within noise | unquantified; A/B uses the same machine + `--stop-after` |

Bit-identical reproduction needs all of the following:

1. `deterministic: true` (default) — cuDNN autotune off + deterministic algorithms + fixed CUBLAS
   workspace.
2. `workers` / `batch` are part of the reproduction conditions (augmentation seeds derive from
   `(seed, epoch, worker_id)`).
3. RNG replays every epoch: seed = `seed + epoch × world_size + rank` (single-GPU:
   `seed + epoch`).
4. Resume is bit-equivalent: `resume.pt` stores the four RNG states + `ema.steps`; requires the
   same GPU count and device.

Bit-identical across machines is impossible (different kernels); only statistical agreement is
promised. New randomness must not call the global `np.random` / `random` directly. Gate:
`FLASH_YOLO_REPRO=1 pytest tests/test_train_repro.py`.

## 6. Tune I/O first

`--batch` / `--workers` are machine-dependent values, deliberately not in the default config:

```bash
python scripts/bench_io.py --data coco            # loader-only grid -> end-to-end top K
python scripts/bench_io.py --data coco --loader-only
```

- **A fast loader ≠ fast training**: only end-to-end numbers can decide.
- More workers is not better (filling all cores starves the main process).
- Never call `cv2.setNumThreads` inside a worker (OpenCV's thread pool is not fork-safe — the
  whole training hangs); thread counts are set by the parent process when `data/loader.py` is
  imported.

## 7. Multi-GPU training (DDP)

Training only; eval / export / infer remain single-process.

```bash
python scripts/train.py --data coco --device 0,1 --batch 32 --workers 8 --name ddp
# or: torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco --batch 32
```

- `--batch` is the **global** batch: per GPU = `batch // world_size`, must divide evenly.
- `--workers` is **per GPU**.
- This project's loss is in Σ_img units; DDP backward averages gradients per rank → the loss must
  be scaled by `world_size` (`train/distributed.py::ddp_scale_loss`; returned unchanged when
  world=1).
- Validation runs on rank 0 only, full set; checkpoints/CSV/diag/samples/plots/meta.json/run.log
  are written by rank 0 only.
- `self.model` / `self.head` are always the raw modules; DDP only stores `self.ddp_model`.
- No SyncBatchNorm: BN statistics are per-GPU; multi-GPU is not bit-identical with single-GPU,
  but repeated runs at the same world_size are bit-reproducible.
- `find_unused_parameters=True` (covers freeze and conditional branches).
- Non-boundary accumulation steps wrap forward+backward in `no_sync()`.
- Resume across GPU counts / devices works (`resume.pt` records world_size/device), but the
  trajectory is no longer bit-identical.
- Throughput: the data pipeline usually saturates first; 2×5090 / yolo26n@640 / batch 64 /
  workers 8 measures ~258 img/s vs 218 single-GPU (~1.2×). Use `bench_io.py` to pick the
  single-GPU values first.

## 8. Common gotchas

1. **`--amp` explodes in from-scratch training** (fp16 NaNs within epoch 1 at 640² + BN
   eps=0.001). Default is fp32; finetuning may try `--amp`.
2. **`--close-mosaic` means "last N epochs"**, not "starting at epoch N"; short runs get it
   scaled down by `min(N, epochs//5)`.
3. `best.safetensors` is picked by FastMetrics; external numbers go through `scripts/eval.py`.
4. Val losses cannot be compared directly with training losses (augmentation distribution / EMA
   vs raw / BN mode).
5. Training-log loss column names derive from `item_keys`: yolo26 → `box/cls/l1/o2m/o2o`,
   segmentation adds `seg/sem`, v3 → `box/obj/cls`.
6. From-scratch training at a different input size computes the bias prior at the training imgsz;
   v3 needs re-clustered anchors.
7. Segmentation accepts COCO json and YOLO-seg txt; a full fallback errors out.
8. **LVIS-style federated datasets**: each image annotates only a subset of classes. Training
   builds a per-image valid-class mask (`positives ∪ neg_category_ids`); unknown and
   `not_exhaustive_category_ids` classes are ignored instead of trained as negatives, and
   mosaic/mixup/cutmix/copy_paste are disabled (the mask is built from the central image).
   Plain COCO has no such fields, so batch shapes and the loss path are unchanged.

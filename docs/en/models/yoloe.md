# YOLOE-26 (open-vocabulary instance segmentation)

> Open-vocabulary segmentation: classes are not fixed at training time — the model answers from
> text prompts, visual prompts (a refer image + bboxes/masks), or a built-in **4,585-name vocabulary**
> at inference time. The current release supports **prompt-free (-pf)**, **text prompts**
> (custom class names encoded by the MobileCLIP2-B text encoder) and **visual prompts**
> (refer-image exemplars encoded by SAVPE) for inference / evaluation / export.
> Structure: `config/models/yoloe-26-seg.yaml`; the backbone/neck rows are identical to
> [yolo26-seg.md](yolo26-seg.md) — only the head line differs.

## 1. Conventions

- The official 26 family ships **seg checkpoints only**: `yoloe-26*-seg.pt` (text/visual modes) /
  `yoloe-26*-seg-pf.pt` (prompt-free); all three prompt modes are in scope on this page.
- Official LVIS minival numbers (paper protocol, **detection configuration**, 640 input):
  YOLOE-26n-pf **16.6 E2E / 17.7 Non-E2E** mAP50-95; text prompts 24.7 / visual prompts 21.9
  (Non-E2E). The paper's params/FLOPs are for the detection configuration — the released seg
  checkpoints also carry the mask branch / SAVPE / text projection and are larger (n-pf 7.84M);
  the official docs note this. FLOPs exclude the region-text similarity, so real cost grows with
  the prompt set.
- This repo's reproduction = same weights, same pre/post-processing: bit-identical with the
  official fp32 forward (§4). The 20-image COCO smoke is a pipeline check; the full
  5,000-image cross-implementation numbers are in §4.
- Text mode needs the MobileCLIP2-B text encoder weights (`mobileclip2_b.safetensors`, converted
  from Apple's `mobileclip2_b.ts` and provided through this repo's Releases) next to the model
  weights — it is loaded only when `set_classes` runs (infer/eval) and never enters the exported
  graph (the embeddings are fused into the cls conv at export). Visual mode needs no text
  encoder: refer-image bboxes/masks are rasterized on the P3 grid (1/8), and SAVPE aggregates
  the three feature scales into one L2-normalized vpe per class.
- "Official" refers to the Ultralytics-released YOLOE-26 weights/docs; this repo does not
  redistribute them.

## 2. Structure (n scale)

24 yaml modules (`model.model` 0–23), 346-layer module tree. The **pf layout** has
**7,837,694 params / 16.0 GFLOPs @640** (s: 18,419,972 params / 44.2 GFLOPs; above
yolo26n-seg's 9.3 GFLOPs — the vocab Linear over 8400 anchors × 4585 classes is the inherent pf
cost); the **text layout** has **5,615,540 params / 10.5 GFLOPs @640** at K=80 prompts (the
region-text similarity scales linearly with the prompt count; the official yaml comment's 11.7
includes the semseg forward, and the official docs' numbers are the detection configuration):

| Section | idx | Contents |
|---|---|---|
| Backbone | 0–10 | identical to yolo26-seg (Conv → C3k2 → SPPF → C2PSA) |
| Neck | 11–22 | identical to yolo26-seg (P3/P4/P5 scales) |
| Head | 23 | YOLOESegment26 (pf layout: truncated cv2/cv3 + lrpc baked vocab; text layout: full cv2/cv3 tails + cv4 contrastive head) + cv5/proto |

### Head (YOLOESegment26, pf layout)

- `cv2[i]` box branch (Conv+Conv, no tail — moved to `lrpc.loc`); `cv3[i]` embedding branch
  (two DWConv+Conv stages, no tail — moved to `lrpc.pf`); `cv5[i]` mask coefficients and `proto`
  (Proto26) are identical to yolo26-seg.
- `lrpc[i]` = `vocab` (Linear(c3→4585), baked vocabulary) + `pf` (proposal filter, channel 0
  only) + `loc` (box regression). **P5 (level 2) skips the proposal filter** (vocab stays a
  Conv2d, all anchors pass — a structural fact of the released weights).
- The vocabulary is **4,585 names (RAM++)**, stored in the weights metadata (`names`); class
  semantics come from the vocabulary, not from any dataset.
- `one2one_*` copies = end2end dual head; `reprta` / `savpe` are in the weights but unused by the
  pf forward (strict-load parity, like yolo26-seg's semseg).
- Proposal filter: conf=0.001 per-image sigmoid threshold + batch-union mask; **export = conf=0
  static path** (official behavior: all anchors pass, no dynamic mask). E2E output
  `(B, 1000, 6+nm)` — **max_det=1000** is the YOLOE convention (300 for the yolo26 family).
- Per-scale quirk (structural fact of the official assets): the n-scale pf conv has 512 channels
  (reuses the cv3 tail conv), the s/m/l/x scales have 1 (a dedicated single channel) —
  behaviorally identical (channel 0 only), expressed by the `pf_channels` scale data in the yaml.

### Head (YOLOESegment26, text layout)

- `cv2[i]` / `cv3[i]` keep their full tails (box regression / 512-d region embeddings);
  `cv4[i]` = BNContrastiveHead: BatchNorm (eps=0.001, a checkpoint fact) over region embeddings,
  cosine similarity against the L2-normalized text embeddings, scaled by a learned
  `exp(logit_scale)` + bias; `cv5`/`proto` identical to the pf layout. No `lrpc`.
- Text prompts flow through `set_classes`: CLIP BPE tokenize (77 context) → MobileCLIP2-B encode
  → `reprta` (SwiGLUFFN residual) → L2 normalize → cached on the head. The class count is the
  prompt count, per forward.
- Visual prompts: refer-image bboxes/masks are rasterized on the P3 grid (input 1/8) → `SAVPE`
  (mask-attention aggregation over the three feature scales) → one L2-normalized vpe per class
  → fed directly as the class embeddings into the same `BNContrastiveHead` (no `reprta`).
  Multiple boxes of one class are OR'ed; the class count is the prompt count.
- **max_det=300** for the text layout (the released text checkpoints carry the Detect default;
  the pf checkpoints carry 1000).
- The text encoder is external (63.4M params, not in the FLOPs count — the official does the
  same) and is fused into the cls conv at export, so the exported graph is class-locked and
  single-input.

## 3. Usage

```bash
# download the official .pt, then convert (one-off): unwrap to plain safetensors + vocab metadata
python scripts/convert_weights.py --src yoloe-26n-seg-pf.pt --dst weights/yoloe-26n-seg-pf.safetensors

# inference (labels = vocabulary names, traveling with the weights; boxes + masks via the yolo26-seg pipeline)
python scripts/infer.py --weights weights/yoloe-26n-seg-pf.safetensors --image assets/traffic.jpg

# evaluation (dataset names -> vocabulary: exact match first, vocab_synonyms fallback, missing = error)
python scripts/eval.py --weights weights/yoloe-26n-seg-pf.safetensors --data coco

# export (E2E fused graph + prototypes, two outputs; conf=0 static path)
python scripts/export.py --weights weights/yoloe-26n-seg-pf.safetensors --out weights/yoloe-26n-seg-pf.onnx

# ---- text mode (yoloe-26*-seg weights; needs mobileclip2_b.safetensors next to the weights) ----

# inference: custom class names / prompts encoded with the MobileCLIP2-B text encoder
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors --image assets/traffic.jpg \
    --classes "person, bicycle, car, bus, truck"

# evaluation: the dataset's own class names become the prompts
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis            # full LVIS val
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival    # official minival protocol

# export: the prompts are fused into the cls conv — a single-input, class-locked ONNX
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --out weights/yoloe-26n-3cls.onnx

# export the Non-E2E deploy graph (dense o2m output + external NMS)
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --nms --out weights/yoloe-26n-3cls-nms.onnx

# TensorRT (pf / text; the engine is bound to the build GPU + TRT version)
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --trt [--fp16]

# ---- visual mode (yoloe-26*-seg weights; refer image + bboxes/masks via JSON) ----

# prompt.json: {"image": "refer.jpg",
#               "prompts": [{"class": "person", "bbox": [x1,y1,x2,y2]},
#                           {"class": "person", "mask": "person.png"}]}
# each prompt uses exactly one of bbox / mask; masks match the refer image size
# (non-zero = foreground); multiple prompts of one class are OR'ed.

# inference: extract vpe from the refer image/prompts, then run on the target image (pt engine)
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors \
    --visual-prompt prompt.json --image assets/traffic.jpg

# evaluation: a single refer-image JSON, or a refer dataset (per-class exemplars)
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco \
    --visual-prompt prompt.json
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --refer-data lvis --refer-split train --refer-limit 1000

# export: vpe fused into the cls conv, class-locked (works with --nms / --trt too)
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --visual-prompt prompt.json --out weights/yoloe-26n-visual.onnx

# ---- mixed text + visual (text classes first, visual classes after — get_cls_pe order) ----
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "bicycle" --visual-prompt prompt.json --image assets/traffic.jpg
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco \
    --classes "bicycle" --visual-prompt prompt.json
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "bicycle" --visual-prompt prompt.json --out weights/yoloe-26n-mixed.onnx

# ---- PE linear probing (text mode; needs the official text checkpoint + mobileclip2_b) ----
# Dataset class names are the text prompts: the embeddings are fused into the cv3 tails and
# only those six convolutions train; backbone / neck / BN / mask branch / proto stay frozen.
# --text-encoder overrides the MobileCLIP2-B path.
python scripts/train.py --data coco --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 32 --imgsz 640 --lr0 0.1

# LVIS (federated masks turn on automatically; cross-image augmentation is disabled; lr0=0.01)
python scripts/train.py --data lvis --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 16 --imgsz 640 --lr0 0.01

# The output is class-locked (metadata: pe/fused/struct_nc/names); eval/export/infer consume it directly
python scripts/eval.py --weights runs/train/train-yoloe-pe-coco/weights/best.safetensors --data coco
python scripts/export.py --weights runs/train/train-yoloe-pe-coco/weights/best.safetensors \
    --out weights/yoloe-26n-coco-pe.onnx
```

- PE training accepts text-mode weights only (prompt-free is rejected, and `--model yoloe-26`
  without `--pe` errors out); the structural nc comes from the weight metadata (80 for the
  official checkpoints), while the prompt count is the dataset class count, so LVIS's 1,203
  classes do not change the cv3 body. `--freeze` cannot be combined with `--pe`, and the fused
  output is class-locked: passing `--classes` / `--visual-prompt` to eval/infer/export errors.
  PE `--resume` must also pass `--weights` (the original text checkpoint rebuilds the fused
  structure; `resume.pt` then restores the state). Suggested lr0: 0.1 for COCO's 80 classes,
  0.01 for LVIS's 1,203 classes (federated class masks turn on automatically; mosaic and the
  other cross-image augmentations are disabled for federated data).

- `--model/--scale` are inferred from the filename (`yoloe-26n-seg-pf` → yoloe-26 / n);
  `--raw` errors out (the pf raw output's column count varies with the proposal filter);
  `--nms` (o2m + class-aware NMS) is supported for **text mode**, while prompt-free remains E2E
  only. The official text/visual 24.7 figure is the Non-E2E protocol.
- COCO-vs-RAM++ naming differences (e.g. `airplane` → `plane`, `tv` → `television`) resolve
  through the dataset descriptor's `vocab_synonyms` field (11 entries built into
  `config/datasets/coco.yaml`, including the official vocabulary's own typo `dinning table`);
  evaluation keeps only dataset-class predictions, the `CocoEvaluator` contract is unchanged.
- Text mode notes: `--classes` takes comma-separated names (each becomes one prompt/class);
  LVIS prompts use each category's first synonym (`name.split("/", 1)[0]` — the official
  convention, so `config/datasets/lvis.yaml` stores them already split); the class-locked ONNX
  rejects new prompts (`is_fused` guard). Class-locked artifacts carry deploy metadata
  (arch/scale/names/end2end), so `--engine onnx|trt` works without
  `--model/--scale/--classes`; TRT uses a `<engine>.json` sidecar.
- Visual mode notes: JSON class order = first appearance (the vpe row order), bboxes are refer
  image pixels, masks match the refer image size, and relative paths resolve against the JSON
  directory; multiple prompts of one class are OR'ed into one mask. `--classes` combines with
  `--visual-prompt` for mixed prompts (text classes first, visual classes after; duplicate names
  error). Infer/eval require the pt engine (SAVPE runs in-graph), while the class-locked export
  runs on onnx/trt. `--refer-data` extracts one vpe per class from the refer set's GT boxes;
  classes without exemplars get zero embeddings (AP 0). The official public API has no mixed
  entry; this project extends the `get_cls_pe` semantics.
- LVIS evaluation runs on **faster_coco_eval `lvis_style`** (the descriptor's `eval_style: lvis`):
  it honors `neg_category_ids` / `not_exhaustive_category_ids`, uses maxDets=300 and reports
  APr/APc/APf. Pycocotools on LVIS does not produce the LVIS metric; add
  `--eval-max-dets 100` for a literal match with ultralytics' current val (it leaves
  faster_coco_eval's default maxDets untouched).
- Official weights (ultralytics assets):
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt` (etc.).

## 4. Verification

- **strict load**: the official -seg-pf checkpoints (all five scales) convert to safetensors
  with all keys matching (nothing missing, nothing extra); the built head's key set matches the
  official ckpt for both the pf and text layouts.
- **Bit-identical (pf)**: forward (E2E detections + prototypes) matches the official fp32 forward
  exactly on zeros and random inputs, all five scales (E2E (1,1000,38) and proto (1,32,160,160)
  via `torch.equal`).
- **Bit-identical (text)**: the whole prompt chain — CLIP BPE tokenize → MobileCLIP2-B encode →
  reprta+L2 — and the head forward (E2E detections + prototypes) match the official pipeline
  exactly, all five scales; the text encoder itself reproduces the official `mobileclip2_b.ts`
  bit-for-bit (L=77, the official input contract). The tokenizer matches the official CLIP fork
  exactly except for the ftfy mojibake repair (§5).
- **Export parity (pf)**: onnx vs pt (same conf=0 static path) agree on all valid detections
  above conf=0.25 — boxes/scores/coefficients < 1e-3, zero class flips (near-tied scores across
  4585 classes only occur in the low-conf tail, broken by the torch/ORT backends).
- **Export parity (text)**: the fused pt model vs unfused differ by 1.8e-6 and the onnx by 6e-5 —
  same detections above conf=0.25, zero class flips.
- **Non-E2E export parity (text)**: the `--nms` o2m graph matches the pt path above conf=0.25
  (boxes/scores < 1e-3, zero class flips); a 2-image coco-tiny eval matches to four decimals
  for both Box and Mask.
- **Visual prompts**: the vpe for the same refer image + bboxes is bit-identical to the official
  environment (`.pt` + `YOLOEVPSegPredictor`, maxdiff 0.0); the fused export vs pt gives 2/2
  detections, boxes <6.1e-5, scores <1.1e-6 and zero class flips; the refer-dataset aggregation
  matches single-image extraction exactly. Top-2 vs the official Non-E2E path: boxes <0.6px,
  scores ≤0.011 (the low-conf tail depends on the official predictor's NMS defaults).
- **Mask prompts**: the 1/8 rasterization is `torch.equal` to the official
  `LoadVisualPrompt(masks=...)`; the class-locked export matches the pt detections.
- **Mixed prompts**: the cached embedding is `cat([tpe, vpe])` (text first, the official
  `get_cls_pe` order); duplicate class names error out; the class-locked export matches pt.
- **Visual val (official default protocol)**: official `load_vp=True` uses the **evaluation set
  itself** as the refer set. On LVIS minival, the official 8.4.175 COCO table gives Box 0.187 /
  Mask 0.141; rescoring the official predictions under the same LVIS protocol gives Box 0.3557 /
  Mask 0.2826. This project, same protocol (same YOLO refer set, `--nms`, maxDets=100), gives
  Box **0.3534** / Mask **0.2805** (Δ 0.0023 / 0.0021) with APr/APc/APf within 0.008. The
  refer set is the YOLO-format GT of the same minival images (a local refer descriptor with
  the same images and labels). The 0.015 COCO-table Mask gap comes from the official table
  backend (its faster-coco-eval errored
  and the table likely fell back); the LVIS rescore is the primary alignment evidence. This
  self-refer protocol is transductive (test GT participates in the vpe), not the paper's 21.9
  independent-refer protocol; the paper's refer data/sampling is not documented publicly, so
  0.35 is not presented as 21.9.
- **TensorRT (fp32 / fp16)**: pf E2E, text E2E and text Non-E2E engines build and run through
  `infer/eval --engine trt`; deploy metadata travels in the `<engine>.json` sidecar. Text E2E
  fp16 vs pt fp32 (@320 smoke): 6/6 detections, boxes <0.35px, scores <3e-3, mask IoU ≥0.996;
  the text NMS fp32 2-image eval is Box 0.2651 vs pt 0.2656 with identical Mask. E2E top-k is
  sensitive to accumulation differences (different low-conf candidates even in fp32), so
  cross-implementation comparisons still use Non-E2E.
- **COCO smoke** (20 images, pipeline-validation protocol, pf): Box 0.3485 / Mask 0.2865
  mAP50-95.
- **LVIS protocol**: `tests/test_lvis_evaluator.py` verifies the three LVIS rules (neg /
  not_exhaustive / unlisted pairs) on synthetic data, plus zero-detection images, APr/APc/APf
  and the segm RLE path; `--data lvis-minival` runs Box + Mask end to end.
- Regression gates in `tests/test_yoloe.py` (structure / conversion metadata / strict load /
  bit-parity / vocabulary mapping / export parity / text-mode parity / fuse equivalence /
  class-locked consumption / Non-E2E / visual prompts incl. masks and mixed),
  `tests/test_visual_prompt.py` (JSON contract / 1/8 rasterization / refer aggregation),
  `tests/test_text_encoder.py` (encoder structure + ts bit-parity, tokenizer goldens),
  `tests/test_yoloe_train.py` (training forward contract / fused-vs-unfused numerics / PE
  parameter boundary / fused checkpoint rebuild) and `tests/test_yoloe_trt.py` (engine
  acceptance, gated by `FLASH_YOLO_TRT=1`).

### Cross-implementation check (LVIS minival / COCO val2017)

Same released `yoloe-26n-seg` weights, text prompts, 640 input, and the same scoring protocol on
both sides (LVIS: faster_coco_eval `lvis_style`, maxDets=100; COCO: COCO protocol, maxDets=100).
`--nms` selects Non-E2E (o2m + class-aware NMS); the default is E2E.

| Dataset | Protocol | Ours Box | Ours Mask | Ultralytics Box | Ultralytics Mask |
|---|---|---:|---:|---:|---:|
| LVIS minival | E2E | 0.2125 | 0.1754 | 0.2231 | 0.1828 |
| LVIS minival | Non-E2E | **0.2218** | **0.1817** | **0.2228** | **0.1826** |
| COCO val2017 | E2E | 0.2832 | 0.2337 | 0.2952 | 0.2408 |
| COCO val2017 | Non-E2E | **0.2952** | **0.2407** | **0.2952** | **0.2407** |

Reproduce with:

```bash
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --eval-max-dets 100 [--nms]
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco [--nms]
```

- Non-E2E agrees with Ultralytics within 0.001 on LVIS and to four decimals on COCO; AP50,
  APr/APc/APf and the COCO area buckets differ by at most ~0.003.
- Full-val E2E is more sensitive to the 300-candidate top-k: the same weights can differ by
  ~0.01 even though the fp32 forward is bit-identical on identical inputs (above). Use Non-E2E
  as the cross-implementation reference.
- The official 24.7 (LVIS minival, text prompts) is the paper's **detection-configuration**
  Non-E2E number. Ultralytics' own val of the released **seg** checkpoint reports 0.2228; this
  repo reports 0.2218 under the same protocol — the implementation matches, and the remaining
  gap to 24.7 is the paper's detection configuration vs the released segmentation checkpoint.
- LVIS maxDets 100 and 300 give identical results on minival; COCO uses the standard maxDets=100.

### PE linear probing (COCO val2017)

Starting from the official `yoloe-26n-seg` text checkpoint, `fuse_text` + only the six cv3
tail convolutions train (full COCO train2017, 10 epochs, lr0 0.1 / batch 32 / 640). Formal
evaluation (pycocotools, all 5,000 images):

| Protocol | Box | Mask | Zero-shot text baseline Box / Mask |
|---|---:|---:|---:|
| E2E | **0.3089** | **0.2509** | 0.2832 / 0.2337 |
| Non-E2E | **0.3211** | **0.2570** | 0.2952 / 0.2407 |

- The fused initialization matches the unfused text path on the first 500 images
  (Box 0.3338 / Mask 0.2790): PE starts at the zero-shot baseline, so the gains above come
  entirely from the six tail convolutions.
- Training metrics were still rising at epoch 10; 10 epochs is not a saturation point.
- Deployment: the same fused weights score Box 0.4031 (pt) vs 0.4032 (exported ONNX) on the
  first 100 COCO images (E2E); the class-locked graph runs with `--engine onnx` and no
  `--classes`.
- Scales and single-GPU options: all five scales pass the strict-load → fuse → six-tail
  backward contract; a one-epoch GPU smoke passes with
  `--amp --multi-scale --single-cls --channels-last` (PE defaults to fp32; AMP is optional).
- **LVIS (1,203 classes, federated)**: each image annotates only a subset of classes.
  Training builds a valid-class mask (`positives ∪ neg_category_ids`); unknown and
  `not_exhaustive_category_ids` classes are ignored by the BCE instead of trained as
  negatives, and mosaic/mixup/cutmix/copy_paste are disabled for federated data. PE for
  10 epochs (lr0=0.01) on minival: E2E Box **0.2346** / Mask **0.1905**, Non-E2E
  **0.2436** / **0.1955** (zero-shot 0.2125 / 0.1754 and 0.2218 / 0.1817). E2E frequency
  buckets: APr 0.1180 / APc 0.2146 / APf 0.2733 (zero-shot 0.1923 / 0.1999 / 0.2272) —
  overall above the baseline, rare classes recovered from 0 but still below zero-shot, with
  gains concentrated in common/frequent. Negative sampling / class balancing is the
  follow-up for rare classes. The fused initialization matches the text path
  (E2E 0.2124 / 0.1753, APr 0.1922).
- Reproduce: `python scripts/train.py --data coco --model yoloe-26 --scale n --pe
  --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 32 --lr0 0.1`; evaluate the
  produced `best.safetensors` with `scripts/eval.py` (E2E / `--nms`).

## 5. Known limits

- Visual-prompt masks must match the refer image size (binary PNG); each prompt uses exactly
  one of bbox/mask, and multiple prompts of one class are OR'ed.
- Mixed text + visual: infer/export support it directly; eval uses `--classes` for the text
  subset (all other classes get zero embeddings), and text/visual class names must be distinct.
  The official public API has no mixed entry; this project extends the `get_cls_pe` semantics.
- `--refer-data` needs the same class names as the evaluation set; classes without exemplars
  get zero embeddings (AP 0). The official visual val defaults to `load_vp=True`, i.e. the
  evaluation set itself; this project uses `--refer-data <same dataset>` for that protocol (§4).
- Text-mode pt paths need `--classes` and the MobileCLIP2-B text encoder; `onnx` / `trt`
  consume only a class-locked graph (`export --classes` / `--visual-prompt` / mixed; class
  names travel in the ONNX `metadata_props` / TRT sidecar JSON, so `--classes` is optional and
  must match the locked list when given). For evaluation the locked classes must cover every
  dataset class (use a subset descriptor for a subset export; missing classes error out rather
  than being silently dropped). Older exports without metadata must be re-exported.
- The tokenizer skips the official fork's `ftfy` cleanup (mojibake repair / fullwidth folding):
  bit-identical on clean UTF-8 text, not guaranteed on broken-encoding inputs.
- The exported graph and the runtime pt path differ in the proposal filter (official behavior:
  export conf=0 static vs runtime conf=0.001) — a few low-conf detections may differ; valid
  detections above conf=0.25 match in practice.
- Parameter counts follow the strict-load-consistent `parameters()`; the text layout's 5,615,540
  already includes savpe/reprta, while the smaller figure in the official docs is the
  **detection-configuration** count and describes a different structure.
- **Training covers PE linear probing only**: `--model yoloe-26 --pe --weights ...` trains
  the six cv3 / one2one_cv3 tail convolutions (backbone, neck, BN, mask branch and proto stay
  frozen; the loss is the detection trio). Full finetuning and from-scratch training are not
  wired; prompt-free weights are inference-only (training errors out explicitly). The PE output
  is class-locked and carries its prompt names in metadata, so evaluation must use the matching
  dataset descriptor.
- **LVIS PE uses federated class masks**: LVIS annotates only part of the categories per image;
  training treats `positives ∪ neg_category_ids` as valid classes, ignores
  `not_exhaustive_category_ids` and unlabeled classes in the BCE, and disables
  mosaic/mixup/cutmix/copy_paste. Overall metrics can exceed zero-shot, but rare-class AP may
  still trail it (gains concentrate in common/frequent). Negative sampling / class balancing
  is the follow-up for rare classes; see §4.

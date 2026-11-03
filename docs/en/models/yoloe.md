# YOLOE-26 (open-vocabulary instance segmentation)

> Open-vocabulary segmentation: classes are not fixed at training time — the model answers from
> text / visual prompts or a built-in **4,585-name vocabulary** at inference time. v1.1.0 supports
> the **prompt-free (-pf)** inference / evaluation / export; text and visual prompt modes come in
> later releases. Structure: `config/models/yoloe-26-seg.yaml`; the backbone/neck rows are
> identical to [yolo26-seg.md](yolo26-seg.md) — only the head line differs.

## 1. Conventions

- The official 26 family ships **seg checkpoints only**: `yoloe-26*-seg.pt` (text mode) /
  `yoloe-26*-seg-pf.pt` (prompt-free, the scope of this page).
- Official LVIS minival numbers (paper protocol, **detection configuration**, 640 input):
  YOLOE-26n-pf **16.6 E2E / 17.7 Non-E2E** mAP50-95; text prompts 24.7 / visual prompts 21.9
  (Non-E2E). The paper's params/FLOPs are for the detection configuration — the released seg
  checkpoints also carry the mask branch / SAVPE / text projection and are larger (n-pf 7.84M);
  the official docs note this. FLOPs exclude the region-text similarity, so real cost grows with
  the prompt set.
- This repo's reproduction = same weights, same pre/post-processing: bit-identical with the
  official fp32 forward (§4); the COCO numbers are pipeline-smoke (20 images), not accuracy
  conclusions.
- "Official" refers to the Ultralytics-released YOLOE-26 weights/docs; this repo does not
  redistribute them.

## 2. Structure (prompt-free layout, n scale)

24 yaml modules (`model.model` 0–23), 346-layer module tree, **7,837,694 params / 16.0 GFLOPs
@640** (s: 18,419,972 params / 44.2 GFLOPs; above yolo26n-seg's 9.3 GFLOPs — the vocab Linear
over 8400 anchors × 4585 classes is the inherent pf cost):

| Section | idx | Contents |
|---|---|---|
| Backbone | 0–10 | identical to yolo26-seg (Conv → C3k2 → SPPF → C2PSA) |
| Neck | 11–22 | identical to yolo26-seg (P3/P4/P5 scales) |
| Head | 23 | YOLOESegment26 (pf layout): truncated cv2/cv3 + lrpc baked vocab + cv5/proto |

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
  (reuses the cv3 tail conv), the s-scale has 1 (a dedicated single channel) — behaviorally
  identical (channel 0 only), expressed by the `pf_channels` scale data in the yaml.

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
```

- `--model/--scale` are inferred from the filename (`yoloe-26n-seg-pf` → yoloe-26 / n);
  `--nms` / `--raw` do not apply (they error out — the pf raw output's column count varies with
  the proposal filter).
- COCO-vs-RAM++ naming differences (e.g. `airplane` → `plane`, `tv` → `television`) resolve
  through the dataset descriptor's `vocab_synonyms` field (11 entries built into
  `config/datasets/coco.yaml`, including the official vocabulary's own typo `dinning table`);
  evaluation keeps only dataset-class predictions, the `CocoEvaluator` contract is unchanged.
- Official weights (ultralytics assets):
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt` (etc.).

## 4. Verification

- **strict load**: the official n/s -seg-pf checkpoints convert to safetensors with all 928 keys
  matching (nothing missing, nothing extra); the built head's key set matches the official ckpt
  for both the pf and text layouts.
- **Bit-identical**: forward (E2E detections + prototypes) matches the official fp32 forward
  exactly on zeros and random inputs, both scales (E2E (1,1000,38) and proto (1,32,160,160) via
  `torch.equal`).
- **Export parity**: onnx vs pt (same conf=0 static path) agree on all valid detections above
  conf=0.25 — boxes/scores/coefficients < 1e-3, zero class flips (near-tied scores across 4585
  classes only occur in the low-conf tail, broken by the torch/ORT backends).
- **COCO smoke** (20 images, pipeline-validation protocol): Box 0.3485 / Mask 0.2865 mAP50-95.
- Regression gates in `tests/test_yoloe.py` (structure / conversion metadata / strict load /
  bit-parity / vocabulary mapping / export parity).

## 5. Known limits

- The text / visual prompt forward is not implemented yet (text-layout weights build and load;
  the forward raises a clear error).
- The exported graph and the runtime pt path differ in the proposal filter (official behavior:
  export conf=0 static vs runtime conf=0.001) — a few low-conf detections may differ; valid
  detections above conf=0.25 match in practice.
- The official yaml comment's parameter count (5,615,540) is stale (excludes savpe/reprta); the
  numbers on this page are the strict-load-consistent `parameters()` counts.

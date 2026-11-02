# YOLO format

Shared descriptor fields: [README.md](README.md). The YOLO format uses `labels/*.txt`; there is
no json annotation file.

## 1. Layout and descriptor

```text
<root>/
├── images/train/      # images (or a .txt image list)
├── images/val/
├── labels/train/      # txt files matching image stems
└── labels/val/
```

```yaml
format: yolo
path: /path/to/mydata
names: [cat, dog]
train: {images: images/train}     # labels default = images -> labels
val:   {images: images/val}
```

- `labels` is optional: when omitted, the `images` segment in the `images` path is replaced with
  `labels`.
- `images` may also be a `.txt` image list (one path per line).
- The `ann` key is **forbidden** (errors out).

## 2. Detection labels

Each line is `cls xc yc w h`, coordinates normalized to `[0,1]`; an empty file = all background.

**Two-level label lookup fallback**:

1. `<stem>.txt` first;
2. on miss, fall back to `str(int(stem)).txt` — compatible both with dataflow-cv `coco2yolo`
   naming by **COCO image id** (`000000397133.jpg` → `397133.txt`) and with `1.txt ↔ 000001.jpg`.

**All-background guard**: zero hits across the whole directory errors out — otherwise training
would silently run on 100% background.

## 3. Segmentation labels

For `--model yolo26-seg` / `flash-yolo-seg`:

- **Polygon rows**: `cls x1 y1 x2 y2 ...` (normalized, >= 3 points); boxes are the polygon
  bounding box.
- **bbox row fallback**: 5 columns `cls xc yc w h` are treated as rectangles (mixed annotations
  allowed), counted in `n_seg_fallback` and shown in the scan line.
- The detection parser errors on >5 columns: `polygon labels require a segmentation dataset/model`.
- Pixel GT depends on image size: `targets()` / `gt_dict()` work only after `load_image(idx)` has
  recorded `(h, w)` (the eval loop satisfies this naturally; a custom script fetching GT first
  gets an explicit error).

## 4. Conversion

```bash
# COCO -> YOLO detection (dataflow-cv; writes only labels/ and classes.txt)
dataflow-cv convert coco2yolo <json> <out-yolo>

# COCO -> YOLO-seg (this repo's script; polygons follow the official merge convention)
python scripts/convert_coco_seg.py --src <json> --dst <out-yolo>/labels
```

dataflow-cv's `coco2yolo` currently writes bbox only, not polygon rows; use
`convert_coco_seg.py` for segmentation.

## 5. Gotchas

- YOLO has no crowd concept (conversion drops it); COCO keeps crowd on val, so the val numbers
  differ only by this one item.
- `--limit N` takes the first N in filename order, unlike COCO's json order; cross-format
  comparisons need the full set or aligned subsets.
- Zero hits across the whole directory errors out (all-background guard) — don't train with an
  empty labels directory.
- A full segmentation fallback errors out; mixed annotations allow some bbox rows, but there must
  be at least one usable polygon/box.

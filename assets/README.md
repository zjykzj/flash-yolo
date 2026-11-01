# Assets

Demo images in this directory are samples from the COCO dataset (val2017), picked for
demo-friendly scenes; the `*_det.jpg` counterparts are detections rendered by `scripts/infer.py`
with the **300-epoch from-scratch flash-yolo checkpoint** (COCO train2017, 35.14 mAP@[.5:.95],
E2E); the `*_seg.jpg` counterparts are instance segmentation rendered with the **300-epoch
from-scratch flash-yolo-seg checkpoint** (Box 35.05 / Mask 29.98 mAP@[.5:.95]). Both use 640 input
and the default inference threshold (conf 0.25). Rendered with:

```bash
# detection checkpoint (flash-yolo)
python scripts/infer.py --weights flash-yolo-e300-640.safetensors --image assets/traffic.jpg
python scripts/infer.py --weights flash-yolo-e300-640.safetensors --image assets/baseball.jpg

# segmentation checkpoint (flash-yolo-seg)
python scripts/infer.py --weights flash-yolo-seg-e300-640.safetensors --image assets/traffic.jpg
python scripts/infer.py --weights flash-yolo-seg-e300-640.safetensors --image assets/baseball.jpg
```

- `traffic.jpg` — COCO val2017 `000000336232.jpg` (dense street traffic: bus, cars, motorcycle, people)
- `traffic_det.jpg` — detections rendered on `traffic.jpg`
- `traffic_seg.jpg` — instance segmentation rendered on `traffic.jpg`
- `baseball.jpg` — COCO val2017 `000000192670.jpg` (baseball scene: batter, catcher, spectators)
- `baseball_det.jpg` — detections rendered on `baseball.jpg`
- `baseball_seg.jpg` — instance segmentation rendered on `baseball.jpg`

## Animated demo

`demo_det.gif` / `demo_seg.gif` — Flash-YOLO / Flash-YOLO-seg inference on a street clip
(300-epoch from-scratch checkpoints: 35.14 mAP@[.5:.95] detection; Box 35.05 / Mask 29.98
segmentation).

- Source: <https://www.pexels.com/video/15247002/> (Pexels license: free to use, attribution not
  required)
- Segment: 23.5–31.5 s of the 1920×1080 / 24 fps original, downscaled to 960×540 for inference
- Output: 520×292, 10 fps, 80 frames, 96-color GIF (~4.2 MB / ~3.9 MB)
- The original video and intermediate mp4 files are not stored in the repository.

Rendered with:

```bash
# detection checkpoint
python scripts/infer.py --weights flash-yolo-e300-640.safetensors \
    --video demo_clip_960x540.mp4 --output annotated_det.mp4
# segmentation checkpoint
python scripts/infer.py --weights flash-yolo-seg-e300-640.safetensors \
    --video demo_clip_960x540.mp4 --output annotated_seg.mp4

# GIF: fps=10, scale=520:292, 96-color palette (ffmpeg palettegen/paletteuse)
```

Usage:

- README visuals: `demo_det.gif` / `demo_seg.gif` in the hero area; the `*_det.jpg` / `*_seg.jpg`
  renders in the Detection / Segmentation usage sections; demo input for `scripts/infer.py`
- Real-image test data for `tests/test_export_parity.py` (onnx vs pt consistency, `traffic.jpg`)

Note: COCO images belong to their original photographers (per-image Flickr licenses; the COCO
annotations are CC-BY 4.0). These files are used for demonstration and testing only.

# Assets

Demo images in this directory are samples from the COCO dataset (val2017), picked for
demo-friendly scenes; the `*_det.jpg` counterparts are detections rendered by `scripts/infer.py`
with the **100-epoch from-scratch flash-yolo checkpoint** (COCO train2017, 33.77 mAP@[.5:.95]);
640 input, default inference threshold (conf 0.25). Rendered with:

```bash
python scripts/train.py --data coco --model flash-yolo --name flash-coco-e100 --batch 64 --workers 16
python scripts/infer.py --weights runs/train/flash-coco-e100/weights/best.safetensors --image assets/traffic.jpg
python scripts/infer.py --weights runs/train/flash-coco-e100/weights/best.safetensors --image assets/baseball.jpg
```

- `traffic.jpg` — COCO val2017 `000000336232.jpg` (dense street traffic: bus, cars, motorcycle, people)
- `traffic_det.jpg` — detections rendered on `traffic.jpg`
- `baseball.jpg` — COCO val2017 `000000192670.jpg` (baseball scene: batter, catcher, spectators)
- `baseball_det.jpg` — detections rendered on `baseball.jpg`

Usage:

- Hero images in the README (the `*_det.jpg` renders) and demo input for `scripts/infer.py`
- Real-image test data for `tests/test_export_parity.py` (onnx vs pt consistency, `traffic.jpg`)

Note: COCO images belong to their original photographers (per-image Flickr licenses; the COCO
annotations are CC-BY 4.0). These files are used for demonstration and testing only.

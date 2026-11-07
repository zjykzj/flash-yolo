# Infer

> Entry point: `python scripts/infer.py --weights <w> --image <file|directory>`, or
> `--video <file|URL|camera index>`. The three runtimes (pt / onnx / trt) share one
> pre/postprocessing; image mode writes a visualization + YOLO txt labels, video mode writes an
> annotated video file.

## 1. Usage

```bash
# single image (E2E, NMS-free by default)
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/traffic.jpg

# directory batch (one level, sorted by filename)
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/

# switch path / engine / decode path
python scripts/infer.py --weights yolo26n.safetensors --image img.jpg --nms
python scripts/infer.py --weights weights/yolo26n.onnx --engine onnx --image img.jpg
python scripts/infer.py --weights weights/yolo26n.engine --engine trt --image img.jpg

# models trained on non-COCO data: the descriptor brings nc and label names
python scripts/infer.py --weights <own weights>.safetensors --data mydataset --image img.jpg

# segmentation weights: arch auto-detected, instance masks overlaid
python scripts/infer.py --weights weights/yolo26n-seg.safetensors --image assets/traffic.jpg

# yoloe prompt-free: labels come from the baked 4,585-name vocabulary in the weights metadata
python scripts/infer.py --weights weights/yoloe-26n-seg-pf.safetensors --image assets/traffic.jpg
# yoloe text prompts: class names encoded with the MobileCLIP2-B text encoder (pt engine only)
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors --image assets/traffic.jpg \
    --classes "person, bicycle, car, bus, truck"

# video file (output <run_dir>/<source>.mp4, default runs/predict/predictN/)
python scripts/infer.py --weights weights/yolo26n.safetensors --video /path/to/video.mp4
# camera / RTSP stream: cap with --max-frames (0 = until EOF; Ctrl-C exits cleanly)
python scripts/infer.py --weights weights/yolo26n.safetensors --video 0 --max-frames 300
python scripts/infer.py --weights weights/yolo26n.safetensors --video rtsp://host/stream --max-frames 600
# window preview (needs GUI; press q to quit) + explicit output file
python scripts/infer.py --weights weights/yolo26n.safetensors --video /path/to/video.mp4 \
    --show --output out.avi
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--image` | — | image file or directory (mutually exclusive with `--video`) |
| `--video` | — | video file / stream URL / camera index (`0`) |
| `--max-frames` | 0 | process at most N frames in video mode (explicit for streaming sources) |
| `--show` | off | live video preview (needs GUI) |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--conf` | **0.25** | confidence threshold (eval defaults to 0.001 — different purposes) |
| `--nms` | off | o2m + NMS (yolo26 family + yoloe-26 text; errors for v3, prompt-free is E2E only) |
| `--agnostic-nms` | off | cross-class NMS (requires `--nms`) |
| `--classes` | — | yoloe-26 text mode: comma-separated class names (pt engine only) |
| `--model` / `--scale` | inferred from filename/metadata | errors when nothing can be inferred |
| `--output` | `runs/predict/predictN/` | image: file or directory; video: output video file |
| `--data` | COCO labels | only `names` + `nc`; no images read (ignored for yoloe) |
| `--imgsz` | metadata > 640 | override the input size |
| `--device` | auto | pt engine device |

## 2. Output

```text
image 1/1 assets/traffic.jpg: 640x427 4 persons, 17 cars, 1 motorcycle, 1 bus, 1 truck, 15.0ms
Speed: 5.9ms preprocess, 8.9ms inference, 0.2ms postprocess per image at shape (1, 3, 640, 640)
Total: 25.8ms end-to-end for 1 image(s) (incl. image read + save)
Visualization -> runs/predict/predict21
Labels        -> runs/predict/predict21/labels
```

- One line per image: original size + class counts + pure inference time; `Speed` is the
  preprocess / inference / postprocess segments (the engine init does 3 warmups, numbers are
  steady-state); `Total` includes image read and save.
- Image artifacts: `<name>.jpg` visualization + `labels/<name>.txt` (YOLO format
  `cls xc yc w h`, normalized).
- Segmentation weights: semi-transparent instance masks overlaid; `labels/*.txt` writes
  **normalized polygon rows** `cls x1 y1 x2 y2 ...` (outer contour, connected components merged
  by nearest point, skipped under 3 points — the same convention as the official `save_txt`).
- Video mode **does not write `labels/`** (per-frame txt has no stable naming semantics); the
  codec follows the suffix: `.avi`/`.mjpg` use MJPG, everything else mp4v; when writing fails it
  errors and suggests switching to `.avi`.
- Camera/stream output size is decided on the **first frame**; `--show` on a headless server
  warns once and disables the preview without affecting the write.

## 3. Engine choice

| Engine | When | Notes |
|---|---|---|
| `pt` | default, alignment/acceptance | GPU/CPU; the model line uses the module-tree convention |
| `onnx` | CPU deployment (ORT) | input size decided by the graph shape |
| `trt` | GPU deployment | engine bound to the build machine's GPU + TRT version; rebuild on a new machine |

## 4. YOLOE (open-vocabulary)

- **prompt-free (`-pf` weights)**: labels are the baked vocabulary names traveling in the
  weights metadata — no `--data` / `--classes` needed; all three engines work.
- **text prompts (`-seg` weights)**: `--classes "a, b, c"` encodes each name with the
  MobileCLIP2-B text encoder (`mobileclip2_b.safetensors` must sit next to the model weights —
  loaded once at `set_classes`, not part of the model graph). Requires `--engine pt`;
  the class-locked ONNX produced by `export --classes` is not wired into the CLI engines yet.
- `--raw` errors out for yoloe; `--nms` is supported for yoloe-26 text mode (o2m + class-aware
  NMS), while prompt-free (lrpc) remains E2E only (the raw column count varies with the
  proposal filter).
- Model details: [../models/yoloe.md](../models/yoloe.md).

## 5. Gotchas

- **conf default differs from eval** (0.25 vs 0.001).
- v3 has a single "decode + obj×cls + NMS" path; `--nms` / `--raw` error out.
- Directory mode scans one level (no recursion); unreadable images warn and skip without
  stopping the batch.
- **`--output` image semantics**: a single image with an image suffix (or an existing file) is
  treated as a file and the filename is honored; a directory / no suffix / non-image suffix is
  treated as a directory and the source filename is kept. Directory input accepts directory
  output only.
- Class-name priority: `--data` / weight-metadata names > COCO default; unmatched names get an
  index label.
- Video output size is decided on the first frame; streaming sources should pass
  `--max-frames`; Ctrl-C releases resources in finally.

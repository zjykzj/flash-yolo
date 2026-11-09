"""单图 / 目录 / 视频流推理 + 可视化（pt / onnx / trt；YOLO26 / YOLO26-seg / YOLOv3-tiny）

用法:
    python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/traffic.jpg
    python scripts/infer.py --weights yolo26n.onnx --engine onnx --image assets/     # 目录批量
    python scripts/infer.py --weights weights/yolo26n.safetensors --image traffic.jpg --nms  # o2m+NMS 路径
    python scripts/infer.py --weights weights/yolov3-tiny.safetensors --image traffic.jpg  # v3（档位由架构固定）
    python scripts/infer.py --weights weights/yolo26n.safetensors --video /path/to/video.mp4  # 视频文件
    python scripts/infer.py --weights weights/yolo26n.safetensors --video 0 --max-frames 300  # 摄像头/流
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import cv2
import numpy as np
import torch

from config import __version__
from config.datasets import load_dataset, load_names
from model.build import ARCHS, YOLO26_FAMILY, YOLOE26_FAMILY, arch_display_name
from model.weights import arch_has_scale, load_meta, prompt_mode_from_weights, resolve_arch_scale, resolve_imgsz
from utils.engines import OnnxEngine, PtEngine, TRTEngine, device_label, resolve_device
from utils.logger import attach_file_log, bold, get_logger, log_params, setup_logging
from utils.paths import increment_path
from utils.postprocess import masks_to_segments
from utils.visualize import draw_detections

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".mjpg"}


def _collect_images(source):
    """单图路径或目录 -> 图片路径列表（目录模式按文件名排序）"""
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
    return [p]


def _resolve_image_output(output, multi):
    """解析图片模式的 --output。

    Returns:
        (run_dir, out_file)：out_file 为 None 时文件名沿用源图名（目录模式/默认）；
        单图给显式文件路径时返回该路径，保存时用用户指定的文件名。

    规则：目录模式只接受目录；单图模式下，已存在的文件或图片扩展名的路径按文件处理，
    已存在的目录 / 无后缀 / 非图片后缀按目录处理（支持 `run.v1` 这类带点的目录名）。
    """
    out = Path(output).resolve()
    if multi:
        if out.exists():
            if out.is_dir():
                return out, None
            raise ValueError(f"--output for a directory input must be a directory: {out}")
        if out.suffix.lower() in _IMG_EXTS:
            raise ValueError(f"--output for a directory input must be a directory, got image path: {out}")
        return out, None
    if out.is_dir():
        return out, None
    if out.exists() or out.suffix.lower() in _IMG_EXTS:
        return out.parent, out
    return out, None


def _video_stem(source):
    """视频源 -> 默认输出名：相机索引 -> cameraN，URL/流 -> stream，文件 -> stem"""
    text = str(source)
    if text.isdigit():
        return f"camera{text}"
    if "://" in text:
        return "stream"
    return Path(text).stem or "video"


def _open_video(source):
    """打开视频文件 / 流 URL / 相机索引 -> (cap, info)；打不开直接报错（在建 run 目录/引擎之前）"""
    text = str(source)
    cap = cv2.VideoCapture(int(text) if text.isdigit() else text)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video source: {source!r} "
                                f"(expected a video file, a stream URL, or a camera index)")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0  # 部分相机/流不上报帧率
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    return cap, info


def _video_writer(path, fps, size):
    """按后缀选编码：.avi/.mjpg -> MJPG，其余 -> mp4v；打不开时报错并提示换 .avi"""
    codec = "MJPG" if path.suffix.lower() in {".avi", ".mjpg"} else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path} (codec {codec}); try an .avi output")
    return writer


def _run_video(engine, names, cap, info, args, imgsz, out_path):
    """逐帧推理 + 画框 + 写视频；返回处理帧数（None = 没读到帧）

    - 输出尺寸在首帧确定（相机/流打开时 CAP_PROP 可能还是 0）
    - 流式源用 --max-frames 限流；Ctrl-C 干净退出（释放 capture/writer）
    - 不写逐帧 txt 标签：视频没有稳定的文件命名语义（图片模式才写 labels/）
    """
    stage_sums = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    writer = None
    n_done = 0
    t_start = time.perf_counter()
    show = args.show
    try:
        while True:
            if args.max_frames and n_done >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            n_done += 1
            dets, tseg = engine.predict_timed(frame, conf_thres=args.conf)
            for k in stage_sums:
                stage_sums[k] += tseg[k]
            annotated = draw_detections(frame, dets, names)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = _video_writer(out_path, info["fps"], (w, h))
                logger.info(f"output: {out_path} · {w}x{h} @ {info['fps']:.1f} fps")
            writer.write(annotated)
            if show:
                try:
                    cv2.imshow("flash-yolo", annotated)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("preview quit (q)")
                        break
                except cv2.error:
                    logger.warning("--show requires a display; continuing without preview")
                    show = False
            if n_done % 30 == 0:
                fps_now = n_done / max(time.perf_counter() - t_start, 1e-6)
                logger.info(f"frame {n_done} · {fps_now:.1f} fps processing · "
                            f"{len(dets.boxes)} detection{'s' if len(dets.boxes) != 1 else ''} · "
                            f"{sum(tseg.values()):.1f}ms")
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if n_done == 0:
        logger.warning("no frames processed")
        return None
    elapsed = time.perf_counter() - t_start
    avg = {k: v / n_done for k, v in stage_sums.items()}
    logger.info(f"Speed: {avg['preprocess']:.1f}ms preprocess, {avg['inference']:.1f}ms inference, "
                f"{avg['postprocess']:.1f}ms postprocess per frame at shape (1, 3, {imgsz}, {imgsz})")
    logger.info(f"Total: {elapsed * 1e3:.1f}ms end-to-end for {n_done} frame(s) "
                f"({n_done / max(elapsed, 1e-6):.1f} fps processing)")
    logger.info(f"Video -> {out_path}")
    return n_done


def _label_lines(dets, task, w_img, h_img):
    """推理结果 -> YOLO txt 行：分割写归一化多边形，检测写归一化框

    分割口径与官方 `save_txt` 一致：每条掩码取外轮廓（多段合并），少于 3 点的实例跳过
    （不退回框——官方同样不写该行）；坐标归一化到原图，`%g` 精度。
    """
    if task == "segment" and dets.masks is not None:
        lines = []
        for cid, seg in zip(dets.class_ids, masks_to_segments(dets.masks)):
            if len(seg) < 3:
                continue
            coords = (seg / np.array([w_img, h_img], np.float32)).reshape(-1)
            lines.append(f"{int(cid)} " + " ".join(f"{v:g}" for v in coords))
        return lines
    lines = []
    for box, cid in zip(dets.boxes, dets.class_ids):
        x1, y1, x2, y2 = box
        xc, yc = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
        w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
        lines.append(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def _summary(image_path, image, dets, pipeline_ms, idx, total, names):
    """ultralytics predict 风格摘要：image i/n <path>: WxH <类别统计>, <管线耗时>ms"""
    s = f"image {idx}/{total} {image_path}: {image.shape[1]}x{image.shape[0]} "
    counts = Counter(int(c) for c in dets.class_ids)
    if counts:
        s += ", ".join(f"{n} {names.get(cid, cid)}{'s' * (n > 1)}" for cid, n in sorted(counts.items())) + ", "
    return s + f"{pipeline_ms:.1f}ms"


def main():
    parser = argparse.ArgumentParser(description="image / directory / video inference (YOLO26 / YOLOv3-tiny)")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights path")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="input image path or directory")
    src.add_argument("--video", help="input video file, stream URL (rtsp://...), or camera index (0)")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                        help="video only: stop after N frames (0 = until EOF; use it for live streams)")
    parser.add_argument("--show", action="store_true",
                        help="video only: display frames in a window (needs a GUI; q quits)")
    parser.add_argument("--engine", choices=["pt", "onnx", "trt"], default="pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nms", action="store_true",
                        help="use o2m+NMS path (yolo26 family + yoloe-26 text; default E2E)")
    parser.add_argument("--agnostic-nms", dest="agnostic_nms", action="store_true",
                        help="class-agnostic NMS (only with --nms; default: per-class suppression)")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument(
        "--output", default=None,
        help="output path: image mode — a directory, or an explicit file path for a single image "
             "(the filename is honored); video mode — a video file (default runs/predict/predictN/)",
    )
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    parser.add_argument("--scale", default=None,
                        help="model scale (yolo26/yoloe-26: n/s/m/l/x; default: inferred from the weights filename)")
    parser.add_argument("--data", default=None,
                        help="dataset descriptor (name or .yaml): sets nc and label names for models "
                             "trained on a non-COCO dataset (default: COCO, nc from the model yaml)")
    parser.add_argument("--classes", default=None,
                        help="yoloe-26 text mode only: comma-separated class names / text prompts "
                             "(pt: encoded with the MobileCLIP text encoder; onnx/trt: must match "
                             "the class-locked metadata; e.g. \"person, bicycle\")")
    parser.add_argument("--text-encoder", dest="text_encoder", default=None,
                        help="MobileCLIP2-B text encoder .safetensors for YOLOE text prompts "
                             "(default: mobileclip2_b.safetensors next to the weights)")
    parser.add_argument("--visual-prompt", dest="visual_prompt", default=None,
                        help="yoloe-26 text mode only: JSON with a refer image and bbox/mask "
                             "exemplars (pt engine; combine with --classes for mixed prompts)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="model input size (default: weights metadata, else 640)")
    args = parser.parse_args()
    video_mode = args.video is not None
    source = args.video if video_mode else args.image
    if args.max_frames and not video_mode:
        parser.error("--max-frames is only supported with --video")
    if args.show and not video_mode:
        parser.error("--show is only supported with --video")

    meta = load_meta(args.weights)  # 可选：训练保存的权重自带 {arch, scale, nc, imgsz, names, anchors}
    names, nc = load_names(), None  # 推理无数据集上下文：默认按 COCO 类名/80 类构建
    if args.data:
        try:
            spec = load_dataset(args.data)
        except (ValueError, FileNotFoundError) as e:
            parser.error(str(e))
        names, nc = spec.names, spec.nc
    elif meta.get("names"):  # 权重 metadata 兜底：自定义数据集训练产物免 --data
        names = {i: n for i, n in enumerate(meta["names"])}
        nc = meta.get("nc", len(names))
    elif meta.get("nc"):
        nc = meta["nc"]

    # 输入清单与 run 目录先确定：日志挂到 run 目录上（engine 构建失败的报错也才进得了 run.log）
    cap, video_info, images, multi = None, None, None, False
    if video_mode:
        cap, video_info = _open_video(source)  # 打不开在引擎构建之前拦下
    else:
        images = _collect_images(source)
        if not images:
            raise FileNotFoundError(f"no images found: {source}")
        multi = len(images) > 1 or Path(source).is_dir()

    # 架构/档位从权重文件名推（yolo26s -> yolo26/s；yolov3-tiny -> v3）——提前解析，参数预览要用
    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch_has_scale(arch) and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    # YOLOE prompt mode：text 支持 E2E + o2m/NMS；pf 仅 E2E（raw 列数随提案过滤变化）
    prompt_mode = (prompt_mode_from_weights(args.weights, meta)
                   if arch in YOLOE26_FAMILY else None)
    if args.nms:
        if arch in YOLOE26_FAMILY and prompt_mode != "text":
            parser.error("--nms is only supported for yoloe-26 text mode (prompt-free is E2E only)")
        if arch not in YOLO26_FAMILY and arch not in YOLOE26_FAMILY:
            parser.error(f"--nms is only supported for yolo26 / yoloe-26 text (got model={arch!r})")
    if args.agnostic_nms and not args.nms:
        parser.error("--agnostic-nms only affects the NMS path — pass --nms as well")
    if args.visual_prompt and prompt_mode != "text":
        parser.error("--visual-prompt requires yoloe-26 text-mode weights")
    visual_prompt_data = None
    text_prompt_names = None
    if meta.get("fused") and (args.classes or args.visual_prompt):
        parser.error("this checkpoint is class-locked (PE fused) — run infer without "
                     "--classes/--visual-prompt; its prompt names travel in metadata")
    if arch in YOLOE26_FAMILY:
        pm = prompt_mode
        if pm == "text":
            locked_names = meta.get("names")
            text_prompt_names = ([c.strip() for c in args.classes.split(",")]
                                 if args.classes else None)
            finetune_names = (list(locked_names)
                              if (meta.get("full_finetune") and locked_names) else None)
            if meta.get("fused"):
                if not locked_names:
                    parser.error("class-locked (PE fused) checkpoint has no 'names' metadata — "
                                 "re-save it with the training metadata")
                names = {i: n for i, n in enumerate(locked_names)}
            elif args.visual_prompt:
                if args.engine != "pt":
                    parser.error("--visual-prompt requires --engine pt (visual prompts run through SAVPE)")
                from utils.visual_prompt import load_visual_prompt

                try:
                    refer_path, vis_names, visual_prompts = load_visual_prompt(args.visual_prompt)
                except ValueError as e:
                    parser.error(str(e))
                if text_prompt_names:
                    dup = sorted(set(text_prompt_names) & set(vis_names))
                    if dup:
                        parser.error(f"text and visual prompts share class names: {dup[:5]} — "
                                     "rename one side or use a single prompt mode")
                ref = Path(refer_path)
                if not ref.is_absolute():  # 相对路径按 JSON 所在目录解析
                    ref = Path(args.visual_prompt).resolve().parent / ref
                refer_img = cv2.imread(str(ref))
                if refer_img is None:
                    parser.error(f"cannot read visual prompt refer image: {ref}")
                visual_prompt_data = (refer_img, vis_names, visual_prompts)
                combined = (text_prompt_names or []) + vis_names
                names = {i: n for i, n in enumerate(combined)}
            elif args.engine == "pt":
                # pt：类名来自 --classes（经 MobileCLIP 编码），类别数随 prompt 数变；
                # 全量微调产物自带训练时的 names，未显式给 --classes 时直接复用
                if not text_prompt_names:
                    if finetune_names:
                        text_prompt_names = finetune_names
                    else:
                        parser.error("yoloe-26 text-mode weights require --classes or --visual-prompt "
                                     '(e.g. --classes "person, bicycle")')
                names = {i: n for i, n in enumerate(text_prompt_names)}
            else:
                # onnx/trt：导出时已 fuse 成类锁定图，类别名随部署 metadata 走
                if not locked_names:
                    parser.error(f"{arch} class-locked {args.engine} export has no 'names' metadata — "
                                 "re-export with scripts/export.py --classes ...")
                if text_prompt_names:
                    if text_prompt_names != list(locked_names):
                        parser.error("--classes does not match the class-locked export: "
                                     f"requested {text_prompt_names}, locked {list(locked_names)}")
                names = {i: n for i, n in enumerate(locked_names)}
            if args.data:
                source = ("visual/mixed prompts" if args.visual_prompt
                          else ("the class-locked checkpoint metadata" if meta.get("fused")
                                else ("the fine-tuned checkpoint metadata"
                                      if (args.engine == "pt" and finetune_names and not args.classes)
                                      else ("--classes" if args.engine == "pt"
                                            else "the class-locked export metadata"))))
                logger.warning(f"--data is ignored for yoloe-26 text mode: class names come from {source}")
            nc = None
        else:  # pf：词汇即权重，名字必须来自烘焙词汇，--data/COCO 名不适用
            vocab_names = meta.get("names")
            if not vocab_names:
                parser.error(f"{arch} needs the vocabulary names in the weights metadata "
                             "(convert with scripts/convert_weights.py, or re-export with metadata)")
            if args.data:
                logger.warning("--data is ignored for yoloe-26: class names come from the baked vocabulary")
            names = {i: n for i, n in enumerate(vocab_names)}
            nc = None  # 头结构用 yaml nc（80），不随数据集走

    # 输入尺寸：CLI > 权重 metadata > 640（预览行用；engine 内部同一规则，构建后以 engine.imgsz 为准）
    imgsz = resolve_imgsz(args.weights, args.imgsz)

    # 结果保存：图片默认 runs/predict/predictN/（--output 目录或单图显式文件）；视频默认 <run_dir>/<源名>.mp4
    video_out = None
    image_out = None
    if args.output:
        out = Path(args.output).resolve()
        if video_mode:
            if out.is_dir():
                parser.error("--output in video mode must be a video file path, not a directory")
            if out.suffix.lower() not in _VIDEO_EXTS:
                out = out.with_suffix(".mp4")
            video_out = out
            run_dir = out.parent
        else:
            try:
                run_dir, image_out = _resolve_image_output(args.output, multi)
            except ValueError as exc:
                parser.error(str(exc))
    else:
        run_dir = increment_path(ROOT / "runs" / "predict" / "predict")
    run_dir.mkdir(parents=True, exist_ok=True)
    if video_mode and video_out is None:
        video_out = run_dir / f"{_video_stem(source)}.mp4"
    attach_file_log(run_dir / "run.log")
    extra = {"classes": args.classes} if args.classes else {}
    if video_mode:
        log_params(logger, __file__, weights=args.weights, video=args.video, model=arch,
                   engine=args.engine, conf=args.conf, imgsz=imgsz,
                   max_frames=args.max_frames, **extra)
    else:
        log_params(logger, __file__, weights=args.weights, image=args.image, model=arch,
                   engine=args.engine, conf=args.conf, imgsz=imgsz, **extra)

    # ---- 头部（与训练/评估同一套五段排版：环境 -> 模型 -> 输入/参数 -> 细节）----
    # ① 环境：engine 构建之前（onnx 后端固定 CPU、trt 后端固定 CUDA，见各自 Engine）
    device = (resolve_device(args.device) if args.engine == "pt"
              else ("cuda" if args.engine == "trt" else "cpu"))
    logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {device_label(device)}"))

    # ② 模型（arch/scale 已在前段解析）
    end2end = not args.nms
    graph_end2end = meta.get("end2end")
    if args.engine != "pt" and graph_end2end is not None and graph_end2end != end2end:
        parser.error(f"{args.weights} was exported as {'E2E' if graph_end2end else 'o2m+NMS'} "
                     f"but {'--nms' if args.nms else 'E2E (default)'} was requested — "
                     f"{'remove --nms' if graph_end2end else 'pass --nms'}")
    engine_cls = {"pt": PtEngine, "onnx": OnnxEngine, "trt": TRTEngine}[args.engine]
    kwargs = {"end2end": end2end, "scale": scale, "model": arch, "imgsz": args.imgsz,
              "agnostic_nms": args.agnostic_nms}
    if args.engine == "pt":
        kwargs["device"] = args.device
        kwargs["nc"] = nc  # onnx 图内置类别数，只有 pt 引擎需要重建头
        kwargs["text_encoder"] = args.text_encoder
    else:
        kwargs["text_mode"] = prompt_mode == "text"  # 类锁定 YOLOE text：o2m 输出不做 decode_raw
    engine = engine_cls(args.weights, **kwargs)
    imgsz = engine.imgsz  # onnx 以图内固定 shape 为准（pt = 上面的解析结果）
    if arch in YOLOE26_FAMILY and prompt_mode == "text":
        if getattr(engine, "fused", False):  # PE 融合产物：类别已锁定，不再接受新 prompt
            if args.classes or args.visual_prompt:
                parser.error("this checkpoint is class-locked (PE fused) — run infer without "
                             "--classes/--visual-prompt; its prompt names travel in metadata")
            mode = ("E2E " if end2end else "o2m+NMS ") + "fused text prompts (class-locked)"
        elif args.visual_prompt and text_prompt_names:
            engine.set_mixed_prompts(text_prompt_names, *visual_prompt_data)
            mode = "E2E (text + visual prompts)" if end2end else "o2m+NMS (text + visual prompts)"
        elif args.visual_prompt:
            engine.set_visual_prompts(*visual_prompt_data)
            mode = "E2E (visual prompts)" if end2end else "o2m+NMS (visual prompts)"
        else:
            locked = args.engine != "pt"
            if not locked:
                engine.set_classes([names[i] for i in sorted(names)])
            kind = "class-locked prompts" if locked else "text prompts"
            mode = ("E2E (" if end2end else "o2m+NMS (") + kind + ")"
    elif arch in YOLOE26_FAMILY:
        mode = "E2E (prompt-free)"
    elif arch in YOLO26_FAMILY:
        mode = "E2E (NMS-free)" if end2end else "o2m+NMS"
    else:
        mode = "decode+NMS"
    logger.info(bold(f"{arch_display_name(arch, scale)} · {engine.summary_line} · "
                     f"{mode} · engine {args.engine}"))

    # ③ 输入清单 + 本次任务参数
    if video_mode:
        dims = (f"{video_info['width']}x{video_info['height']}"
                if video_info["width"] > 0 and video_info["height"] > 0 else "unknown size")
        frames = f"{video_info['frames']} frames" if video_info["frames"] > 0 else "live stream"
        limit = f" · max {args.max_frames} frames" if args.max_frames else ""
        logger.info(f"infer: {source} · video {dims} @ {video_info['fps']:.1f} fps · {frames}{limit} · "
                    f"conf {args.conf} · imgsz {imgsz}")
        _run_video(engine, names, cap, video_info, args, imgsz, video_out)
        return
    logger.info(f"infer: {source} · {len(images)} image{'s' if len(images) > 1 else ''} · "
                f"conf {args.conf} · imgsz {imgsz}")

    t_start = time.perf_counter()  # 端到端计时（含图像读/写 I/O）
    stage_sums = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    n_done = 0
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"skip unreadable image: {path}")
            continue
        n_done += 1
        dets, tseg = engine.predict_timed(image, conf_thres=args.conf)
        for k in stage_sums:
            stage_sums[k] += tseg[k]

        out_path = image_out if image_out is not None else run_dir / path.name
        draw_detections(image, dets, names, save_path=str(out_path))

        # YOLO 格式标签：labels/<输出图同名>.txt；分割写多边形行，检测写 cls xc yc w h（均归一化）
        h_img, w_img = image.shape[:2]
        with open(labels_dir / (out_path.stem + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in _label_lines(dets, engine.task, w_img, h_img))

        pipeline_ms = sum(tseg.values())
        logger.info(_summary(path, image, dets, pipeline_ms, idx, len(images), names))

    if n_done == 0:
        logger.warning("no valid images processed")
        return
    elapsed_ms = (time.perf_counter() - t_start) * 1e3

    # ---- 汇总（单图与多图同格式）----
    avg = {k: v / n_done for k, v in stage_sums.items()}
    logger.info(
        f"Speed: {avg['preprocess']:.1f}ms preprocess, {avg['inference']:.1f}ms inference, "
        f"{avg['postprocess']:.1f}ms postprocess per image at shape (1, 3, {imgsz}, {imgsz})"
    )
    logger.info(f"Total: {elapsed_ms:.1f}ms end-to-end for {n_done} image(s) (incl. image read + save)")
    logger.info(f"Visualization -> {run_dir}")
    logger.info(f"Labels        -> {labels_dir}")


if __name__ == "__main__":
    main()

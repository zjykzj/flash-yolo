"""视频推理：源/写视频/逐帧循环（伪引擎，无模型）+ 端到端 CLI（FLASH_YOLO_SMOKE）

纯单测不依赖权重：伪引擎返回固定 Detections，验证 --max-frames、输出帧数/尺寸/帧率。
端到端用例（门控）训练一个迷你模型后跑 `scripts/infer.py --video`，验证参数解析、
引擎构建、画框与 VideoWriter 全链路。
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_infer():
    """按路径加载 scripts/infer.py（不是包，importlib 避免 sys.path 歧义）"""
    spec = importlib.util.spec_from_file_location("infer_script", ROOT / "scripts" / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_video(path, n=6, size=(64, 48), fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), f"cannot write test video: {path}"
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    writer.release()
    return path


def _video_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, size, fps


class _FakeEngine:
    """只实现 _run_video 用到的 predict_timed：固定返回一个框"""

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=None):
        from utils.engines.base import Detections

        h, w = image_bgr.shape[:2]
        dets = Detections(boxes=np.array([[1, 1, w - 2, h - 2]], np.float32),
                          scores=np.array([0.9], np.float32), class_ids=np.array([0], np.int64))
        return dets, {"preprocess": 0.5, "inference": 1.5, "postprocess": 0.2}


def test_open_video_metadata(tmp_path):
    infer = _load_infer()
    src = _make_video(tmp_path / "in.avi", n=5, size=(80, 60), fps=12.0)
    cap, info = infer._open_video(src)
    try:
        assert (info["width"], info["height"]) == (80, 60), info
        assert info["frames"] == 5 and abs(info["fps"] - 12.0) < 1e-3, info
    finally:
        cap.release()


def test_open_video_missing_source(tmp_path):
    infer = _load_infer()
    with pytest.raises(FileNotFoundError):
        infer._open_video(tmp_path / "missing.mp4")


def test_run_video_max_frames_and_output(tmp_path):
    """逐帧循环：--max-frames 生效、输出帧数/尺寸/帧率与源一致（伪引擎，无训练）"""
    infer = _load_infer()
    src = _make_video(tmp_path / "in.avi", n=6, size=(64, 48), fps=10.0)
    cap, info = infer._open_video(src)
    out = tmp_path / "out.avi"
    args = SimpleNamespace(conf=0.25, max_frames=3, show=False)
    n = infer._run_video(_FakeEngine(), {0: "cls0"}, cap, info, args, 64, out)
    assert n == 3
    frames, size, fps = _video_frames(out)
    assert frames == 3 and size == (64, 48) and abs(fps - 10.0) < 1e-3, (frames, size, fps)


def test_run_video_reads_all_without_limit(tmp_path):
    infer = _load_infer()
    src = _make_video(tmp_path / "in.avi", n=4, size=(32, 32), fps=5.0)
    cap, info = infer._open_video(src)
    out = tmp_path / "out.avi"
    args = SimpleNamespace(conf=0.25, max_frames=0, show=False)
    assert infer._run_video(_FakeEngine(), {0: "cls0"}, cap, info, args, 32, out) == 4
    assert _video_frames(out)[0] == 4


@pytest.mark.skipif(os.environ.get("FLASH_YOLO_SMOKE") != "1",
                    reason="FLASH_YOLO_SMOKE=1 才执行视频推理 CLI 冒烟")
def test_infer_video_cli(tmp_path):
    """训练 1 轮迷你模型 -> scripts/infer.py --video：mp4 输出 2 帧（端到端 CLI）"""
    import torch

    from config.train_config import TrainConfig
    from test_train_smoke import _make_mini_coco
    from train.trainer import Trainer

    data = _make_mini_coco(tmp_path, n_img=8, size=160)
    cfg = TrainConfig(
        data=str(data), epochs=1, batch=4, nbs=4, imgsz=160, workers=0, val_epochs=0,
        amp=False, mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
        aug_samples=0, plots=False, save_period=0, diag_interval=0,
    )
    torch.manual_seed(0)
    run = tmp_path / "train"
    Trainer(cfg, device="cpu", run_dir=run).train()

    src = _make_video(tmp_path / "in.avi", n=4, size=(64, 48), fps=10.0)
    out = tmp_path / "out.mp4"
    subprocess.run(
        [sys.executable, "scripts/infer.py", "--weights", str(run / "weights" / "last.safetensors"),
         "--video", str(src), "--output", str(out), "--imgsz", "160", "--max-frames", "2"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert out.exists(), "video output missing"
    frames, size, _ = _video_frames(out)
    assert frames == 2 and size == (64, 48), (frames, size)

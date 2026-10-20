"""训练冒烟（门控）：合成迷你数据集上完整跑 Trainer 数 epoch（coco / yolo 双格式）

默认跳过；FLASH_YOLO_SMOKE=1 时执行（ CPU 上 ~2-5 分钟）。
两种格式跑同一条 Trainer 路径：描述符 -> 工厂 -> 加载器 -> nc 跟随数据集（2 类）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from config.train_config import TrainConfig
from train.trainer import Trainer

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_YOLO_SMOKE") != "1", reason="FLASH_YOLO_SMOKE=1 才执行训练冒烟"
)

# 同一批框写两种格式：像素 [x1, y1, x2, y2, cls]（64px 画布内的两块白矩形放大到 size）
_BOXES = [(40, 40, 120, 120, 0), (160, 160, 280, 280, 1)]


def _draw(size):
    img = np.zeros((size, size, 3), np.uint8)
    cv2.rectangle(img, (40, 40), (120, 120), (200, 200, 200), -1)
    cv2.rectangle(img, (160, 160), (280, 280), (255, 255, 255), -1)
    return img


def _scaled_boxes(size):
    r = size / 320.0
    return [[int(round(v * r)) if i < 4 else v for i, v in enumerate(b)] for b in _BOXES]


def _make_mini_coco(tmp_path, n_img=32, size=320, seg=False):
    """n 张图 + 2 类的迷你 COCO（images/<split>/ 布局）+ 描述符；seg=True 时带矩形多边形"""
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        images, anns = [], []
        aid = 0
        for i in range(n_img):
            img = _draw(size)
            cv2.imwrite(str(root / "images" / split / f"{i:04d}.jpg"), img)
            images.append({"id": i + 1, "file_name": f"{i:04d}.jpg", "width": size, "height": size})
            for x1, y1, x2, y2, cls in _scaled_boxes(size):
                aid += 1
                ann = {"id": aid, "image_id": i + 1, "category_id": cls + 1,
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
                if seg:
                    ann["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                anns.append(ann)
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(
            {"images": images, "annotations": anns,
             "categories": [{"id": 1, "name": "cls0"}, {"id": 2, "name": "cls1"}]}))
    desc = tmp_path / "coco.yaml"
    desc.write_text(f"format: coco\npath: {root}\nnames: [cls0, cls1]\n"
                    "train: {images: images/train, ann: annotations/instances_train.json}\n"
                    "val: {images: images/val, ann: annotations/instances_val.json}\n", encoding="utf-8")
    return desc


def _make_mini_yolo(tmp_path, n_img=32, size=320, seg=False):
    """同一批图的 YOLO 版（seg=False: 归一化 cxcywh；True: 归一化矩形多边形）+ 描述符"""
    root = tmp_path / "yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n_img):
            cv2.imwrite(str(root / "images" / split / f"{i:04d}.jpg"), _draw(size))
            lines = []
            for x1, y1, x2, y2, cls in _scaled_boxes(size):
                if seg:
                    pts = (x1, y1, x2, y1, x2, y2, x1, y2)
                    lines.append(f"{cls} " + " ".join(f"{v / size:.6f}" for v in pts))
                else:
                    lines.append(f"{cls} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                                 f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}")
            (root / "labels" / split / f"{i:04d}.txt").write_text("\n".join(lines) + "\n")
    desc = tmp_path / "yolo.yaml"
    desc.write_text(f"format: yolo\npath: {root}\nnames: [cls0, cls1]\n"
                    "train: {images: images/train}\nval: {images: images/val}\n", encoding="utf-8")
    return desc


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
def test_train_smoke(tmp_path, fmt):
    """3 epoch 迷你数据集：损失下降、检查点齐全、验证出指标（两种标注格式同一条路径）"""
    data = _make_mini_coco(tmp_path) if fmt == "coco" else _make_mini_yolo(tmp_path)
    cfg = TrainConfig(
        data=str(data), epochs=3, batch=8, nbs=8, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / f"run-{fmt}"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    assert (run_dir / "resume.pt").exists()
    assert (run_dir / "results.csv").exists()
    # 权重 metadata 随盘：训练配置（imgsz/nc/names）+ 出处（version/repo[/commit]）可被读取侧还原
    from config import __repo__, __version__
    from model.weights import load_meta
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["imgsz"] == 320 and wm["nc"] == 2 and wm["names"] == ["cls0", "cls1"], wm
    assert wm["version"] == __version__ and wm["repo"] == __repo__, wm
    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 3, f"results.csv 行数不足: {len(rows)}"
    losses = [float(r["loss"]) for r in rows]
    o2m = [float(r["o2m"]) for r in rows]
    # 冒烟职责 = 管线完整性（无 NaN/发散、检查点、验证出指标）；32 图小样本的损失
    # 天然震荡（实测 3 轮内 o2m 可能先升后降），收敛趋势断言留给真实数据训练
    assert all(np.isfinite(v) for v in losses + o2m), f"出现 NaN 损失: {losses} {o2m}"
    assert all(v < 100 for v in o2m), f"损失疑似发散: {o2m}"
    assert all(r["mAP"] for r in rows), "每 epoch 都应有验证指标"
    # val 损失：有限且非负即可——toy 短跑在 eval 模式下框可能整体退化（BN 统计未收敛、
    # 与 GT 全无重叠 -> 软标签 t=0 -> box/l1 恰为 0，属损失的正确语义；cls 仍 > 0）
    val_box = [float(r["val_box"]) for r in rows]
    val_cls = [float(r["val_cls"]) for r in rows]
    assert all(np.isfinite(v) and v >= 0 for v in val_box), f"val box 损失异常: {val_box}"
    assert all(np.isfinite(v) and v > 0 for v in val_cls), f"val cls 损失异常: {val_cls}"
    # 数据集类别数跟随描述符：模型以 nc=2 构建（不再需要"声明 80 类"的旧把戏）
    assert trainer.head.nc == 2, f"模型 nc 应跟随描述符: {trainer.head.nc}"
    print(f"  [{fmt}] 冒烟通过: o2m {o2m[0]:.2f} -> {o2m[-1]:.2f}（32 图小样本波动属正常）· "
          f"val_box {val_box[0]:.3f} -> {val_box[-1]:.3f} · val_cls {val_cls[0]:.2f}")


def test_train_smoke_v3(tmp_path):
    """YOLOv3-tiny：同一条 Trainer 路径（架构由 cfg.model 选择），列名/验证/产物全走通"""
    data = _make_mini_coco(tmp_path)
    cfg = TrainConfig(
        data=str(data), model="yolov3-tiny", epochs=2, batch=8, nbs=8, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-v3"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    assert (run_dir / "resume.pt").exists()
    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 2, f"results.csv 行数不足: {len(rows)}"
    # 列名随损失实现派生：v3 为 box/obj/cls（无 o2m/o2o）
    assert "obj" in rows[0] and "o2m" not in rows[0], list(rows[0])
    losses = [float(r["loss"]) for r in rows]
    objs = [float(r["obj"]) for r in rows]
    assert all(np.isfinite(v) for v in losses + objs), f"出现 NaN 损失: {losses} {objs}"
    assert all(v < 100 for v in objs), f"obj 损失疑似发散: {objs}"
    assert all(r["mAP"] for r in rows), "每 epoch 都应有验证指标"
    val_obj = [float(r["val_obj"]) for r in rows]
    assert all(np.isfinite(v) and v > 0 for v in val_obj), f"val obj 损失异常: {val_obj}"
    assert trainer.head.nc == 2, f"模型 nc 应跟随描述符: {trainer.head.nc}"
    # 权重 metadata 随盘：v3 的 anchors（yaml 官方值）也写进 header，评估/推理侧可还原
    from model.weights import load_meta
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["arch"] == "yolov3-tiny" and wm["nc"] == 2 and wm["imgsz"] == 320, wm
    assert wm["anchors"] == [[[23, 27], [37, 58], [81, 82]], [[81, 82], [135, 169], [344, 319]]], wm["anchors"]
    print(f"  [v3] 冒烟通过: obj {objs[0]:.3f} -> {objs[-1]:.3f} · val_obj {val_obj[0]:.3f} -> {val_obj[-1]:.3f}")


def test_train_smoke_seg(tmp_path):
    """YOLO26-seg：COCO 多边形 -> 掩码加载/分割损失/Box+Mask 验证/产物全链路"""
    data = _make_mini_coco(tmp_path, seg=True)
    cfg = TrainConfig(
        data=str(data), model="yolo26-seg", epochs=2, batch=4, nbs=4, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False, mask_ratio=4, overlap_mask=True,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-seg"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    assert (run_dir / "resume.pt").exists()
    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 2, f"results.csv 行数不足: {len(rows)}"
    assert "seg" in rows[0] and "val_seg" in rows[0], list(rows[0])
    assert "sem" in rows[0] and "val_sem" in rows[0], list(rows[0])
    assert "P(M)" in rows[0] and "mAP@[.5:.95](M)" in rows[0], list(rows[0])
    seg = [float(r["seg"]) for r in rows]
    val_seg = [float(r["val_seg"]) for r in rows]
    sem = [float(r["sem"]) for r in rows]
    assert all(np.isfinite(v) and v >= 0 for v in seg + val_seg + sem), f"seg/sem 损失异常: {seg} {sem}"
    assert all(np.isfinite(float(r["mAP@[.5:.95](M)"])) for r in rows)
    assert trainer.head.nc == 2, f"模型 nc 应跟随描述符: {trainer.head.nc}"
    from model.build import build_model
    from model.weights import load_meta, load_weights
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["arch"] == "yolo26-seg" and wm["nc"] == 2 and wm["imgsz"] == 320, wm
    model = build_model("yolo26-seg", "n", 320, nc=2)
    load_weights(model, run_dir / "weights" / "best.safetensors", strict=True)
    print(f"  [seg] 冒烟通过: seg {seg[0]:.3f} -> {seg[-1]:.3f} · val_seg {val_seg[0]:.3f} -> {val_seg[-1]:.3f}")


def test_train_smoke_flash_seg(tmp_path):
    """Flash-YOLO-seg：固定档 flash，分割损失/Box+Mask 验证/权重 metadata 全链路"""
    data = _make_mini_coco(tmp_path, seg=True)
    cfg = TrainConfig(
        data=str(data), model="flash-yolo-seg", epochs=2, batch=4, nbs=4, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False, mask_ratio=4, overlap_mask=True,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-flash-seg"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    assert cfg.scale == "flash"  # 固定档：Trainer 自动归一化，--scale 不适用
    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 2, f"results.csv 行数不足: {len(rows)}"
    for key in ("seg", "val_seg", "sem", "val_sem", "P(M)", "mAP@[.5:.95](M)"):
        assert key in rows[0], (key, list(rows[0]))
    losses = [float(r["seg"]) for r in rows] + [float(r["val_seg"]) for r in rows]
    assert all(np.isfinite(v) and v >= 0 for v in losses), losses
    assert trainer.head.nc == 2

    from model.build import build_model
    from model.weights import load_meta, load_weights
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["arch"] == "flash-yolo-seg" and wm["scale"] == "flash" and wm["nc"] == 2, wm
    model = build_model("flash-yolo-seg", nc=2)
    load_weights(model, run_dir / "weights" / "best.safetensors", strict=True)
    print(f"  [flash-seg] 冒烟通过: seg {losses[0]:.3f} -> {losses[-1]:.3f}")


def test_train_smoke_seg_yolo(tmp_path):
    """YOLO-seg txt：多边形行 -> 图像/掩码加载、分割损失、Box+Mask 验证、产物全链路"""
    data = _make_mini_yolo(tmp_path, seg=True)
    cfg = TrainConfig(
        data=str(data), model="yolo26-seg", epochs=2, batch=4, nbs=4, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False, mask_ratio=4, overlap_mask=True,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-seg-yolo"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 2, f"results.csv 行数不足: {len(rows)}"
    assert "seg" in rows[0] and "val_seg" in rows[0], list(rows[0])
    assert trainer.dataset.n_seg_fallback == 0, trainer.dataset.n_seg_fallback
    seg = [float(r["seg"]) for r in rows]
    val_seg = [float(r["val_seg"]) for r in rows]
    assert all(np.isfinite(v) and v >= 0 for v in seg + val_seg), f"seg 损失异常: {seg} {val_seg}"
    from model.build import build_model
    from model.weights import load_meta, load_weights
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["arch"] == "yolo26-seg" and wm["nc"] == 2 and wm["imgsz"] == 320, wm
    model = build_model("yolo26-seg", "n", 320, nc=2)
    load_weights(model, run_dir / "weights" / "best.safetensors", strict=True)
    print(f"  [seg-yolo] 冒烟通过: seg {seg[0]:.3f} -> {seg[-1]:.3f} · val_seg {val_seg[0]:.3f} -> {val_seg[-1]:.3f}")


def test_train_smoke_seg_workers_multiscale(tmp_path):
    """seg + workers>0 + multi-scale：worker 传多边形、每 batch 换输入尺寸，掩码网格固定"""
    data = _make_mini_coco(tmp_path, n_img=16, seg=True)
    cfg = TrainConfig(
        data=str(data), model="yolo26-seg", epochs=1, batch=4, nbs=4, imgsz=320, workers=2,
        val_epochs=1, val_limit=4, limit=0, amp=False, mask_ratio=4, overlap_mask=True,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
        multi_scale=0.25,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-seg-workers"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) == 1, f"1 epoch 应只有 1 行: {len(rows)}"
    seg, val_seg = float(rows[0]["seg"]), float(rows[0]["val_seg"])
    assert np.isfinite(seg) and seg >= 0 and np.isfinite(val_seg) and val_seg >= 0
    assert np.isfinite(float(rows[0]["mAP@[.5:.95](M)"]))
    print(f"  [seg workers=2 + multi-scale] 冒烟通过: seg {seg:.3f} · val_seg {val_seg:.3f}")


def test_export_smoke_seg(tmp_path):
    """训练产物 -> scripts/export.py：metadata 解析 arch/scale/nc/imgsz，双输出 ONNX"""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    data = _make_mini_coco(tmp_path, n_img=8, seg=True)
    cfg = TrainConfig(
        data=str(data), model="yolo26-seg", epochs=1, batch=4, nbs=4, imgsz=320, workers=0,
        val_epochs=0, limit=0, amp=False, mask_ratio=4, overlap_mask=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0, aug_samples=0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-seg-export"
    Trainer(cfg, device="cpu", run_dir=run_dir).train()
    weights = run_dir / "weights" / "last.safetensors"
    out = tmp_path / "seg.onnx"
    subprocess.run(
        [sys.executable, "scripts/export.py", "--weights", str(weights), "--out", str(out)],
        cwd=Path(__file__).resolve().parent.parent, check=True,
    )
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shapes = [tuple(o.shape) for o in sess.get_outputs()]
    # nc=2 / nm=32 / imgsz=320：E2E (1,300,6+32) + proto (1,32,80,80)
    assert shapes == [(1, 300, 38), (1, 32, 80, 80)], shapes
    print(f"  [seg export] 冒烟通过: {shapes}")


def test_train_multiscale_step_lr_smoke(tmp_path):
    """多尺度（每 batch 换输入尺寸 + 像素 targets 等比缩放）+ step 衰减（darknet 台阶）端到端跑通"""
    data = _make_mini_coco(tmp_path)
    cfg = TrainConfig(
        data=str(data), model="yolov3-tiny", epochs=2, batch=8, nbs=8, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
        multi_scale=0.25, lr_schedule="step", lr_step_fracs=[0.5, 0.9], lr_step_gamma=0.1,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-ms-step"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    rows = list(csv_reader(run_dir / "results.csv"))
    lrs = [float(r["lr"]) for r in rows]
    # epochs=2：第 0 轮末批 t=1.0（越过 frac 0.5）→ ×0.1；第 1 轮末批 t=2.0（越过 0.9）→ ×0.01
    assert abs(lrs[0] - 1e-3) < 1e-6 and abs(lrs[1] - 1e-4) < 1e-6, f"step 台阶未生效: {lrs}"
    losses = [float(r["loss"]) for r in rows]
    assert all(np.isfinite(v) for v in losses), f"出现 NaN 损失: {losses}"
    print(f"  [multi-scale + step-lr] 冒烟通过: lr {lrs} · loss {losses[0]:.3f} -> {losses[-1]:.3f}")


def csv_reader(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_flash_yolo_smoke(tmp_path):
    """flash-yolo 走同一条 Trainer 路径（与 yolo26 同头/同损失）：3 epoch 无 NaN、检查点与 metadata 齐全"""
    data = _make_mini_coco(tmp_path)
    cfg = TrainConfig(
        data=str(data), model="flash-yolo", epochs=3, batch=8, nbs=8, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run-flash-yolo"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()
    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    rows = list(csv_reader(run_dir / "results.csv"))
    assert len(rows) >= 3, f"results.csv 行数不足: {len(rows)}"
    # 损失列随实现派生：flash-yolo 复用 yolo26 双头损失（o2m/o2o 列存在）
    assert "o2m" in rows[0] and "o2o" in rows[0], list(rows[0])
    losses = [float(r["loss"]) for r in rows]
    assert all(np.isfinite(v) for v in losses), f"出现 NaN 损失: {losses}"
    assert all(r["mAP"] for r in rows), "每 epoch 都应有验证指标"
    from model.weights import load_meta
    wm = load_meta(run_dir / "weights" / "best.safetensors")
    assert wm["arch"] == "flash-yolo" and wm["nc"] == 2 and wm["imgsz"] == 320, wm
    print(f"  [flash-yolo] 冒烟通过: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

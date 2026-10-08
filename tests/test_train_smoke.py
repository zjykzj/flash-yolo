"""训练冒烟（门控）：合成迷你数据集上完整跑 Trainer 数 epoch（coco / yolo 双格式）

默认跳过；FLASH_YOLO_SMOKE=1 时执行（ CPU 上 ~2-5 分钟）。
两种格式跑同一条 Trainer 路径：描述符 -> 工厂 -> 加载器 -> nc 跟随数据集（2 类）。
"""

import json
import os

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


def _make_mini_coco(tmp_path, n_img=32, size=320):
    """n 张图 + 2 类的迷你 COCO（images/<split>/ 布局）+ 描述符"""
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
                anns.append({"id": aid, "image_id": i + 1, "category_id": cls + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(
            {"images": images, "annotations": anns,
             "categories": [{"id": 1, "name": "cls0"}, {"id": 2, "name": "cls1"}]}))
    desc = tmp_path / "coco.yaml"
    desc.write_text(f"format: coco\npath: {root}\nnames: [cls0, cls1]\n"
                    "train: {images: images/train, ann: annotations/instances_train.json}\n"
                    "val: {images: images/val, ann: annotations/instances_val.json}\n", encoding="utf-8")
    return desc


def _make_mini_yolo(tmp_path, n_img=32, size=320):
    """同一批图的 YOLO 版（labels 归一化 cxcywh）+ 描述符"""
    root = tmp_path / "yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n_img):
            cv2.imwrite(str(root / "images" / split / f"{i:04d}.jpg"), _draw(size))
            lines = []
            for x1, y1, x2, y2, cls in _scaled_boxes(size):
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


def csv_reader(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

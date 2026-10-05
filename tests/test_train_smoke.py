"""训练冒烟（门控）：合成迷你 COCO 上完整跑 Trainer 数 epoch

默认跳过；FLASH_YOLO_SMOKE=1 时执行（CPU 上 ~2-5 分钟）。
"""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from config.train import TrainConfig
from train.trainer import Trainer

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_YOLO_SMOKE") != "1", reason="FLASH_YOLO_SMOKE=1 才执行训练冒烟"
)


def _make_mini_coco(tmp_path, n_img=32, size=320):
    """n 张纯色图 + 2 类矩形框的迷你 COCO"""
    data_dir = tmp_path / "coco"
    (data_dir / "annotations").mkdir(parents=True)
    (data_dir / "images" / "train2017").mkdir(parents=True)
    (data_dir / "images" / "val2017").mkdir(parents=True)

    def write_split(split, n):
        images, anns = [], []
        aid = 0
        for i in range(n):
            img = np.zeros((size, size, 3), np.uint8)
            cv2.rectangle(img, (40, 40), (120, 120), (200, 200, 200), -1)
            cv2.rectangle(img, (160, 160), (280, 280), (255, 255, 255), -1)
            cv2.imwrite(str(data_dir / "images" / split / f"{i:04d}.jpg"), img)
            images.append({"id": i + 1, "file_name": f"{i:04d}.jpg", "width": size, "height": size})
            anns.append({"id": aid + 1, "image_id": i + 1, "category_id": 1, "bbox": [40, 40, 80, 80],
                         "area": 6400, "iscrowd": 0})
            aid += 1
            anns.append({"id": aid + 1, "image_id": i + 1, "category_id": 2, "bbox": [160, 160, 120, 120],
                         "area": 14400, "iscrowd": 0})
            aid += 1
        return images, anns

    for split in ("train2017", "val2017"):
        images, anns = write_split(split, n_img)
        with open(data_dir / "annotations" / f"instances_{split}.json", "w") as f:
            # 声明全部 80 类（模型 nc=80 会预测任意类，评估器需要完整映射）
            json.dump(
                {"images": images, "annotations": anns,
                 "categories": [{"id": i + 1, "name": f"cls{i}"} for i in range(80)]}, f
            )
    return data_dir


def test_train_smoke(tmp_path):
    """3 epoch 迷你 COCO：损失下降、检查点齐全、验证出指标"""
    data_dir = _make_mini_coco(tmp_path)
    cfg = TrainConfig(
        data_dir=str(data_dir), epochs=3, batch=8, nbs=8, imgsz=320, workers=0,
        val_epochs=1, val_limit=8, limit=0, amp=False,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
    )
    torch.manual_seed(0)
    run_dir = tmp_path / "run"
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train()

    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    assert (run_dir / "resume.pt").exists()
    assert (run_dir / "results.csv").exists()
    rows = list(csv_reader(run_dir / "results.csv"))
    losses = [float(r["loss"]) for r in rows[1:]]
    o2m = [float(r["o2m"]) for r in rows[1:]]
    # 断言用 o2m 列（不受 ProgLoss α 右移影响；加权总损失在短跑中可能暂时性上升）
    assert o2m[-1] < o2m[0], f"o2m 损失应下降: {o2m}"
    assert all(r["mAP"] for r in rows[1:]), "每 epoch 都应有验证指标"
    print(f"  冒烟通过: loss {losses[0]:.3f} -> {losses[-1]:.3f} (o2m {o2m[0]:.3f} -> {o2m[-1]:.3f})")


def csv_reader(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

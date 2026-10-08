"""compute_anchors 验收：k-means 聚类还原 / 覆盖判定口径 / 两种格式提取一致 / CLI 冒烟"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from config.datasets import load_dataset
from scripts.compute_anchors import anchor_stats, best_ratio, collect_norm_wh, kmeans_anchors

ROOT = Path(__file__).resolve().parent.parent

# 320 画布上的两种框：64×64 与 192×192（归一化 0.2 / 0.6，便于手算）
_BOXES = [(40, 40, 104, 104), (128, 128, 320, 320)]


def _draw(size=320):
    img = np.zeros((size, size, 3), np.uint8)
    for x1, y1, x2, y2 in _BOXES:
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
    return img


def _make_mini(tmp_path, fmt, n_img=8, size=320):
    """同批图写两种格式（与 test_train_smoke 同风格的小夹具）"""
    root = tmp_path / f"mini-{fmt}"
    (root / "images" / "train").mkdir(parents=True)
    for i in range(n_img):
        cv2.imwrite(str(root / "images" / "train" / f"{i:04d}.jpg"), _draw(size))
    if fmt == "coco":
        (root / "annotations").mkdir()
        images, anns = [], []
        aid = 0
        for i in range(n_img):
            images.append({"id": i + 1, "file_name": f"{i:04d}.jpg", "width": size, "height": size})
            for x1, y1, x2, y2 in _BOXES:
                aid += 1
                anns.append({"id": aid, "image_id": i + 1, "category_id": 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        (root / "annotations" / "train.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "cls0"}]}))
        desc = tmp_path / "mini-coco.yaml"
        desc.write_text(f"format: coco\npath: {root}\nnames: [cls0]\n"
                        "train: {images: images/train, ann: annotations/train.json}\n", encoding="utf-8")
    else:
        (root / "labels" / "train").mkdir(parents=True)
        for i in range(n_img):
            lines = [f"0 {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                     f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}" for x1, y1, x2, y2 in _BOXES]
            (root / "labels" / "train" / f"{i:04d}.txt").write_text("\n".join(lines) + "\n")
        desc = tmp_path / "mini-yolo.yaml"
        desc.write_text(f"format: yolo\npath: {root}\nnames: [cls0]\n"
                        "train: {images: images/train}\n", encoding="utf-8")
    return desc


def test_kmeans_recovers_clusters():
    """两类形状的合成点：k=6 时前 3 锚 ≈ 小簇、后 3 锚 ≈ 大簇，且可复现"""
    rng = np.random.default_rng(0)
    wh = np.concatenate([
        64 + rng.normal(0, 2, (200, 2)),
        192 + rng.normal(0, 4, (200, 2)),
    ]).astype(np.float32)
    anchors, score = kmeans_anchors(wh, 6, n_init=5, seed=0)
    assert np.allclose(anchors[:3], 64, atol=8), anchors
    assert np.allclose(anchors[3:], 192, atol=12), anchors
    assert score > 0.9, score
    again, _ = kmeans_anchors(wh, 6, n_init=5, seed=0)
    assert np.array_equal(anchors, again), "同 seed 应完全可复现"
    print(f"  k-means 还原正确（score {score:.3f}，锚 {anchors[:, 0].tolist()}）")


def test_coverage_criterion():
    """v5 口径判定：完全匹配 -> 覆盖 1.0；错位锚组 -> 覆盖率塌陷"""
    wh = np.array([[64, 64], [192, 192], [64, 64], [192, 192]], np.float32)
    good = np.array([[64, 64], [64, 64], [64, 64], [192, 192], [192, 192], [192, 192]], np.float32)
    bad = np.array([[4, 4], [4, 4], [4, 4], [4000, 4000], [4000, 4000], [4000, 4000]], np.float32)
    st_good, st_bad = anchor_stats(wh, good), anchor_stats(wh, bad)
    assert st_good["coverage"] == 1.0 and st_good["iou_mean"] > 0.99
    assert st_bad["coverage"] < 0.5, st_bad
    assert abs(best_ratio(np.array([[64, 64]], np.float32), np.array([[32, 32]], np.float32))[0] - 2.0) < 1e-6
    print(f"  判定口径正确（good coverage {st_good['coverage']:.2f} / bad {st_bad['coverage']:.2f}）")


def test_both_formats_equal(tmp_path):
    """COCO 与 YOLO 两种格式提取的归一化宽高一致；聚类结果一致"""
    whs = {}
    for fmt in ("coco", "yolo"):
        spec = load_dataset(str(_make_mini(tmp_path, fmt)))
        wh, ds, n_deg = collect_norm_wh(spec, "train", limit=0, progress=False)
        assert len(wh) == 16 and n_deg == 0
        whs[fmt] = wh
    assert np.allclose(whs["coco"], whs["yolo"])
    assert np.allclose(np.unique(np.round(whs["coco"], 6), axis=0), [[0.2, 0.2], [0.6, 0.6]])

    a_c, _ = kmeans_anchors(whs["coco"] * 320, 6, n_init=3, seed=0)
    a_y, _ = kmeans_anchors(whs["yolo"] * 320, 6, n_init=3, seed=0)
    assert np.array_equal(a_c, a_y)
    assert np.allclose(a_c[:3], 64, atol=4) and np.allclose(a_c[3:], 192, atol=4), a_c
    print(f"  双格式提取一致（16 框，锚 {a_c[:, 0].tolist()}）")


def test_cli_smoke(tmp_path):
    """CLI 端到端：v3-tiny 默认（判定 + 片段）；通用路径（无锚段模型 + 3 级自定义 n）"""
    desc = _make_mini(tmp_path, "coco")
    r = subprocess.run(["python", "scripts/compute_anchors.py", "--data", str(desc),
                        "--imgsz", "320", "--n", "6", "--n-init", "2", "--limit", "8"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "anchors:" in out and "verdict" in out and "coverage" in out, out
    assert out.count("  - [[") == 2, out  # 默认 2 级（来自 v3-tiny yaml）

    r2 = subprocess.run(["python", "scripts/compute_anchors.py", "--data", str(desc),
                         "--model", "yolo26", "--n", "9", "--levels", "3", "--n-init", "2", "--limit", "8"],
                        cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "evaluation skipped" in r2.stdout, r2.stdout  # yolo26 yaml 无 anchors 段
    assert r2.stdout.count("  - [[") == 3, r2.stdout  # --levels 3 分组
    print("  CLI 冒烟通过（默认 v3-tiny + 通用 --model/--levels 路径）")

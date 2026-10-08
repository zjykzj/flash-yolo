"""YOLO txt 数据集读取：标签解析 / 命名回退 / 背景 / 全背景守卫 / 尺寸依赖的 targets / 描述符链路"""

import cv2
import numpy as np
import pytest

from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_eval_dataset, build_train_dataset
from data.loader import collate_fn
from data.yolo import YoloDataset, YoloTrainDataset, label_path_for, parse_label_file


def _write_image(path, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((size, size, 3), np.uint8))


def _make_yolo(tmp_path, n=3, label_for=None, labels_rel="labels/train", names="cat, dog"):
    """迷你 YOLO 数据集 + 描述符；label_for(i, lbl_dir) 可自定义第 i 张图的标签文件

    默认每图一框 `0 0.5 0.5 0.25 0.25`（64px 图 -> 像素框 [24, 24, 40, 40]）。
    返回 (root, img_dir, lbl_dir, spec 文件路径)。
    """
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    lbl_dir = root / labels_rel
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _write_image(img_dir / f"{i:06d}.jpg")
        if label_for is not None:
            label_for(i, lbl_dir)
        else:
            (lbl_dir / f"{i:06d}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    spec_file = tmp_path / "yolo-ds.yaml"
    spec_file.write_text(
        f"format: yolo\npath: {root}\nnames: [{names}]\n"
        "train: {images: images/train}\nval: {images: images/train}\n", encoding="utf-8")
    return root, img_dir, lbl_dir, spec_file


def test_label_parse_and_pixel_roundtrip(tmp_path):
    """归一化标签 -> 像素 GT（必须等 load_image 拿到图尺寸之后）"""
    _, _, _, spec_file = _make_yolo(tmp_path, n=1)
    ds = build_eval_dataset(load_dataset(str(spec_file)), "val")
    with pytest.raises(RuntimeError, match="call load_image"):
        ds.targets(0)  # 未知尺寸前不给 GT
    img = ds.load_image(0)
    assert img.shape == (64, 64, 3)
    (box, cls), = ds.targets(0)
    assert cls == 0
    np.testing.assert_allclose(box, [24, 24, 40, 40], atol=1e-4)
    print("  归一化 -> 像素 GT 往返正确:", box)


def test_parse_label_file_errors(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 0.5 0.5 0.25 0.25 0.1\n")  # 6 段 = 分割
    with pytest.raises(ValueError, match="segmentation masks are not supported"):
        parse_label_file(p, nc=2)
    p.write_text("2 0.5 0.5 0.25 0.25\n")  # cls 越界（nc=2）
    with pytest.raises(ValueError, match="out of range"):
        parse_label_file(p, nc=2)
    p.write_text("0 0.5 0.5 0.0 0.25\n")  # 零宽
    with pytest.raises(ValueError, match="degenerate box"):
        parse_label_file(p, nc=2)
    p.write_text("0 nan 0.5 0.25 0.25\n")  # 任一字段非有限值都拦下
    with pytest.raises(ValueError, match="non-finite value"):
        parse_label_file(p, nc=2)
    p.write_text("\n  \n0 0.5 0.5 0.25 0.25\n")  # 空行跳过
    assert len(parse_label_file(p, nc=2)) == 1
    print("  标签解析的 4 类报错与空行跳过正确")


def test_missing_or_empty_label_is_background(tmp_path):
    def label_for(i, lbl_dir):
        if i == 0:
            (lbl_dir / f"{i:06d}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        elif i == 1:
            (lbl_dir / f"{i:06d}.txt").write_text("")  # 空标签 = 背景
        # i == 2 干脆不写文件 = 背景
    _, _, _, spec_file = _make_yolo(tmp_path, n=3, label_for=label_for)
    ds = build_train_dataset(TrainConfig(), load_dataset(str(spec_file)), "train", augment=False)
    assert len(ds) == 3 and ds.n_backgrounds == 2
    assert ds.n_instances == 1 and ds.n_categories == 2
    _, labels = ds[0]
    assert len(labels) == 1 and labels[0][0] == 0
    _, labels = ds[1]
    assert len(labels) == 0
    print("  无/空标签文件 = 背景（YOLO 合法样本）")


def test_numeric_stem_fallback_and_priority(tmp_path):
    """<stem>.txt 优先；缺失时退 str(int(stem)).txt（dataflow-cv 命名 + 常见数字约定）"""
    def label_for(i, lbl_dir):
        if i == 0:  # 000000.jpg：标签只有 0.txt（去前导零）
            (lbl_dir / "0.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        elif i == 1:  # 000001.jpg：两种命名都在 -> stem 版本优先（写不同的类，便于判别）
            (lbl_dir / "000001.txt").write_text("1 0.5 0.5 0.25 0.25\n")
            (lbl_dir / "1.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        else:
            (lbl_dir / f"{i:06d}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    _, img_dir, lbl_dir, spec_file = _make_yolo(tmp_path, n=3, label_for=label_for)
    assert label_path_for(img_dir / "000000.jpg", lbl_dir).name == "0.txt"
    assert label_path_for(img_dir / "000001.jpg", lbl_dir).name == "000001.txt"
    ds = build_train_dataset(TrainConfig(), load_dataset(str(spec_file)), "train", augment=False)
    assert ds.n_backgrounds == 0, "两张回退命中的图都不该是背景"
    _, labels = ds[1]
    assert labels[0][0] == 1, "两种命名并存时 <stem>.txt 优先"
    print("  数字 stem 回退与优先级正确")


def test_all_backgrounds_guard(tmp_path):
    """标签目录填错（一个都没命中）时必须报错——否则会安静地训练 100% 背景"""
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    (root / "labels" / "train").mkdir(parents=True)  # 目录存在但为空
    for i in range(3):
        _write_image(img_dir / f"{i:06d}.jpg")
    spec_file = tmp_path / "yolo-ds.yaml"
    spec_file.write_text(f"format: yolo\npath: {root}\nnames: [cat, dog]\n"
                         "train: {images: images/train}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no label files found"):
        build_train_dataset(TrainConfig(), load_dataset(str(spec_file)), "train")
    print("  全背景守卫报错正确")


def test_dir_and_list_modes_equivalent(tmp_path):
    """图片目录模式与 .txt 清单模式产出相同（图片按排序、标签同口径）"""
    root, img_dir, _, _ = _make_yolo(tmp_path, n=3)
    (root / "train.txt").write_text("\n".join(f"images/train/{i:06d}.jpg" for i in range(3)) + "\n")
    spec_file = tmp_path / "list-ds.yaml"
    spec_file.write_text(f"format: yolo\npath: {root}\nnames: [cat, dog]\n"
                         "train: {images: train.txt, labels: labels/train}\n"
                         "val: {images: train.txt, labels: labels/train}\n", encoding="utf-8")
    spec = load_dataset(str(spec_file))
    ds = build_train_dataset(TrainConfig(), spec, "train", augment=False)
    ref = build_train_dataset(TrainConfig(), load_dataset(str(tmp_path / "yolo-ds.yaml")), "train",
                              augment=False)
    assert [p.name for p in ds.images] == [p.name for p in ref.images]
    assert ds.n_instances == ref.n_instances and ds.n_backgrounds == ref.n_backgrounds
    _, lb = ds[0]
    _, lb_ref = ref[0]
    np.testing.assert_allclose(lb, lb_ref, atol=1e-6)
    print("  目录模式 / .txt 清单模式一致")


def test_missing_image_in_list_counted(tmp_path):
    """清单里有、磁盘上没有的图：剔除 + 计数（与 COCO 侧同口径）"""
    root, _, _, _ = _make_yolo(tmp_path, n=2)
    (root / "train.txt").write_text("images/train/000000.jpg\nimages/train/gone.jpg\n")
    spec_file = tmp_path / "list-ds.yaml"
    spec_file.write_text(f"format: yolo\npath: {root}\nnames: [cat, dog]\n"
                         "train: {images: train.txt, labels: labels/train}\n", encoding="utf-8")
    ds = build_train_dataset(TrainConfig(), load_dataset(str(spec_file)), "train")
    assert len(ds) == 1 and ds.n_missing == 1 and ds.first_missing == "gone.jpg"
    print("  清单缺图剔除与计数正确")


def test_limit_slices_first_n(tmp_path):
    _, _, _, spec_file = _make_yolo(tmp_path, n=5)
    ds = build_train_dataset(TrainConfig(), load_dataset(str(spec_file)), "train", limit=2)
    assert len(ds) == 2
    assert [p.name for p in ds.images] == ["000000.jpg", "000001.jpg"]
    print("  limit 取排序后前 N 张")


def test_corrupt_image_and_gt_dict_skip(tmp_path):
    """训练侧坏图 -> 空白画布 + 空标签 + 计数；评估侧坏图 -> GT dict 跳过该图"""
    root, img_dir, lbl_dir, spec_file = _make_yolo(tmp_path, n=2)
    (img_dir / "000000.jpg").write_bytes(b"not a jpeg")
    spec = load_dataset(str(spec_file))
    train = build_train_dataset(TrainConfig(), spec, "train", augment=False)
    tensor, labels = train[0]
    assert tensor.shape == (3, 640, 640) and len(labels) == 0 and train.n_corrupt == 1

    ev = build_eval_dataset(spec, "val")
    ev.load_image(0)  # 坏图：不记录尺寸
    ev.load_image(1)
    gt = ev.gt_dict()
    assert [i["id"] for i in gt["images"]] == [2], "坏图不进 GT（该图不会参与评估）"
    assert gt["categories"] == [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
    assert len(gt["annotations"]) == 1 and gt["annotations"][0]["iscrowd"] == 0
    print("  corrupt 语义与 GT dict 跳过正确")


def test_factory_returns_yolo_classes(tmp_path):
    """工厂按 format 分派；描述符链路（含 labels 镜像）解析正确"""
    _, _, lbl_dir, spec_file = _make_yolo(tmp_path)
    spec = load_dataset(str(spec_file))
    ds = build_train_dataset(TrainConfig(), spec, "train", augment=False)
    ev = build_eval_dataset(spec, "val")
    assert isinstance(ds, YoloTrainDataset) and isinstance(ev, YoloDataset)
    assert spec.role("train").labels == lbl_dir  # images -> labels 镜像
    imgs, targets = collate_fn([ds[0], ds[1], ds[2]])
    assert imgs.shape == (3, 3, 640, 640) and targets.shape == (3, 6)
    print("  工厂分派与 labels 镜像正确")

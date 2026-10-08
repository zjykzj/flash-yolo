"""训练数据集验收：类别映射口径 / crowd 剔除 / limit / letterbox 标签映射 / collate / worker 线程与 fork 契约
/ 扫描统计（背景·缺图·crowd）与惰性 corrupt 计数 / 描述符 names 与 json 对账

数据集一律经「描述符 -> data.build 工厂 -> 加载器」的真实链路构建（与训练一致）。
"""

import io
import json
import os
import time
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import data.loader
from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_eval_dataset, build_train_dataset
from data.loader import collate_fn, worker_init_fn
from data.scan import scan_summary


def _add_extra_image(data_dir, name, ann=None):
    """往迷你 COCO 的标注里追加一张图（是否落盘由调用方决定）

    ann 非 None 时同时追加一条标注。用于构造"标注里有、磁盘上没有"与"磁盘上损坏"两类图。
    """
    ann_file = data_dir / "annotations" / "instances_train2017.json"
    anns = json.loads(ann_file.read_text())
    img_id = max(i["id"] for i in anns["images"]) + 1
    anns["images"].append({"id": img_id, "file_name": name, "width": 64, "height": 64})
    if ann is not None:
        anns["annotations"].append({"id": 100 + img_id, "image_id": img_id, "category_id": 1,
                                    "bbox": ann, "iscrowd": 0})
    ann_file.write_text(json.dumps(anns))


def _make_fixture(tmp_path):
    """3 图 2 类（非连续 category_id）+ 1 crowd 标注的迷你 COCO"""
    data_dir = tmp_path / "data"
    img_dir = data_dir / "train2017"
    ann_dir = data_dir / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(img_dir / name), np.zeros((64, 64, 3), np.uint8))

    anns = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
            {"id": 3, "file_name": "c.jpg", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 3, "bbox": [30, 30, 10, 10], "iscrowd": 1},
            {"id": 12, "image_id": 2, "category_id": 3, "bbox": [0, 0, 8, 8], "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}],
    }
    with open(ann_dir / "instances_train2017.json", "w") as f:
        json.dump(anns, f)
    return data_dir


def _descriptor(data_dir, tmp_path, names="cat, dog"):
    """写一个指向夹具的描述符 yaml 并解析（train/val 角色共用同一份夹具）"""
    p = tmp_path / "ds.yaml"
    p.write_text(
        f"format: coco\npath: {data_dir}\nnames: [{names}]\n"
        "train: {images: train2017, ann: annotations/instances_train2017.json}\n"
        "val: {images: train2017, ann: annotations/instances_train2017.json}\n",
        encoding="utf-8")
    return load_dataset(str(p))


def test_category_mapping_consistent_with_train_and_eval(tmp_path):
    """训练/评估两侧的类别映射与描述符 names 口径一致（工厂的 names 对账是硬校验）"""
    data_dir = _make_fixture(tmp_path)
    spec = _descriptor(data_dir, tmp_path)
    ds = build_train_dataset(TrainConfig(), spec, "train")
    ref = build_eval_dataset(spec, "val")
    assert ds.cat_id_to_idx == ref.cat_id_to_idx, f"{ds.cat_id_to_idx} vs {ref.cat_id_to_idx}"
    assert ds.names == {0: "cat", 1: "dog"} == ref.names
    print("  类别映射与描述符 names 一致", ds.cat_id_to_idx)


def test_names_mismatch_with_json_raises(tmp_path):
    """描述符 names 与 json categories 不一致 -> 数据集构建即报错（报首个不一致下标）"""
    data_dir = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match=r"names\[1\]"):
        build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path, names="cat, bird"), "train")
    with pytest.raises(ValueError, match="defines 3 classes"):
        build_eval_dataset(_descriptor(data_dir, tmp_path, names="cat, dog, bird"), "val")
    print("  names 与 json 对账报错正确")


def test_crowd_excluded_from_train(tmp_path):
    """iscrowd=1 训练侧剔除；评估侧（CocoDataset 口径）保留"""
    data_dir = _make_fixture(tmp_path)
    spec = _descriptor(data_dir, tmp_path)
    ds = build_train_dataset(TrainConfig(), spec, "train")
    ref = build_eval_dataset(spec, "val")
    assert len(ds.labels[0]) == 1, f"img1 训练标签应剔除 crowd: {ds.labels[0]}"
    assert len(ref.targets(0)) == 2, "评估侧应保留 crowd（val 口径）"
    assert ds.labels[0][0][0] == 0 and list(ds.labels[0][0][1:]) == [10, 10, 30, 30]
    assert len(ds.labels[2]) == 0, "img3 无标签应为空数组"
    print("  crowd 剔除正确，训练/验证口径分离")


def test_limit_and_len(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train", limit=2)
    assert len(ds) == 2
    assert ds.images[0]["id"] == 1
    print("  limit 切片正确")


def test_letterbox_labels(tmp_path):
    """augment=False 路径：等比缩放 + 灰边，标签随 ratio/pad 变换"""
    data_dir = _make_fixture(tmp_path)
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train", augment=False)
    tensor, labels = ds[0]
    assert tensor.shape == (3, 640, 640)
    # 64x64 -> 640x640: ratio=10, pad=0; [10,10,30,30] -> [100,100,300,300]
    assert len(labels) == 1
    np.testing.assert_allclose(labels[0], [0, 100, 100, 300, 300], atol=1e-3)
    print("  letterbox 标签映射正确:", labels[0])


def test_collate(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train", augment=False)
    imgs, targets = collate_fn([ds[0], ds[1], ds[2]])
    assert imgs.shape == (3, 3, 640, 640)
    # img0 1 框, img1 1 框, img2 0 框 -> (2, 6)
    assert targets.shape == (2, 6), f"targets: {targets.shape}"
    assert (targets[:, 0].numpy() == [0, 1]).all(), "batch_idx 错误"
    print("  collate 形状与 batch_idx 正确")


def test_scan_summary_line():
    """`└` 续行排版：训练与评估共用同一函数（两处漂移会让两套日志对不上）"""
    ds = SimpleNamespace(n_instances=849949, n_categories=80, n_crowd_excluded=10052,
                         parse_time=12.55, scan_time=6.62)
    assert scan_summary(ds) == ("       └ 849949 instances · 80 categories · crowd 10052 excluded · "
                                "parse 12.6s + scan 6.6s")
    assert "crowd" not in scan_summary(SimpleNamespace(n_instances=36781, n_categories=80, n_crowd_excluded=0,
                                                       parse_time=0.31, scan_time=0.05))
    # YOLO 侧无解析段（parse_time=0）：省略 `parse Xs + `
    yolo = SimpleNamespace(n_instances=100, n_categories=2, n_crowd_excluded=0, parse_time=0.0, scan_time=1.2)
    assert scan_summary(yolo) == "       └ 100 instances · 2 categories · scan 1.2s"
    assert scan_summary(ds, " · limit 8").endswith("· limit 8"), "val 侧的 --limit 注记应接在尾"
    print("  `└` 续行排版正确（含 YOLO 无解析段）")


def test_scan_stats(tmp_path):
    """扫描统计：实例数 / 类别数 / 背景图（无框）/ crowd 剔除数 / 缺图数"""
    data_dir = _make_fixture(tmp_path)
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train")
    assert (ds.n_instances, ds.n_categories) == (2, 2), f"{ds.n_instances} / {ds.n_categories}"
    assert ds.n_crowd_excluded == 1, "img1 的 iscrowd=1 应被剔除并计数"
    assert ds.n_backgrounds == 1, "img3 无框 = 背景图"
    assert ds.n_missing == 0 and ds.first_missing is None
    assert ds.parse_time >= 0 and ds.scan_time >= 0
    print(f"  扫描统计正确：{ds.n_instances} 实例 / {ds.n_backgrounds} 背景 / {ds.n_crowd_excluded} crowd")


def test_missing_image_excluded_and_counted(tmp_path):
    """标注里有、磁盘上没有的图：剔除 + 计数 + 记首个文件名（官方 train2017 少 1 张的兜底）"""
    data_dir = _make_fixture(tmp_path)
    _add_extra_image(data_dir, "missing.jpg")  # 只进标注，不落盘
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train")
    assert len(ds) == 3, "缺图应从 images 中剔除"
    assert ds.n_missing == 1 and ds.first_missing == "missing.jpg"
    print("  缺图剔除与计数正确")


def test_scan_progress_line(tmp_path, monkeypatch):
    """progress=True：先一行静态提示（stdout，含完整路径），再画扫描条（写 progress_file）

    动态行用文件名（完整路径在静态提示行），且行宽受终端宽度钳制（COLUMNS 固定 120：不触发截断，
    截断本身由 tests/test_progress.py 覆盖）。
    """
    monkeypatch.setenv("COLUMNS", "120")
    data_dir = _make_fixture(tmp_path)
    buf = io.StringIO()
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train",
                             progress=True, progress_file=buf)
    out = buf.getvalue()
    assert "Scanning" in out and "instances_train2017.json" in out
    assert "3 images, 1 backgrounds, 0 missing" in out, f"扫描条计数：{out!r}"
    assert "100% [████████████] 3/3" in out, f"扫描条渲染：{out!r}"
    assert all(len(f.rstrip("\n")) <= 119 for f in out.split("\r") if f.strip()), f"进度行超宽：{out!r}"
    assert len(ds) == 3
    print("  扫描行渲染正确（静态提示 + 进度条 + 计数 + 宽度钳制）")


def test_corrupt_image_counted_as_background(tmp_path):
    """imread 失败：不抛异常，返回空白画布 + 空标签（当背景样本），并计数"""
    data_dir = _make_fixture(tmp_path)
    _add_extra_image(data_dir, "broken.jpg", ann=[1, 1, 5, 5])
    (data_dir / "train2017" / "broken.jpg").write_bytes(b"not a jpeg")
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train", augment=False)
    assert len(ds) == 4 and ds.n_corrupt == 0
    tensor, labels = ds[3]  # broken.jpg 在磁盘上（扫描时保留），但读不出来
    assert tensor.shape == (3, 640, 640) and len(labels) == 0
    assert ds.n_corrupt == 1
    print("  corrupt 图返回空白样本并计数")


def test_val_corrupt_image_counted(tmp_path):
    """评估侧读失败：返回空白图 + GT 保留（诚实记为漏检），计数在父进程内"""
    data_dir = _make_fixture(tmp_path)
    _add_extra_image(data_dir, "broken.jpg", ann=[1, 1, 5, 5])
    (data_dir / "train2017" / "broken.jpg").write_bytes(b"not a jpeg")
    ref = build_eval_dataset(_descriptor(data_dir, tmp_path), "val")
    assert ref.n_corrupt == 0
    assert ref.load_image(3).shape == (1, 1, 3)
    assert ref.n_corrupt == 1
    assert len(ref.targets(3)) == 1, "评估侧 GT 保留（该图记为漏检）"
    print("  评估侧 corrupt 计数与 GT 保留正确")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="需要 os.fork（复现 DataLoader worker 的 fork 路径）")
def test_corrupt_counter_visible_across_fork(tmp_path):
    """worker 进程里 +1，父进程读得到（DataLoader 是多进程，计数必须跨 fork 可见）"""
    data_dir = _make_fixture(tmp_path)
    _add_extra_image(data_dir, "broken.jpg")
    (data_dir / "train2017" / "broken.jpg").write_bytes(b"not a jpeg")
    ds = build_train_dataset(TrainConfig(), _descriptor(data_dir, tmp_path), "train")
    with warnings.catch_warnings():  # 多线程进程 fork 的 DeprecationWarning——本测试正是要 fork
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        code = 0
        try:
            ds.load_image(3)
        except BaseException:
            code = 3
        os._exit(code)  # 子进程里不再跑 pytest 的 atexit
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, f"子进程异常 status={status}"
    assert ds.n_corrupt == 1, "子进程里的 +1 应在父进程可见（共享计数器）"
    print("  corrupt 计数跨 fork 共享正确")


def test_worker_thread_contract():
    """cv2 线程数由父进程在导入时压到 1、子进程继承；worker 内绝不再调 setNumThreads

    OpenCV 的 pthreads 线程池非 fork 安全：父进程跑过一次 cv2 并行运算后（训练里 = epoch 1 的
    验证 imread + 增强抽样 imwrite），子进程再调 cv2.setNumThreads 会永久阻塞在 futex。
    """
    n_cv = cv2.getNumThreads()
    assert n_cv == 1, (
        f"data/loader.py 导入后 cv2 应为单线程，实际 {n_cv} —— 父进程侧没压线程数："
        "worker 会按核数超订，且 fork 后有死锁风险（见 data/loader.py 顶部注释）"
    )

    prev_cv, prev_torch, prev_rng = n_cv, torch.get_num_threads(), data.loader._worker_rng
    try:
        cv2.setNumThreads(3)  # 取非 1：worker 里若调用 setNumThreads(1) 就能被看见
        worker_init_fn(0)
        assert cv2.getNumThreads() == 3, "worker_init_fn 里不许调 cv2.setNumThreads（fork 不安全）"
        assert torch.get_num_threads() == 1, "worker 内 torch 线程数应为 1"
    finally:
        cv2.setNumThreads(prev_cv)
        torch.set_num_threads(prev_torch)
        data.loader._worker_rng = prev_rng
    print("  worker 线程契约：cv2 继承父进程单线程、torch 置 1")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="需要 os.fork（复现 DataLoader worker 的 fork 路径）")
def test_worker_init_survives_fork_after_parent_cv2_use():
    """fork 安全性闸门：父进程已按多线程初始化 OpenCV 线程池时，fork 出的 worker 仍须正常起来

    复现 2026-10 的训练挂死：父进程跑完 epoch 1（验证 imread + 增强抽样 imwrite）后再建
    DataLoader，16/16 worker 全卡在 cv2.setNumThreads 上（futex），训练停在 epoch 2 第一个 batch 前。
    """
    prev_cv = cv2.getNumThreads()
    timeout = 20.0
    try:
        cv2.setNumThreads(4)  # 父进程：按多线程初始化线程池（= epoch 1 结束后父进程的真实状态）
        rng = np.random.default_rng(0)
        img = (rng.random((1200, 1200, 3)) * 255).astype(np.uint8)
        big = cv2.resize(img, (2000, 2000), interpolation=cv2.INTER_AREA)  # 走 parallel_for_
        cv2.GaussianBlur(big, (31, 31), 5.0)

        with warnings.catch_warnings():  # 多线程进程 fork 的 DeprecationWarning——本测试正是要 fork
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # 子进程 = worker 启动路径；os._exit 避免在子进程里跑 pytest 的 atexit
            code = 0
            try:
                worker_init_fn(0)
                cv2.GaussianBlur(big, (31, 31), 5.0)  # fork 之后仍要能真正用 cv2
            except BaseException:
                code = 3
            os._exit(code)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            wpid, status = os.waitpid(pid, os.WNOHANG)
            if wpid == pid:
                assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, f"worker 子进程异常退出 status={status}"
                print("  fork 后 worker_init_fn 正常返回（父进程线程池 4 线程，子进程 cv2 可用）")
                return
            time.sleep(0.05)
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        pytest.fail(f"worker_init_fn 在 fork 出的子进程中阻塞 >{timeout:.0f}s —— 不要在 worker 里调 "
                    "cv2.setNumThreads（OpenCV pthreads 池非 fork 安全，见 data/loader.py 顶部注释）")
    finally:
        cv2.setNumThreads(prev_cv)

"""训练数据集验收：类别映射口径 / crowd 剔除 / limit / letterbox 标签映射 / collate / worker 线程与 fork 契约"""

import json
import os
import time
import warnings

import cv2
import numpy as np
import pytest
import torch

import data.dataset
from data.coco import CocoDataset
from data.dataset import CocoTrainDataset, collate_fn, worker_init_fn
from config.train import TrainConfig


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


def test_category_mapping_consistent_with_coco(tmp_path):
    """与 CocoDataset 的类别映射口径逐项一致"""
    data_dir = _make_fixture(tmp_path)
    cfg = TrainConfig(data_dir=str(data_dir))
    ds = CocoTrainDataset(cfg, split="train2017")
    ref = CocoDataset(str(data_dir), split="train2017")
    assert ds.cat_id_to_idx == ref.cat_id_to_idx, f"{ds.cat_id_to_idx} vs {ref.cat_id_to_idx}"
    print("  类别映射与 CocoDataset 一致", ds.cat_id_to_idx)


def test_crowd_excluded_from_train(tmp_path):
    """iscrowd=1 训练侧剔除；CocoDataset（val 口径）保留"""
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017")
    ref = CocoDataset(str(data_dir), split="train2017")
    assert len(ds.labels[0]) == 1, f"img1 训练标签应剔除 crowd: {ds.labels[0]}"
    assert len(ref.targets(0)) == 2, "CocoDataset 应保留 crowd（val 口径）"
    assert ds.labels[0][0][0] == 0 and list(ds.labels[0][0][1:]) == [10, 10, 30, 30]
    assert len(ds.labels[2]) == 0, "img3 无标签应为空数组"
    print("  crowd 剔除正确，训练/验证口径分离")


def test_limit_and_len(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", limit=2)
    assert len(ds) == 2
    assert ds.images[0]["id"] == 1
    print("  limit 切片正确")


def test_letterbox_labels(tmp_path):
    """augment=False 路径：等比缩放 + 灰边，标签随 ratio/pad 变换"""
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", augment=False)
    tensor, labels = ds[0]
    assert tensor.shape == (3, 640, 640)
    # 64x64 -> 640x640: ratio=10, pad=0; [10,10,30,30] -> [100,100,300,300]
    assert len(labels) == 1
    np.testing.assert_allclose(labels[0], [0, 100, 100, 300, 300], atol=1e-3)
    print("  letterbox 标签映射正确:", labels[0])


def test_collate(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", augment=False)
    imgs, targets = collate_fn([ds[0], ds[1], ds[2]])
    assert imgs.shape == (3, 3, 640, 640)
    # img0 1 框, img1 1 框, img2 0 框 -> (2, 6)
    assert targets.shape == (2, 6), f"targets: {targets.shape}"
    assert (targets[:, 0].numpy() == [0, 1]).all(), "batch_idx 错误"
    print("  collate 形状与 batch_idx 正确")


def test_worker_thread_contract():
    """cv2 线程数由父进程在导入时压到 1、子进程继承；worker 内绝不再调 setNumThreads

    OpenCV 的 pthreads 线程池非 fork 安全：父进程跑过一次 cv2 并行运算后（训练里 = epoch 1 的
    验证 imread + 增强抽样 imwrite），子进程再调 cv2.setNumThreads 会永久阻塞在 futex。
    """
    n_cv = cv2.getNumThreads()
    assert n_cv == 1, (
        f"data/dataset.py 导入后 cv2 应为单线程，实际 {n_cv} —— 父进程侧没压线程数："
        "worker 会按核数超订，且 fork 后有死锁风险（见 data/dataset.py 顶部注释）"
    )

    prev_cv, prev_torch, prev_rng = n_cv, torch.get_num_threads(), data.dataset._worker_rng
    try:
        cv2.setNumThreads(3)  # 取非 1：worker 里若调用 setNumThreads(1) 就能被看见
        worker_init_fn(0)
        assert cv2.getNumThreads() == 3, "worker_init_fn 里不许调 cv2.setNumThreads（fork 不安全，见文件顶部注释）"
        assert torch.get_num_threads() == 1, "worker 内 torch 线程数应为 1"
    finally:
        cv2.setNumThreads(prev_cv)
        torch.set_num_threads(prev_torch)
        data.dataset._worker_rng = prev_rng
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
                    "cv2.setNumThreads（OpenCV pthreads 池非 fork 安全，见 data/dataset.py 顶部注释）")
    finally:
        cv2.setNumThreads(prev_cv)

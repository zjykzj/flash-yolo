"""训练可复现闸门（门控）：同 seed 两次训练逐位相同 + resume 与不中断等价

默认跳过；`FLASH_YOLO_REPRO=1` 时执行（CPU 上 ~40s，GPU 上更快）。
两套机制分开守：① 默认 `deterministic: true` 的 kernel 侧开关 + 种子管线（同 seed 两跑）；
② 每 epoch 重播的 RNG 流域 + `capture_rng()`（中断续训）。任何一处回退都会在这里红。
"""

import os

import pytest
import torch

from config.train_config import TrainConfig
from safetensors.torch import load_file
from train.trainer import Trainer
from test_train_smoke import _make_mini_coco  # 复用冒烟夹具（同一份迷你 COCO 构造器）

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_YOLO_REPRO") != "1", reason="FLASH_YOLO_REPRO=1 才执行复现闸门"
)


def _cfg(data, **over):
    base = dict(data=str(data), epochs=2, batch=8, nbs=8, imgsz=320, workers=0,
                val_epochs=0, limit=0, amp=False, mosaic=0.5, mixup=0.0, copy_paste=0.0,
                close_mosaic=0, warmup_epochs=0.0, save_period=0, aug_samples=0)
    base.update(over)
    return TrainConfig(**base)


def _weights(run_dir):
    return load_file(run_dir / "weights" / "last.safetensors")


def _assert_bit_identical(a, b, what):
    diff = {k: (a[k].float() - b[k].float()).abs().max().item() for k in a}
    worst = max(diff.values())
    assert all(torch.equal(a[k], b[k]) for k in a), f"{what}: {sum(v > 0 for v in diff.values())}/{len(a)} " \
                                                    f"张量有位差, max|Δ| = {worst:.3e}"
    print(f"  {what}: {len(a)} 张量逐位相同")


def test_same_seed_bit_identical(tmp_path):
    """同 seed 两次训练（default deterministic + workers=0 主进程增强）-> 权重逐位相同"""
    data = _make_mini_coco(tmp_path)
    for tag in ("a", "b"):
        torch.manual_seed(0)
        Trainer(_cfg(data), device="cpu", run_dir=tmp_path / f"run-{tag}").train()
    _assert_bit_identical(_weights(tmp_path / "run-a"), _weights(tmp_path / "run-b"),
                          "同 seed 两跑（2 epoch）")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_same_seed_bit_identical_gpu(tmp_path):
    """同 seed 两次 GPU 训练逐位相同 —— 靠默认 deterministic 的 kernel 侧开关（关掉必红）"""
    data = _make_mini_coco(tmp_path)
    for tag in ("a", "b"):
        torch.manual_seed(0)
        Trainer(_cfg(data), device="cuda", run_dir=tmp_path / f"gpu-{tag}").train()
    _assert_bit_identical(_weights(tmp_path / "gpu-a"), _weights(tmp_path / "gpu-b"),
                          "同 seed 两跑（GPU, 2 epoch）")


def test_same_seed_bit_identical_seg(tmp_path):
    """分割训练同 seed 两跑逐位相同（多边形增强 + 掩码栅格化进入 RNG 流域）"""
    data = _make_mini_coco(tmp_path, seg=True)
    cfg = _cfg(data, model="yolo26-seg", mask_ratio=4, overlap_mask=True)
    for tag in ("a", "b"):
        torch.manual_seed(0)
        Trainer(cfg, device="cpu", run_dir=tmp_path / f"seg-{tag}").train()
    _assert_bit_identical(_weights(tmp_path / "seg-a"), _weights(tmp_path / "seg-b"),
                          "同 seed 两跑（seg, 2 epoch）")


def test_resume_matches_uninterrupted(tmp_path):
    """1 轮后从 resume.pt 续训 == 一次跑完 2 轮（RNG 流域每 epoch 重播 + 三套 RNG 入 checkpoint）"""
    data = _make_mini_coco(tmp_path)
    torch.manual_seed(0)
    Trainer(_cfg(data), device="cpu", run_dir=tmp_path / "run-full").train()

    torch.manual_seed(0)
    part = tmp_path / "run-part"
    Trainer(_cfg(data, stop_after=1), device="cpu", run_dir=part).train()  # stop_after 不改调度，可比
    Trainer(_cfg(data), device="cpu", run_dir=part, resume=str(part / "resume.pt")).train()
    _assert_bit_identical(_weights(tmp_path / "run-full"), _weights(part), "一次跑完 vs resume 续训")


def test_resume_matches_uninterrupted_seg(tmp_path):
    """分割续训等价：掩码网格/多边形增强 RNG/优化器状态都随 resume.pt 恢复"""
    data = _make_mini_coco(tmp_path, n_img=8, seg=True)
    seg_kw = dict(model="yolo26-seg", mask_ratio=4, overlap_mask=True)
    torch.manual_seed(0)
    Trainer(_cfg(data, **seg_kw), device="cpu", run_dir=tmp_path / "seg-full").train()

    torch.manual_seed(0)
    part = tmp_path / "seg-part"
    Trainer(_cfg(data, **seg_kw, stop_after=1), device="cpu", run_dir=part).train()
    Trainer(_cfg(data, **seg_kw), device="cpu", run_dir=part, resume=str(part / "resume.pt")).train()
    _assert_bit_identical(_weights(tmp_path / "seg-full"), _weights(part), "seg 一次跑完 vs resume 续训")

"""DDP 训练闸门：设备解析 / loss 补偿（无环境依赖）+ 2 进程 CPU gloo 训练冒烟（门控）

默认只跑纯函数单测；`FLASH_YOLO_DDP=1` 时追加 2 进程冒烟（检测 + 分割 + resume 等价 +
world=1 路径等价）。2 进程由 subprocess 启动 tests/ddp_worker.py（显式 env:// rendezvous、
CUDA_VISIBLE_DEVICES=""），避免 mp.spawn 在 pytest 下的 pickle/import-path 问题。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from test_train_smoke import _make_mini_coco  # 复用冒烟夹具（迷你 COCO + 描述符）
from train.distributed import DistInfo, ddp_scale_loss, free_port, parse_device_ids, resolve_single_device

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKER = HERE / "ddp_worker.py"
DDP_ENABLED = os.environ.get("FLASH_YOLO_DDP") == "1"
REPRO_ENABLED = os.environ.get("FLASH_YOLO_REPRO") == "1"


# ---- 纯函数单测（任何环境都跑）----

def test_parse_device_ids_auto():
    assert parse_device_ids(None) == []
    assert parse_device_ids("") == []
    assert parse_device_ids("  ") == []


def test_parse_device_ids_single():
    assert parse_device_ids("cpu") == [torch.device("cpu")]
    assert parse_device_ids("cuda") == [torch.device("cuda")]
    assert parse_device_ids("0") == [torch.device("cuda", 0)]
    assert parse_device_ids("cuda:1") == [torch.device("cuda", 1)]


def test_parse_device_ids_multi():
    expected = [torch.device("cuda", 0), torch.device("cuda", 1)]
    assert parse_device_ids("0,1") == expected
    assert parse_device_ids("cuda:0,cuda:1") == expected
    assert parse_device_ids(" 1 , 0 ") == [torch.device("cuda", 1), torch.device("cuda", 0)]


@pytest.mark.parametrize("bad", ["0,0", "0,cpu", "cuda,cuda", "x", "0,", ",1", "cuda:x"])
def test_parse_device_ids_rejects(bad):
    with pytest.raises(ValueError):
        parse_device_ids(bad)


def test_resolve_single_device():
    assert resolve_single_device("cpu") == torch.device("cpu")
    assert resolve_single_device("1") == torch.device("cuda", 1)
    with pytest.raises(ValueError):  # 多卡必须走 DDP 入口
        resolve_single_device("0,1")


def test_dist_info_semantics():
    single = DistInfo()
    assert not single.enabled and single.is_main and single.backend == ""
    ddp = DistInfo(rank=1, local_rank=1, world_size=2, backend="nccl", device=torch.device("cuda", 1))
    assert ddp.enabled and not ddp.is_main


def test_ddp_scale_loss():
    """world=1 必须返回同一对象（单卡逐位不变）；world=2 = 2×（DDP 梯度平均的补偿）"""
    loss = torch.tensor(3.5)
    assert ddp_scale_loss(loss, 1) is loss
    assert ddp_scale_loss(loss, 0) is loss
    scaled = ddp_scale_loss(loss, 2)
    assert scaled is not loss and torch.equal(scaled, torch.tensor(7.0))


# ---- 2 进程 CPU gloo 冒烟（FLASH_YOLO_DDP=1）----

def _launch(tmp_path, data, model, run_dir, tag, epochs=1, stop_after=0, resume=None):
    """起 2 个 worker（env:// rendezvous），返回 (rank0 参数哈希, rank1 参数哈希)

    子进程日志落文件（不进 PIPE：单管道写满会让 rank 在集体操作前阻塞）。
    """
    port = free_port()
    procs, outs, logs = [], [], []
    for rank in (0, 1):
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
            "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank),
            "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        })
        out = tmp_path / f"{tag}-rank{rank}.sha"
        log = open(tmp_path / f"{tag}-rank{rank}.log", "w", encoding="utf-8")  # noqa: SIM115
        cmd = [sys.executable, str(WORKER), "--data", str(data), "--model", model,
               "--run-dir", str(run_dir), "--epochs", str(epochs), "--stop-after", str(stop_after),
               "--out", str(out)]
        if resume:
            cmd += ["--resume", str(resume)]
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT))
        outs.append(out)
        logs.append((log, tmp_path / f"{tag}-rank{rank}.log"))
    try:
        for rank, proc in enumerate(procs):
            try:
                proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                raise AssertionError(f"rank{rank} timed out\n{logs[rank][1].read_text()[-4000:]}")
            assert proc.returncode == 0, f"rank{rank} failed:\n{logs[rank][1].read_text()[-4000:]}"
    finally:
        for log, _ in logs:
            log.close()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    return [p.read_text().strip() for p in outs]


def _assert_run_products(run_dir, world_size=2):
    assert (run_dir / "weights" / "last.safetensors").exists()
    assert (run_dir / "weights" / "best.safetensors").exists()
    assert (run_dir / "resume.pt").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "run.rank1.log").exists()  # 非 0 rank 也落自己的日志
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["train"]["world_size"] == world_size, meta["train"]
    assert meta["train"]["backend"] == "gloo", meta["train"]
    import csv

    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["mAP"], rows  # 验证在 rank0 全量跑过、指标已落盘


@pytest.mark.skipif(not DDP_ENABLED, reason="FLASH_YOLO_DDP=1 才执行 2 进程 DDP 冒烟")
def test_ddp_detect_smoke(tmp_path):
    """检测：2 rank 梯度同步后参数逐位一致，rank0 产物齐全（global batch 4 = 每卡 2，nbs 8 -> accum 2）"""
    data = _make_mini_coco(tmp_path, n_img=8, size=160)
    run_dir = tmp_path / "run-det"
    hashes = _launch(tmp_path, data, "yolo26", run_dir, "det")
    assert hashes[0] == hashes[1], "ranks diverged after all-reduce"
    _assert_run_products(run_dir)


@pytest.mark.skipif(not DDP_ENABLED, reason="FLASH_YOLO_DDP=1 才执行 2 进程 DDP 冒烟")
def test_ddp_seg_smoke(tmp_path):
    """分割：同一条 DDP 路径（掩码 GT / 分割损失 / Box+Mask 验证），参数逐位一致"""
    data = _make_mini_coco(tmp_path, n_img=8, size=160, seg=True)
    run_dir = tmp_path / "run-seg"
    hashes = _launch(tmp_path, data, "yolo26-seg", run_dir, "seg")
    assert hashes[0] == hashes[1], "ranks diverged after all-reduce"
    _assert_run_products(run_dir)


@pytest.mark.skipif(not (DDP_ENABLED and REPRO_ENABLED),
                    reason="FLASH_YOLO_DDP=1 + FLASH_YOLO_REPRO=1 才执行 DDP resume 等价")
def test_ddp_resume_matches_uninterrupted(tmp_path):
    """2 轮一次跑完 vs 1 轮 + resume：每个 rank 的参数分别逐位相同"""
    data = _make_mini_coco(tmp_path, n_img=8, size=160)
    full = _launch(tmp_path, data, "yolo26", tmp_path / "run-full", "full", epochs=2)
    part_dir = tmp_path / "run-part"
    _launch(tmp_path, data, "yolo26", part_dir, "part", epochs=2, stop_after=1)
    resumed = _launch(tmp_path, data, "yolo26", part_dir, "resumed", epochs=2,
                      resume=part_dir / "resume.pt")
    assert full == resumed, "DDP resume diverged from an uninterrupted run"
    assert full[0] == full[1]


@pytest.mark.skipif(not DDP_ENABLED, reason="FLASH_YOLO_DDP=1 才执行 world=1 路径等价")
def test_world_one_path_matches_default(tmp_path):
    """DistInfo(world_size=1) 与不传 dist_info 的训练逐位相同（单卡路径零影响）"""
    from config.train_config import TrainConfig
    from train.trainer import Trainer

    data = _make_mini_coco(tmp_path, n_img=8, size=160)
    kwargs = dict(data=str(data), epochs=1, batch=4, nbs=4, imgsz=160, workers=0,
                  val_epochs=0, amp=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                  close_mosaic=0, warmup_epochs=0.0, aug_samples=0, plots=False,
                  save_period=0, diag_interval=0)
    for tag, info in (("plain", None), ("world1", DistInfo(world_size=1))):
        torch.manual_seed(0)
        Trainer(TrainConfig(**kwargs), device="cpu", run_dir=tmp_path / tag, dist_info=info).train()
    a = load_file(tmp_path / "plain" / "weights" / "last.safetensors")
    b = load_file(tmp_path / "world1" / "weights" / "last.safetensors")
    assert all(torch.equal(a[k], b[k]) for k in a), "world=1 DDP path changed single-GPU training"

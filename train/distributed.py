"""DDP 训练支撑：设备解析 / 进程组 / rank 工具（world_size=1 时全部退化为 no-op）

设计要点：
- 设备解析：`--device` 支持 None（auto）/ cpu / cuda / cuda:N / 裸索引 N / "0,1" / "cuda:0,cuda:1"。
  多设备列表只接受显式 CUDA 索引（DDP 走 NCCL），拒绝重复与 CPU/CUDA 混合。
- 启动方式：`--device 0,1` 由 scripts/train.py 起 N 个子进程（父进程设 RANK/LOCAL_RANK/
  WORLD_SIZE/MASTER_ADDR/MASTER_PORT）；也支持 `torchrun --standalone --nproc_per_node=2
  scripts/train.py ...`（检测到 WORLD_SIZE 环境变量即按已有 rendezvous 初始化）。
- 进程组：CUDA -> nccl，CPU -> gloo；超时 3h（rank0 单独验证时其余 rank 在 barrier 上等待属正常）。
- `ddp_scale_loss`：本工程的损失是 Σ_img 口径（见 train/loss/yolo26_detect.py 文件头），
  DDP 在 backward 时对梯度按 world_size 求平均；backward 前乘 world_size 才能让
  "每 rank 局部 Σ 再平均" = "全局 Σ"，与单卡同一 global batch 的梯度尺度一致
  （梯度裁剪阈值 10 的饱和行为也随之保持一致）。单卡原样返回同一对象 → 逐位不变。

只服务训练；eval / export / infer 不走这里（单进程）。
"""

import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

__all__ = [
    "DistInfo", "parse_device_ids", "resolve_single_device", "ddp_env", "init_distributed",
    "cleanup_distributed", "sync_barrier", "broadcast_object", "free_port", "ddp_scale_loss",
]


@dataclass(frozen=True)
class DistInfo:
    """一个训练进程的分布式身份（world_size=1 = 单卡/单进程，enabled=False）

    device 只在 init 后已知；直接构造（如测试里 `DistInfo(world_size=1)`）时可留 None，
    Trainer 沿用调用方给的 device。
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""  # "nccl" / "gloo"；world_size=1 时为空（无进程组）
    device: torch.device | None = None

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0


def _parse_one(part):
    """单个 device 记号 -> torch.device（cpu / cuda / cuda:N / 裸索引 N）"""
    low = part.lower()
    if low == "cpu":
        return torch.device("cpu")
    if low == "cuda":
        return torch.device("cuda")
    if low.startswith("cuda:"):
        index = low[5:]
    else:
        index = low
    if index.isdigit():
        return torch.device("cuda", int(index))
    raise ValueError(f"invalid --device entry {part!r} (expected cpu / cuda / cuda:N / N)")


def parse_device_ids(value):
    """`--device` -> [torch.device]；空列表 = 自动选择（CUDA 可用则 cuda，否则 cpu）

    多设备列表是 DDP 启动的判据，因此只接受互不重复的显式 CUDA 索引
    （`0,1` 或 `cuda:0,cuda:1`）；`cuda,cuda` 这类无索引写法会被拒绝。
    """
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    if any(not p for p in parts):
        raise ValueError(f"invalid --device {value!r}: empty entry")
    devices = [_parse_one(p) for p in parts]
    if len(devices) > 1:
        if any(d.type != "cuda" for d in devices):
            raise ValueError(f"invalid --device {value!r}: DDP requires CUDA device indices")
        if any(d.index is None for d in devices):
            raise ValueError(f"invalid --device {value!r}: give explicit indices (0,1 / cuda:0,cuda:1)")
        indexes = [d.index for d in devices]
        if len(set(indexes)) != len(indexes):
            raise ValueError(f"invalid --device {value!r}: duplicate device index")
    return devices


def resolve_single_device(value):
    """单进程设备解析（--device 给出 ≥2 个设备时明确报错，提示走 DDP 入口）"""
    devices = parse_device_ids(value)
    if len(devices) > 1:
        raise ValueError(
            f"--device {value!r} lists {len(devices)} devices but this process is single-device: "
            f"multi-GPU training needs the DDP entry (`--device 0,1` spawns one worker per GPU, "
            f"or use torchrun)")
    if not devices:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return devices[0]


def ddp_env():
    """torchrun / 父进程设置的环境变量 -> (rank, local_rank, world_size)；非 DDP 启动 -> None"""
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        return None
    world_size = int(world_size)
    if world_size <= 1:
        return None
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, local_rank, world_size


def _ddp_device(device_arg, local_rank, world_size):
    """子进程按 LOCAL_RANK 取设备；--device 给全列表时按 local_rank 对号"""
    devices = parse_device_ids(device_arg)
    if not devices:
        return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if len(devices) == 1:
        only = devices[0]
        if only.type == "cuda" and only.index is None:
            return torch.device("cuda", local_rank)
        return only
    if len(devices) != world_size:
        raise ValueError(f"--device lists {len(devices)} entries but WORLD_SIZE={world_size}")
    return devices[local_rank]


def _check_cuda_device(device):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but torch.cuda.is_available() is False")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device.index} out of range "
                           f"(device_count={torch.cuda.device_count()})")


def init_distributed(device_arg=None):
    """初始化训练进程（DDP 时建进程组），返回 DistInfo

    DDP（环境变量 WORLD_SIZE>1）：rank/local_rank 取自环境，设备按 LOCAL_RANK 选，
    CUDA 走 nccl / CPU 走 gloo。单进程：只解析并校验设备，不建进程组。
    调用方负责在结束时 cleanup_distributed()。
    """
    env = ddp_env()
    if env is None:
        devices = parse_device_ids(device_arg)
        if len(devices) > 1:
            raise RuntimeError(
                f"--device {device_arg!r} lists {len(devices)} devices but no DDP rendezvous was found "
                f"(WORLD_SIZE unset): run it through scripts/train.py, which spawns one worker per GPU")
        device = resolve_single_device(device_arg)
        if device.type == "cuda":
            _check_cuda_device(device)
            torch.cuda.set_device(device)  # 无索引时等价 current device 0；有索引时绑定本进程
        return DistInfo(device=device)

    rank, local_rank, world_size = env
    device = _ddp_device(device_arg, local_rank, world_size)
    if device.type == "cuda":
        _check_cuda_device(device)
        torch.cuda.set_device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    # nccl 显式绑定本 rank 的设备：barrier/集合操作不再猜 current device（省掉 torch 的告警）
    pg_kwargs = {"device_id": device} if backend == "nccl" else {}
    try:
        dist.init_process_group(backend=backend, timeout=timedelta(hours=3), **pg_kwargs)
    except Exception as e:  # noqa: BLE001 —— 转成带上下文的错误（常见：NCCL 不可用 / 端口被占）
        raise RuntimeError(
            f"failed to initialize {backend} process group (rank {rank}/{world_size}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}): {e}") from e
    info = DistInfo(rank=dist.get_rank(), local_rank=local_rank, world_size=dist.get_world_size(),
                    backend=backend, device=device)
    if info.rank != rank or info.world_size != world_size:
        raise RuntimeError(f"process group reports rank {info.rank}/{info.world_size} but the launcher "
                           f"set {rank}/{world_size}")
    return info


def cleanup_distributed():
    """销毁进程组（不做可能挂死的 barrier；调用方应在所有 rank 的正常退出路径上调用）"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_barrier():
    """DDP 同步点（world=1 / 未初始化时 no-op）"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src=0):
    """rank src 的对象广播到所有 rank（用于 run 目录等只在 rank0 决定的值；非 DDP 原样返回）"""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def free_port():
    """取一个空闲的 localhost TCP 端口（DDP 父进程给子进程选 MASTER_PORT）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ddp_scale_loss(loss, world_size):
    """DDP 梯度平均的补偿：backward 前把 Σ_img 口径的损失乘 world_size

    推导：单卡 global batch 的梯度 = Σ_over_all_images ∂L。DDP 中每 rank 算 Σ_its_images ∂L，
    all-reduce 取平均 → (1/W)·Σ_ranks Σ_local = (1/W)·全局 Σ。乘 W 后恢复全局 Σ。
    world_size<=1 返回**同一对象**（单卡逐位不变；回归测试守这一点）。
    """
    return loss if world_size <= 1 else loss * world_size

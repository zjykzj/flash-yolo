"""训练 checkpoint 落盘与恢复

- resume.pt：完整训练状态快照（模型/EMA/优化器/缩放器/epoch/best/rng），
  仅内部断点续训用（pickle，不出 runs/ 目录）
- best/last.safetensors：EMA 权重交付物（纯张量，可审计）
"""

from dataclasses import asdict

import numpy as np
import torch

from model.weights import save_weights

__all__ = ["save_resume", "load_resume", "save_best_last", "save_periodic", "capture_rng"]


def capture_rng():
    """当前 torch/cuda/numpy 随机状态（resume 可复现的关键）"""
    out = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def save_resume(path, model, ema, optimizer, scaler, epoch, best_fitness, cfg, run_dir):
    """完整训练状态 -> resume.pt（内部文件）"""
    ckpt = {
        "model_sd": model.state_dict(),
        "ema_sd": ema.ema.state_dict() if ema is not None else None,
        "ema_steps": ema.steps if ema is not None else None,  # decay 爬升依赖步数，必须恢复
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_fitness": best_fitness,
        "cfg": asdict(cfg),  # 纯 dict：dataclass 实例会把模块路径写进 pickle（改名后旧 resume.pt 无法反序列化）
        "run_dir": str(run_dir),
        "rng": capture_rng(),
    }
    torch.save(ckpt, path)


def load_resume(path, model, ema, optimizer, scaler=None):
    """恢复完整状态（strict 回载验证 key 完整性），返回 (epoch, best_fitness, cfg, run_dir)"""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_sd"], strict=True)
    if ema is not None and ckpt.get("ema_sd") is not None:
        ema.ema.load_state_dict(ckpt["ema_sd"], strict=True)
        ema.steps = ckpt.get("ema_steps", 0)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    rng = ckpt.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    return ckpt["epoch"], ckpt["best_fitness"], ckpt["cfg"], ckpt.get("run_dir")


def save_best_last(run_dir, model, is_best, raw_model=None):
    """EMA 权重 -> weights/last.safetensors（每 epoch）+ weights/best.safetensors（新高时）

    raw_model 非 None 时同步存 best_raw.safetensors（同一 epoch 的原始权重，
    用于复盘 EMA 与 raw 的差距；其 fitness 未单独评估）。
    """
    wdir = run_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    save_weights(model, wdir / "last.safetensors")
    if is_best:
        save_weights(model, wdir / "best.safetensors")
        if raw_model is not None:
            save_weights(raw_model, wdir / "best_raw.safetensors")


def save_periodic(run_dir, model, epoch, keep=3):
    """周期 checkpoint：weights/epoch{N:03d}.safetensors（1-based），仅保留最近 keep 个

    用途：训练中途分叉实验（换配方/换增强/微调），不必从头重跑。
    """
    wdir = run_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    save_weights(model, wdir / f"epoch{epoch + 1:03d}.safetensors")
    for old in sorted(wdir.glob("epoch*.safetensors"))[:-max(keep, 1)]:
        old.unlink()

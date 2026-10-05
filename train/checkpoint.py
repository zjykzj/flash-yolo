"""训练 checkpoint 落盘与恢复

- resume.pt：完整训练状态快照（模型/EMA/优化器/缩放器/epoch/best/rng），
  仅内部断点续训用（pickle，不出 runs/ 目录）
- best/last.safetensors：EMA 权重交付物（纯张量，可审计）
"""

import numpy as np
import torch

from model.weights import save_weights

__all__ = ["save_resume", "load_resume", "save_best_last", "capture_rng"]


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
        "cfg": cfg,
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


def save_best_last(run_dir, model, is_best):
    """EMA 权重 -> weights/last.safetensors（每 epoch）+ weights/best.safetensors（新高时）"""
    wdir = run_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    save_weights(model, wdir / "last.safetensors")
    if is_best:
        save_weights(model, wdir / "best.safetensors")

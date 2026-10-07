"""路径工具：runs/ 结果目录递增 + 运行元数据

ultralytics 惯例：首次运行 runs/predict/predict，之后 predict2、predict3...，
多次运行互不覆盖、产物可追溯。
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

__all__ = ["increment_path", "env_meta", "write_run_meta"]


def increment_path(path, mkdir=True):
    """返回递增后的目录路径（不存在则原样返回，已存在则后缀 +1）

    Args:
        path: 基准目录（如 runs/predict/predict）
        mkdir: 是否创建目录
    """
    path = Path(path)
    if not path.exists():
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path
    for n in range(2, 9999):
        candidate = path.with_name(f"{path.name}{n}")
        if not candidate.exists():
            if mkdir:
                candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"too many run directories: {path}")


def _git_state():
    """git 提交/分支/是否脏（不在仓库内或无 git 时返回空 dict）"""
    out = {}
    try:
        root = Path(__file__).resolve().parent.parent
        for key, cmd in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
            out[key] = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout
        out["dirty"] = bool(dirty.strip())
    except Exception:  # noqa: BLE001 —— 元数据不应影响训练
        return {}
    return out


def env_meta():
    """运行环境元数据（复盘用）：git / python / torch / cuda / 主机"""
    from config import __version__

    meta = {
        "flash_yolo": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": _git_state(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["gpu_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        pass
    return meta


def write_run_meta(run_dir, extra=None):
    """run_dir/meta.json：环境 + 配置 + 数据规模 + 命令行（复盘不必翻日志）"""
    meta = env_meta()
    meta.update(extra or {})
    meta["argv"] = sys.argv
    path = Path(run_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path

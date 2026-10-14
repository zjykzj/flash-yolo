"""路径工具：runs/ 结果目录递增 + 运行元数据

ultralytics 惯例：首次运行 runs/predict/predict，之后 predict2、predict3...，
多次运行互不覆盖、产物可追溯。
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

__all__ = ["increment_path", "env_meta", "write_run_meta", "git_commit", "git_describe"]


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


def git_commit(root=None):
    """仓库 HEAD 的完整 commit sha；不在仓库内 / 无 git / 任何异常 -> None（best-effort）

    与 meta.json 的 git.commit 同源（_git_state 复用本函数）：读不到就返回 None 而不是空串，
    由调用方决定"不写这个键"——权重 metadata 的出处信息不该用空值冒充有值。
    """
    try:
        root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 —— 元数据不应影响训练
        return None


def git_describe(root=None):
    """git describe 的原始输出（如 `v0.5.0-17-g984d198-dirty`）；读不到 -> None

    格式 = `<最近可达 tag>-<距该 tag 的提交数>-g<缩写 sha>[-dirty]`：一个字符串说清"离最近发布
    多远 + 精确落点 + 工作区是否脏"，正好补上"version 是发布线、commit 是精确坐标"之间的落差。
    **不能带 commit-ish 参数**（--dirty 只对工作区生效，带参数直接报错）；无 tag 时 --always
    退化成短 sha，无 git 时返回 None（与 git_commit 同一条降级路径）。
    """
    try:
        root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 —— 元数据不应影响训练
        return None


def _git_state():
    """git 提交/分支/是否脏（不在仓库内或无 git 时返回空 dict）"""
    out = {}
    try:
        root = Path(__file__).resolve().parent.parent
        out["commit"] = git_commit() or ""  # 历史口径：读不到写空串（与 git 失败时的 stdout 一致）
        out["branch"] = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                                       capture_output=True, text=True, timeout=5).stdout.strip()
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

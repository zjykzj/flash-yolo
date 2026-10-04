"""路径工具：runs/ 结果目录递增

ultralytics 惯例：首次运行 runs/predict/predict，之后 predict2、predict3...，
多次运行互不覆盖、产物可追溯。
"""

from pathlib import Path

__all__ = ["increment_path"]


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

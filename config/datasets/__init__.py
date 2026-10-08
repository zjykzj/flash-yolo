"""数据集元数据注册表：类别名（config/datasets/<name>.yaml）

与 data/ 的分工：data/ 是数据读取/增强代码，这里只放静态元数据（类别名等）；
yaml 是数据、本模块是唯一读取入口。新增数据集加一个 `config/datasets/<name>.yaml` 即可。
"""

from functools import lru_cache
from pathlib import Path

import yaml

__all__ = ["DATASETS_DIR", "load_names"]

DATASETS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _names_tuple(name: str) -> tuple[str, ...]:
    """<name>.yaml（或显式 .yaml 路径）-> names 元组（缓存；不可变对象防跨调用污染）"""
    p = Path(name)
    if p.suffix in {".yaml", ".yml"}:
        if not p.is_file():
            raise FileNotFoundError(f"dataset file not found: {p}")
    else:
        p = DATASETS_DIR / f"{name}.yaml"
        if not p.is_file():
            available = sorted(x.stem for x in DATASETS_DIR.glob("*.yaml")) if DATASETS_DIR.is_dir() else []
            raise ValueError(f"unknown dataset {name!r}; available: {available} (or pass a .yaml path)")
    entry = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    names = entry.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{p.name}: 'names' must be a non-empty list of strings (index = class id)")
    return tuple(names)


def load_names(name="coco"):
    """数据集名（或 .yaml 路径）-> {class_id: name}；每次返回新 dict，调用方可安全修改"""
    return dict(enumerate(_names_tuple(name)))

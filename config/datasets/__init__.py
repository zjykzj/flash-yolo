"""数据集元数据注册表：描述符与类别名（config/datasets/[local/]<name>.yaml）

与 data/ 的分工：data/ 是数据读取/增强代码，这里只放静态元数据（描述符与类别名）；
描述符 schema 与解析校验在 spec.py，本模块是唯一的读取入口。
新增数据集：加一个 `config/datasets/<name>.yaml`（模板见 coco.yaml）；
本机专用（含机器路径）的描述符放 `config/datasets/local/<name>.yaml`（gitignored，名字寻址优先）。
"""

from functools import lru_cache

from config.datasets.spec import (DATASETS_DIR, LOCAL_DATASETS_DIR, DatasetSpec, RoleSpec,
                                  load_dataset, normalize_names, parse_descriptor, read_names_file,
                                  resolve_dataset_path)

__all__ = ["DATASETS_DIR", "LOCAL_DATASETS_DIR", "DatasetSpec", "RoleSpec", "load_dataset",
           "load_names", "normalize_names", "parse_descriptor", "resolve_dataset_path"]


@lru_cache(maxsize=None)
def _names_tuple(name: str) -> tuple[str, ...]:
    """数据集名（或 .yaml 路径）-> names 元组（缓存；不可变对象防跨调用污染）"""
    names = read_names_file(resolve_dataset_path(name))
    return tuple(names[i] for i in range(len(names)))


def load_names(name="coco"):
    """数据集名（或 .yaml 路径）-> {class_id: name}；每次返回新 dict，调用方可安全修改"""
    return dict(enumerate(_names_tuple(name)))

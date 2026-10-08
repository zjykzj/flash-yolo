"""数据集描述符：ultralytics 式 dataset yaml 的解析与校验（yaml 是数据，本模块是唯一读取入口）

schema（完整模板见 config/datasets/coco.yaml）：

    format: coco | yolo       # 必填：标注读取方式
    path: ""                  # 数据集根；相对路径以本 yaml 所在目录为基准；空 = 只用绝对路径
    names: [...] | {0: ...}   # 必填：index = 类别 id，长度 = nc（列表或映射等价）
    train:                    # 角色键：train / val / test（test 接受但当前不消费）
      images: images/train2017                   # coco: 图片目录；yolo: 目录或 .txt 图片清单
      ann: annotations/instances_train2017.json  # coco 必填；yolo 禁止
      labels: train/labels                       # yolo 可选；缺省把 images 路径里的 images 段换成 labels

路径解析只有这一处：data/ 的加载器一律接收本模块解出的绝对路径，不再自己拼目录。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

__all__ = ["FORMATS", "ROLES", "DATASETS_DIR", "LOCAL_DATASETS_DIR", "RoleSpec", "DatasetSpec",
           "normalize_names", "read_names_file", "resolve_dataset_path", "parse_descriptor",
           "load_dataset"]

FORMATS = ("coco", "yolo")
ROLES = ("train", "val", "test")

DATASETS_DIR = Path(__file__).resolve().parent
LOCAL_DATASETS_DIR = DATASETS_DIR / "local"  # 本机描述符（gitignored）；名字寻址优先于包内模板

_ROLE_KEYS = {"images", "ann", "labels"}


@dataclass(frozen=True)
class RoleSpec:
    """一个角色（train/val/test）的已解析路径（全部绝对）"""

    role: str
    images: Path  # 图片目录；yolo 下也可是 .txt 图片清单（清单内相对项相对清单文件解析）
    ann: Path | None = None  # coco 专用：标注 json
    labels: Path | None = None  # yolo 专用：标签目录


@dataclass(frozen=True)
class DatasetSpec:
    """数据集描述符（解析 + 校验后的结果）"""

    name: str  # 描述符名（yaml 文件名去后缀）或显式路径的 stem
    source: Path  # 实际读取的 yaml 路径（绝对）
    format: str  # coco | yolo
    root: Path | None  # `path:`（空 = None）
    names: dict[int, str]  # 0..nc-1 -> 类别名
    roles: dict[str, RoleSpec]

    @property
    def nc(self):
        return len(self.names)

    def role(self, role="val"):
        """取角色（缺省 val）；缺失时报错并列出可用角色"""
        if role not in self.roles:
            raise ValueError(f"dataset {self.name!r} has no {role!r} role; "
                             f"available: {sorted(self.roles)} (defined in {self.source})")
        return self.roles[role]


def normalize_names(value, where="names"):
    """names 字段（列表或 {index: name} 映射）-> {int: str}；键必须恰为 0..n-1"""
    if isinstance(value, dict):
        try:
            items = {int(k): v for k, v in value.items()}
        except (TypeError, ValueError):
            raise ValueError(f"{where}: 'names' keys must be integers (class ids)") from None
    elif isinstance(value, list):
        items = dict(enumerate(value))
    else:
        raise ValueError(f"{where}: 'names' must be a non-empty list of strings (index = class id) "
                         f"or a {{index: name}} mapping")
    if not items or not all(isinstance(v, str) and v for v in items.values()):
        raise ValueError(f"{where}: 'names' must be a non-empty list of strings (index = class id)")
    if sorted(items) != list(range(len(items))):
        raise ValueError(f"{where}: 'names' indices must be exactly 0..{len(items) - 1}, "
                         f"got {sorted(items)}")
    return items


def read_names_file(path) -> dict[int, str]:
    """yaml -> names 字段（只做 names 校验；描述符其余部分不查——load_names 的轻量入口）"""
    path = Path(path)
    entry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "names" not in entry:
        raise ValueError(f"{path.name}: missing required key 'names'")
    return normalize_names(entry["names"], path.name)


def resolve_dataset_path(name_or_path) -> Path:
    """描述符名或 .yaml 路径 -> yaml 文件路径

    - 显式 .yaml/.yml 路径：原样（缺失 -> FileNotFoundError）
    - 名字：config/datasets/local/<name>.yaml -> config/datasets/<name>.yaml（本机覆盖优先）
    - 目录：报迁移错误（--data 已不收数据集目录，改收描述符）
    """
    p = Path(name_or_path)
    if p.is_dir():
        raise ValueError(
            f"--data takes a dataset descriptor yaml (a name or a .yaml path), not a directory: {p}; "
            "see config/datasets/coco.yaml for the schema (set `path:` there to the dataset root)")
    if p.suffix in {".yaml", ".yml"}:
        if not p.is_file():
            raise FileNotFoundError(f"dataset file not found: {p}")
        return p
    for d in (LOCAL_DATASETS_DIR, DATASETS_DIR):
        candidate = d / f"{p.name}.yaml"
        if candidate.is_file():
            return candidate
    available = sorted({x.stem for d in (LOCAL_DATASETS_DIR, DATASETS_DIR) if d.is_dir()
                        for x in d.glob("*.yaml")})
    raise ValueError(f"unknown dataset {p.name!r}; available: {available} (or pass a .yaml path)")


def _resolve_role_path(value, root, source, role, key):
    """角色内路径：绝对原样；相对 -> root / value（root 为空时报 actionable 错误）"""
    p = Path(value)
    if p.is_absolute():
        return p
    if root is None:
        raise ValueError(f"{source.name}: role {role!r} key {key!r} is relative ({value!r}) but `path:` "
                         f"is empty — set `path:` to the dataset root, or use absolute paths")
    return root / p


def _mirror_labels(images: Path, source, role):
    """images 路径中最后一个 `images` 段换成 `labels`（train/images -> train/labels）"""
    parts = list(images.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"{source.name}: role {role!r}: cannot infer the labels dir from {images} "
                     f"(no 'images' path segment) — set `labels:` explicitly")


def _parse_role(role, block, root, source, fmt):
    if not isinstance(block, dict):
        raise ValueError(f"{source.name}: role {role!r} must be a mapping with an 'images' key")
    for k in sorted(set(block) - _ROLE_KEYS):
        logger.warning(f"{source.name} role {role!r} 未识别字段（忽略）: {k}")
    if "images" not in block:
        raise ValueError(f"{source.name}: role {role!r} is missing 'images'")
    images = _resolve_role_path(block["images"], root, source, role, "images")

    ann = labels = None
    if fmt == "coco":
        if "labels" in block:
            raise ValueError(f"{source.name}: role {role!r}: 'labels' is only valid for format yolo")
        if "ann" not in block:
            raise ValueError(f"{source.name}: role {role!r}: format coco requires 'ann' "
                             f"(the annotation json path, relative to `path:`)")
        ann = _resolve_role_path(block["ann"], root, source, role, "ann")
    else:
        if "ann" in block:
            raise ValueError(f"{source.name}: role {role!r}: 'ann' is only valid for format coco")
        labels = (_resolve_role_path(block["labels"], root, source, role, "labels")
                  if "labels" in block else _mirror_labels(images, source, role))
    return RoleSpec(role=role, images=images, ann=ann, labels=labels)


def parse_descriptor(path, name=None) -> DatasetSpec:
    """描述符 yaml -> DatasetSpec（完整校验；解析出的路径全部为绝对路径）"""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for k in sorted(set(raw) - {"format", "path", "names", *ROLES}):
        if isinstance(raw[k], dict):
            raise ValueError(f"{path.name}: unknown role {k!r}; roles must be one of {list(ROLES)}")
        logger.warning(f"{path.name} 未识别字段（忽略）: {k}")

    fmt = raw.get("format")
    if fmt not in FORMATS:
        raise ValueError(f"{path.name}: 'format' must be one of {list(FORMATS)}, got {fmt!r}")
    if "names" not in raw:
        raise ValueError(f"{path.name}: missing required key 'names'")
    names = normalize_names(raw["names"], path.name)

    root = raw.get("path") or None
    if root is not None:
        root = Path(root)
        root = root if root.is_absolute() else (path.parent / root)
        root = root.resolve()

    roles = {}
    for role in ROLES:
        if role in raw:
            roles[role] = _parse_role(role, raw[role], root, path, fmt)
    if not roles:
        raise ValueError(f"{path.name}: no roles defined; add at least a `train:` block")

    return DatasetSpec(name=name or path.stem, source=path, format=fmt, root=root, names=names, roles=roles)


def load_dataset(name_or_path) -> DatasetSpec:
    """描述符名（config/datasets/[local/]<name>.yaml）或 .yaml 路径 -> DatasetSpec"""
    path = resolve_dataset_path(name_or_path)
    name = Path(name_or_path).stem if Path(name_or_path).suffix in {".yaml", ".yml"} else Path(name_or_path).name
    return parse_descriptor(path, name)

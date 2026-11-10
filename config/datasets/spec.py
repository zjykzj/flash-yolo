"""数据集描述符：ultralytics 式 dataset yaml 的解析与校验（yaml 是数据，本模块是唯一读取入口）

schema（完整模板见 config/datasets/coco.yaml）：

    format: coco | yolo       # 必填：标注读取方式
    path: ""                  # 数据集根；相对路径以本 yaml 所在目录为基准；空 = 只用绝对路径
    eval_style: coco | lvis   # 可选：评估协议（默认 coco；lvis 走 faster_coco_eval lvis_style）
    names: [...] | {0: ...}   # 必填：index = 类别 id，长度 = nc（列表或映射等价）
    vocab_synonyms: {..: ..}  # 可选：数据集类名 -> 开放词汇模型词汇名（评估映射用；见 coco.yaml）
    train:                    # 角色键：train / val / test（test 接受但当前不消费）
      images: images/train2017                   # coco: 图片目录；yolo: 目录或 .txt 图片清单
      ann: annotations/instances_train2017.json  # coco 必填；yolo 禁止
      labels: train/labels                       # yolo 可选；缺省把 images 路径里的 images 段换成 labels

路径解析只有这一处：data/ 的加载器一律接收本模块解出的绝对路径，不再自己拼目录。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

__all__ = ["FORMATS", "ROLES", "DATASETS_DIR", "LOCAL_DATASETS_DIR", "RoleSpec", "DatasetSpec",
           "normalize_names", "read_names_file", "resolve_dataset_path", "parse_descriptor",
           "load_dataset"]

FORMATS = ("coco", "yolo")
ROLES = ("train", "val", "test")
# 姿态任务的描述符字段：kpt_shape 是结构参数，其余是数据/可视化参数（COCO-pose 口径）。
# 只有 pose 数据集需要；普通检测/分割描述符不写这些键，行为与历史一致。
POSE_KEYS = ("kpt_shape", "flip_idx", "kpt_names", "skeleton")

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
    vocab_synonyms: dict[str, str] = field(default_factory=dict)  # 数据集类名 -> 词汇名（可选）
    eval_style: str = "coco"  # 评估协议：coco（pycocotools）| lvis（faster_coco_eval lvis_style）
    # 姿态（可选）：kpt_shape = (K, D)；flip_idx 是水平翻转后的关键点重排；kpt_names/skeleton
    # 只用于可视化（skeleton 为 0-based 点对）。检测/分割描述符上述字段为空。
    kpt_shape: tuple[int, int] | None = None
    flip_idx: tuple[int, ...] | None = None
    kpt_names: dict[int, list[str]] = field(default_factory=dict)
    skeleton: tuple[tuple[int, int], ...] = field(default_factory=tuple)

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
    for k in sorted(set(raw) - {"format", "path", "eval_style", "names", "vocab_synonyms",
                                *ROLES, *POSE_KEYS}):
        if isinstance(raw[k], dict):
            raise ValueError(f"{path.name}: unknown role {k!r}; roles must be one of {list(ROLES)}")
        logger.warning(f"{path.name} 未识别字段（忽略）: {k}")

    fmt = raw.get("format")
    if fmt not in FORMATS:
        raise ValueError(f"{path.name}: 'format' must be one of {list(FORMATS)}, got {fmt!r}")
    eval_style = raw.get("eval_style", "coco")
    if eval_style not in ("coco", "lvis"):
        raise ValueError(f"{path.name}: 'eval_style' must be 'coco' or 'lvis', got {eval_style!r}")
    if eval_style == "lvis" and fmt != "coco":
        raise ValueError(f"{path.name}: eval_style 'lvis' requires format 'coco' "
                         f"(YOLO labels carry no LVIS neg/not_exhaustive fields)")
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

    vocab_synonyms = raw.get("vocab_synonyms") or {}
    if not isinstance(vocab_synonyms, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k and v for k, v in vocab_synonyms.items()):
        raise ValueError(f"{path.name}: 'vocab_synonyms' must be a non-empty {{dataset name: vocab name}} mapping")

    kpt_shape, flip_idx, kpt_names, skeleton = _parse_pose(raw, path)
    return DatasetSpec(name=name or path.stem, source=path, format=fmt, root=root, names=names, roles=roles,
                       vocab_synonyms=vocab_synonyms, eval_style=eval_style,
                       kpt_shape=kpt_shape, flip_idx=flip_idx, kpt_names=kpt_names, skeleton=skeleton)


def _parse_pose(raw, path):
    """解析姿态描述符的可选字段（缺失 -> None/空；校验失败点名文件与字段）"""
    shape = raw.get("kpt_shape")
    if shape is None:
        return None, None, {}, ()
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or not all(isinstance(v, int) and v > 0 for v in shape) or shape[1] not in (2, 3)):
        raise ValueError(f"{path.name}: 'kpt_shape' must be [K, 2] or [K, 3], got {shape!r}")
    shape = tuple(shape)
    k = shape[0]

    flip = raw.get("flip_idx")
    if flip is not None:
        if not isinstance(flip, (list, tuple)) or len(flip) != k \
                or not all(isinstance(v, int) and 0 <= v < k for v in flip):
            raise ValueError(f"{path.name}: 'flip_idx' must be {k} indices in 0..{k - 1}, got {flip!r}")
        flip = tuple(flip)

    names = raw.get("kpt_names") or {}
    if not isinstance(names, dict):
        raise ValueError(f"{path.name}: 'kpt_names' must be a {{class_id: [names...]}} mapping")
    try:
        names = {int(c): [str(n) for n in ns] for c, ns in names.items()}
    except (TypeError, ValueError):
        raise ValueError(f"{path.name}: 'kpt_names' keys must be class ids and values lists of strings") from None
    for c, ns in names.items():
        if len(ns) != k:
            raise ValueError(f"{path.name}: 'kpt_names[{c}]' has {len(ns)} names, expected {k} (kpt_shape[0])")

    skel = raw.get("skeleton") or ()
    if not isinstance(skel, (list, tuple)):
        raise ValueError(f"{path.name}: 'skeleton' must be a list of [i, j] point pairs")
    pairs = []
    for pair in skel:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, int) and 0 <= v < k for v in pair)):
            raise ValueError(f"{path.name}: 'skeleton' entries must be [i, j] with 0 <= i,j < {k}, got {pair!r}")
        pairs.append(tuple(pair))
    return shape, flip, names, tuple(pairs)


def load_dataset(name_or_path) -> DatasetSpec:
    """描述符名（config/datasets/[local/]<name>.yaml）或 .yaml 路径 -> DatasetSpec"""
    path = resolve_dataset_path(name_or_path)
    name = Path(name_or_path).stem if Path(name_or_path).suffix in {".yaml", ".yml"} else Path(name_or_path).name
    return parse_descriptor(path, name)

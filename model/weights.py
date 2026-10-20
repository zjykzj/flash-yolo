"""权重读写：safetensors <-> 模型（strict load，成功即证明拓扑一致）

metadata（可选）：随权重一起走的模型参数，写在 safetensors header 的 metadata 字段（纯 JSON
字符串，不碰张量）——训练保存的 best/last/epochN 与官方权重转换产物会带
`{arch, scale, nc, imgsz[, names, anchors]}` + 出处 `{version, repo[, commit, describe]}`；旧权重 /
`.onnx` / `.pt` 没有则安全降级为空 dict。
读取侧三级回退：CLI 显式 > metadata > 文件名/默认值（metadata 永远是可选的）。
"""

import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from config.inference import IMGSZ

__all__ = ["load_weights", "load_weights_finetune", "save_weights", "scale_from_weights",
           "arch_from_weights", "arch_has_scale", "resolve_arch_scale", "resolve_imgsz", "load_meta",
           "encode_meta", "encode_anchors", "apply_meta", "provenance_meta"]

logger = logging.getLogger(__name__)

# 官方档位命名：yolo26n/s/m/l/x（scripts/download_weights.py 与 scripts/export.py 都沿用）
_SCALE_RE = re.compile(r"yolo26([nsmlx])($|[-_.])", re.I)
# 架构命名：yolov3-tiny（官方 darknet 权重命名）/ yolov3-tiny-silu（算子替换版）/
# flash-yolo（本项目自有架构）/ flash-yolo-seg（分割变体）。长名在前，否则 "yolov3-tiny-silu"
# 会被 "yolov3-tiny"、"flash-yolo-seg" 会被 "flash-yolo" 前缀误吸
_ARCH_RE = re.compile(r"(yolov3-tiny-silu|yolov3-tiny|flash-yolo-seg|flash-yolo)($|[-_.])", re.I)
# 任务后缀：yolo26n-seg（官方分割权重命名）。先于档位判定：yolo26 系里带 -seg 归 yolo26-seg
_TASK_SUFFIX_RE = re.compile(r"yolo26[nsmlx](?:[-_.][a-z0-9]+)*[-_.]seg($|[-_.])", re.I)
# 带档位维度（scale 参与解析）的架构；其余架构档位固定（v3: tiny / flash-yolo: flash）
_SCALED_ARCHS = ("yolo26", "yolo26-seg")

_META_STR_KEYS = ("arch", "scale", "version", "repo", "commit", "describe")  # header 直读字符串
_META_INT_KEYS = ("nc", "imgsz")        # int() 还原
_META_JSON_KEYS = ("names", "anchors")  # JSON 解码（写入侧统一 json.dumps）


def provenance_meta():
    """权重出处：version / repo（项目身份常量）+ commit / describe（best-effort 读本地仓库）

    回答"这份权重是哪版代码产的"——trainer 保存与两个转换器都写。version 是**声明版本（发布线）**、
    commit 是**精确坐标**，两者不要求对应（版本号只在发布时 bump，两次发布之间的提交都顶着旧版本
    号）；describe（`v0.5.0-17-g984d198-dirty`）把这段落差显式写出来，人读到即知"不是 tag 那次提交"。
    读不到 commit/describe 时**不写对应键**：与转换器"读不到就不写"的既有约定一致，缺键 = 未知，
    不用空串冒充有值（Docker 镜像里 `.git` 被 .dockerignore 排除，正是这条路径）。
    """
    from config import __repo__, __version__
    from utils.paths import git_commit, git_describe

    meta = {"version": __version__, "repo": __repo__}
    commit, describe = git_commit(), git_describe()
    if commit:
        meta["commit"] = commit
    if describe:
        meta["describe"] = describe
    return meta


def encode_meta(metadata):
    """metadata dict -> safetensors header（值必须为 str；list/dict 走 JSON）"""
    return {str(k): (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for k, v in (metadata or {}).items()}


def encode_anchors(anchors):
    """(nl, 3, 2) 锚点（tensor / 嵌套序列）-> int/float 嵌套 list（darknet 原值为整型；写 metadata 用）"""
    return [[[int(v) if float(v).is_integer() else float(v) for v in slot] for slot in level]
            for level in torch.as_tensor(anchors).tolist()]


def load_meta(path):
    """读 safetensors header 里的 metadata -> dict（无 / 不可读 -> {}）

    已知键做类型还原（nc/imgsz -> int；names/anchors -> JSON 解码），未知键忽略；文件不存在、
    非 safetensors（.onnx/.pt）、单个坏键都安全降级——metadata 永远是可选的，不阻断权重加载。
    """
    p = Path(path)
    if p.suffix.lower() != ".safetensors" or not p.exists():
        return {}
    try:
        with safe_open(str(p), framework="pt") as f:
            raw = f.metadata() or {}
    except Exception:  # 内容损坏等：权重本体加载（load_file）时会再报错，这里按"无 metadata"处理
        return {}
    meta = {}
    for k in _META_STR_KEYS:
        if raw.get(k):
            meta[k] = raw[k]
    for k in _META_INT_KEYS:
        if raw.get(k) is not None:
            try:
                meta[k] = int(raw[k])
            except ValueError:
                logger.warning(f"weights metadata: bad int for {k!r} ({raw[k]!r}) — ignored")
    for k in _META_JSON_KEYS:
        if raw.get(k) is not None:
            try:
                meta[k] = json.loads(raw[k])
            except json.JSONDecodeError:
                logger.warning(f"weights metadata: bad JSON for {k!r} — ignored")
    return meta


def scale_from_weights(path):
    """从权重文件名推模型档位：`yolo26s.safetensors` / `yolo26s.onnx` -> "s"；认不出返回 None

    推理脚本据此不必强制 `--scale`：官方命名自带档位。认不出（如 `best.safetensors`）时再看
    权重 metadata，仍无则由调用方报错要求显式指定——比闷头按 n 档建模型、再到 strict load
    处报尺寸不匹配要好。
    """
    m = _SCALE_RE.search(Path(path).stem)
    return m.group(1).lower() if m else None


def arch_from_weights(path):
    """从权重文件名推架构：`yolov3-tiny.safetensors` -> "yolov3-tiny"；认不出返回 None

    yolo26 系带任务后缀时归到对应任务架构：`yolo26n-seg.safetensors` -> "yolo26-seg"
    （先判后缀，再看其它架构名——档位正则不区分任务，只有后缀能区分）。
    """
    stem = Path(path).stem
    if _TASK_SUFFIX_RE.search(stem):
        return "yolo26-seg"
    m = _ARCH_RE.search(stem)
    return m.group(1).lower() if m else None


def arch_has_scale(arch):
    """该架构是否有档位维度（yolo26 / yolo26-seg 有；v3 系与 flash-yolo 档位固定）"""
    return arch in _SCALED_ARCHS


def resolve_arch_scale(weights, model=None, scale=None):
    """(arch, scale) 解析：显式参数 > 权重 metadata > 文件名推断；都认不出返回 (None, None)

    metadata 由权重生产者写入（训练保存 / 官方转换），比可被随意改名的文件更可信；旧权重没有
    metadata 时完全退回文件名推断（行为与历史一致）。带档位的架构（`_SCALED_ARCHS`，yolo26 与
    yolo26-seg）：`yolo26s.safetensors` -> ("yolo26", "s")、`yolo26n-seg.safetensors` ->
    ("yolo26-seg", "n")（档位仍认不出时 scale=None，调用方报错）；v3 系：
    `yolov3-tiny.safetensors` -> ("yolov3-tiny", None)（档位由架构固定）。
    """
    meta = load_meta(weights)
    if model is None:
        # 文件名兜底时保留既有约定：文件名里的架构优先于误传的 --scale
        model = meta.get("arch") or arch_from_weights(weights) or (
            "yolo26" if (scale or meta.get("scale") or scale_from_weights(weights)) else None)
    if model is None:
        return None, None
    if model in _SCALED_ARCHS:
        return model, scale or meta.get("scale") or scale_from_weights(weights)
    return model, None  # 其余架构：档位固定（--scale 不适用）


def resolve_imgsz(weights, imgsz=None):
    """模型输入尺寸解析：显式参数 > 权重 metadata > 640（config.inference.IMGSZ）

    `.onnx` 没有 metadata——输入尺寸以图为准（引擎读图内 shape），此处返回显式值或默认值。
    """
    if imgsz is not None:
        return int(imgsz)
    return int(load_meta(weights).get("imgsz") or IMGSZ)


def apply_meta(model, meta):
    """把 metadata 中"结构级"参数应用到已构建的模型，返回实际应用的键名列表

    目前只有 v3 anchors（darknet 固定像素值、随权重走）：自定义重聚类训练后，即使手头 yaml
    是原版，用带 metadata 的权重也能还原训练时的锚点。yolo26 等无锚架构 / 无该项时不做任何事；
    形状与头不符则明确报错（权重与 yaml 架构不匹配，strict load 之外的又一道闸门）。
    """
    applied = []
    anchors = meta.get("anchors")
    head = model.model[-1]
    if anchors and hasattr(head, "anchors"):
        t = torch.as_tensor(anchors, dtype=head.anchors.dtype)
        if t.shape != head.anchors.shape:
            raise ValueError(f"weights metadata anchors shape {tuple(t.shape)} != model "
                             f"{tuple(head.anchors.shape)} — incompatible checkpoint")
        head.anchors = t
        applied.append("anchors")
    return applied


def load_weights(model, path, strict=True):
    """加载转换后的纯权重（由 scripts/convert_weights.py / 训练保存生成）

    Args:
        model: YOLO26 模型
        path: .safetensors 路径
        strict: True 时任何 key 不匹配都会报错（官方语义：加载成功即拓扑一致）
    """
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if not strict and (missing or unexpected):
        logger.warning(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            logger.warning(f"missing sample: {missing[:5]}")
        if unexpected:
            logger.warning(f"unexpected sample: {unexpected[:5]}")
    return missing, unexpected


def load_weights_finetune(model, path):
    """微调初始化：key 必须一一对应，只跳过任务头内 shape 变化的张量（如 O365 365 类 -> COCO 80 类）

    与 `load_weights(strict=True)` 的区别只有一条：允许同 key 的 shape 变化并按交集加载。
    - key 缺失/多余 -> 报错（架构改名/换错权重不会被静默吞掉）；
    - shape 变化只允许发生在任务头 `model.model[-1]` 内（分类分支/语义头随 nc 变）；
      头外任何 shape 不匹配 -> 报错（不同档位/架构的权重不会被部分加载）；
    - 加载后返回 (loaded, skipped)：skipped 为 [(key, ckpt_shape, model_shape), ...]，
      调用方据此打印“跳过了哪些类别相关层”。

    官方 O365 预训练权重是 365 类；COCO 训练建 80 类头时用它初始化，类别头随机初始化、
    backbone/neck/proto/cv2 等逐位加载——与 ultralytics 的 intersect 加载语义一致。
    """
    sd = load_file(path)
    target = model.state_dict()
    missing = sorted(set(target) - set(sd))
    unexpected = sorted(set(sd) - set(target))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint/model key mismatch: missing {len(missing)} {missing[:5]}, "
            f"unexpected {len(unexpected)} {unexpected[:5]} — use the matching architecture"
        )
    names = {id(m): n for n, m in model.named_modules()}
    head_prefix = names[id(model.model[-1])] + "."
    loaded, skipped = {}, []
    for k, v in sd.items():
        if tuple(v.shape) == tuple(target[k].shape):
            loaded[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(target[k].shape)))
    bad = [k for k, _, _ in skipped if not k.startswith(head_prefix)]
    if bad:
        raise ValueError(
            f"shape mismatch outside the task head ({head_prefix}): {bad[:5]} — "
            "check the model scale/architecture; only class-dependent head tensors may be re-initialized"
        )
    if not loaded:
        raise ValueError("no tensors loaded from checkpoint")
    model.load_state_dict(loaded, strict=False)
    return loaded, skipped


def save_weights(model, path, metadata=None):
    """模型 state_dict -> safetensors（纯权重交付物，配合 load_weights strict 回载）

    metadata（可选）：随权重走的模型参数（arch/scale/nc/imgsz/names/anchors），写入 header 的
    metadata 字段——只加 JSON 头、不碰张量（逐位对拍/转换器对账全部不受影响）；缺省 None 与
    历史文件完全等价。保存前统一转连续：channels_last 训练时参数为 NHWC 步长，safetensors
    只接受连续张量（仅规范化内存布局，数值不变；回载时按目标参数布局 copy，不受影响）。
    """
    header = encode_meta(metadata)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path,
              metadata=header or None)

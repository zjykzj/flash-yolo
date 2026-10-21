"""数据集工厂：描述符（DatasetSpec）-> 具体加载器（train / eval 两侧的唯一构造入口）

格式分派在这里；COCO 侧额外做描述符 `names` 与 json categories 的硬校验——
nc 由描述符 names 决定（模型据此构建），json 的类别表必须与之一致；不对账的话
要到训练中途才会以越界 / KeyError 的形式暴露，这里在构造数据集时就报出来。
"""

from data.coco import CocoDataset, CocoTrainDataset
from data.yolo import YoloDataset, YoloTrainDataset

__all__ = ["build_train_dataset", "build_eval_dataset"]


def _check_paths(spec, role):
    """角色路径必须先于加载器构建存在性检查（报错点名描述符与角色）"""
    r = spec.role(role)
    if not r.images.exists():
        raise FileNotFoundError(f"{spec.name}: role {role!r} images not found: {r.images}")
    if spec.format == "coco":
        if not r.ann.is_file():
            raise FileNotFoundError(f"{spec.name}: role {role!r} annotation json not found: {r.ann}")
    elif not r.labels.exists():
        raise FileNotFoundError(f"{spec.name}: role {role!r} labels dir not found: {r.labels}")


def _check_names(spec, ds):
    """描述符 names 与 COCO json categories 对账（报首个不一致下标）"""
    if len(ds.names) != spec.nc:
        raise ValueError(f"{spec.name}: descriptor defines {spec.nc} classes but the annotation file has "
                         f"{len(ds.names)} — `names` must match the json categories")
    for i in range(spec.nc):
        if ds.names[i] != spec.names[i]:
            raise ValueError(f"{spec.name}: descriptor names[{i}]={spec.names[i]!r} but the annotation file "
                             f"has {ds.names[i]!r} — names must match the json categories in sorted-id order")


def build_train_dataset(cfg, spec, role="train", augment=True, limit=0, progress=False, progress_file=None,
                        task="detect", single_cls=False):
    """训练集（按 spec.format 分派）；cfg 供增强/尺寸使用（duck-typing）

    task="segment"：COCO json（多边形 / RLE 解码 / 损坏时 bbox 回退）与 YOLO txt
    （多边形行，bbox 行矩形回退）
    都支持；两种格式最终都走同一条 (imgs, targets, masks) 训练契约。
    """
    _check_paths(spec, role)
    r = spec.role(role)
    if spec.format == "coco":
        ds = CocoTrainDataset(cfg, r.ann, r.images, label=role, augment=augment, limit=limit,
                              progress=progress, progress_file=progress_file,
                              segment=(task == "segment"), single_cls=single_cls)
        if not single_cls:  # 单类时描述符 nc 与数据类别数故意不同（模型头为 1 类）
            _check_names(spec, ds)
        return ds
    return YoloTrainDataset(cfg, r.images, r.labels, spec.names, label=role, augment=augment,
                            limit=limit, progress=progress, progress_file=progress_file,
                            segment=(task == "segment"), single_cls=single_cls)


def build_eval_dataset(spec, role="val", progress=False, progress_file=None, task="detect",
                       single_cls=False):
    """评估集（按 spec.format 分派）；--limit 由调用方对评估循环切片（口径与既有 eval 一致）

    task="segment" 时 YOLO 侧额外保留多边形（seg_val / gt_dict 的 pycocotools segmentation）。
    """
    _check_paths(spec, role)
    r = spec.role(role)
    if spec.format == "coco":
        ds = CocoDataset(r.ann, r.images, label=role, progress=progress, progress_file=progress_file,
                         single_cls=single_cls)
        if not single_cls:
            _check_names(spec, ds)
        return ds
    return YoloDataset(r.images, r.labels, spec.names, label=role, progress=progress,
                       progress_file=progress_file, segment=(task == "segment"), single_cls=single_cls)

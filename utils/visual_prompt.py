"""视觉 prompt：JSON 解析 + 掩码光栅化（对齐官方 LoadVisualPrompt）

视觉 prompt = 一张 refer 图 + 若干 bbox（每类可有多个）。掩码在 **模型输入坐标** 的
1/8 分辨率（P3 网格）上生成：bbox 先按 letterbox 的 ratio/pad 变换到输入坐标，再乘 1/8，
最后以 `x1 <= x < x2`、`y1 <= y < y2` 光栅化；同类别多框按 OR 合并。SAVPE 用这些掩码
从 refer 图的三尺度特征中聚合出每类一个视觉嵌入（vpe）。

JSON 契约：
    {"image": "refer.jpg",
     "prompts": [{"class": "person", "bbox": [x1, y1, x2, y2]},
                 {"class": "person", "bbox": [...]}]}
类别顺序 = 首次出现顺序（vpe 行序与之对应）；bbox 为 refer 图像素坐标 xyxy。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["load_visual_prompt", "visual_masks", "collect_visual_embeddings"]


def load_visual_prompt(path):
    """读视觉 prompt JSON -> (image_path, names, boxes_by_class)

    names 按首次出现排序；boxes_by_class[q] 是第 q 类的原始图像 xyxy 框列表。
    格式错误直接 ValueError（不静默丢 prompt）。
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read visual prompt JSON {p}: {e}") from e
    image = data.get("image")
    prompts = data.get("prompts")
    if not image:
        raise ValueError(f"visual prompt JSON {p}: missing 'image'")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"visual prompt JSON {p}: 'prompts' must be a non-empty list")

    names, boxes = [], []
    for i, item in enumerate(prompts):
        name, bbox = item.get("class"), item.get("bbox")
        if not name or not isinstance(name, str):
            raise ValueError(f"visual prompt JSON {p}: prompts[{i}] needs a non-empty 'class'")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"visual prompt JSON {p}: prompts[{i}] 'bbox' must be [x1,y1,x2,y2]")
        box = [float(v) for v in bbox]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"visual prompt JSON {p}: prompts[{i}] bbox has no area: {box}")
        if name not in names:
            names.append(name)
            boxes.append([])
        boxes[names.index(name)].append(box)
    return image, names, boxes


def visual_masks(names, boxes_by_class, ratio, pad, imgsz):
    """原始图像 bbox -> (Q, imgsz/8, imgsz/8) 二值掩码（float32）

    ratio/pad 来自 `data.preprocess.preprocess` 的 letterbox 结果；口径与官方
    `LoadVisualPrompt` 一致：bbox 变换到输入坐标后乘 1/8，再按半开区间光栅化；
    同类别多框 OR 合并。Q = len(names)，行序与 names 一致。
    """
    if len(names) != len(boxes_by_class):
        raise ValueError(f"names ({len(names)}) and boxes ({len(boxes_by_class)}) length mismatch")
    h8 = w8 = int(imgsz) // 8
    xs = np.arange(w8, dtype=np.float32)
    ys = np.arange(h8, dtype=np.float32)
    pad_top, pad_left = pad
    out = np.zeros((len(names), h8, w8), dtype=np.float32)
    for q, boxes in enumerate(boxes_by_class):
        for x1, y1, x2, y2 in boxes:
            x1 = (x1 * ratio + pad_left) / 8
            y1 = (y1 * ratio + pad_top) / 8
            x2 = (x2 * ratio + pad_left) / 8
            y2 = (y2 * ratio + pad_top) / 8
            mask = ((xs[None, :] >= x1) & (xs[None, :] < x2) &
                    (ys[:, None] >= y1) & (ys[:, None] < y2)).astype(np.float32)
            out[q] = np.maximum(out[q], mask)
    return torch.from_numpy(out)


def collect_visual_embeddings(model, dataset, eval_names, imgsz, device="cpu", limit=0):
    """参考数据集的逐类 vpe（评估用）：每图每类一个嵌入，跨图求和后按类 L2 归一化

    与官方 `get_visual_pe` 等价：同图同类的多个 GT 框先 OR 成一张掩码，SAVPE 出一个
    嵌入；跨图按类别求和（逐类归一化会抵消官方按实例数的缩放）。缺失类别置零并返回
    名称列表（官方同口径：零嵌入类别仍参与评估，AP 自然为 0）。

    Returns:
        (vpe (1, len(eval_names), embed), missing_names)
    """
    from data.preprocess import preprocess

    head = model.model[-1]
    names = list(eval_names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    # COCO 的 annotations.category_id 是原始 id（经 cat_id_to_idx 映射到 0..nc-1）；
    # YOLO 的 targets() 直接给 0..nc-1。两种格式统一成“参考类别 id -> 评估类别 id”。
    coco_like = hasattr(dataset, "annotations")
    if coco_like:
        cat_to_idx = {cid: name_to_idx[dataset.names[idx]]
                      for cid, idx in dataset.cat_id_to_idx.items()
                      if dataset.names[idx] in name_to_idx}
    else:
        cat_to_idx = {cid: name_to_idx[n] for cid, n in dataset.names.items() if n in name_to_idx}
    missing = list(dict.fromkeys(n for n in dataset.names.values() if n not in name_to_idx))

    n_total = len(dataset) if not limit else min(limit, len(dataset))
    sums = torch.zeros(1, len(names), head.embed, device=device)
    present = torch.zeros(len(names), dtype=torch.bool, device=device)
    for i in range(n_total):
        image = dataset.load_image(i)  # YOLO 侧记录尺寸后 targets() 才可用
        by_cls = {}
        if coco_like:
            for ann in dataset.annotations.get(dataset.image_id(i), []):
                if ann.get("iscrowd", 0):
                    continue
                idx = cat_to_idx.get(ann["category_id"])
                if idx is None:
                    continue
                x, y, w, h = ann["bbox"]
                by_cls.setdefault(idx, []).append([x, y, x + w, y + h])
        else:
            for (x1, y1, x2, y2), cls in dataset.targets(i):
                idx = cat_to_idx.get(cls)
                if idx is not None:
                    by_cls.setdefault(idx, []).append([x1, y1, x2, y2])
        if not by_cls:
            continue
        tensor, ratio, pad = preprocess(image, imgsz)
        ids = sorted(by_cls)
        masks = visual_masks([names[j] for j in ids], [by_cls[j] for j in ids], ratio, pad, imgsz)
        with torch.no_grad():
            feats = model.forward_feats(tensor.to(device))
            vpe = head.get_vpe(feats, masks.unsqueeze(0).to(device))  # (1, Q, D)
        for q, idx in enumerate(ids):
            sums[0, idx] += vpe[0, q]
            present[idx] = True
    if present.any():
        sums[0, present] = F.normalize(sums[0, present], dim=-1, p=2)
    sums[0, ~present] = 0
    return sums, missing

"""视觉 prompt：JSON 解析 + 掩码光栅化（对齐官方 LoadVisualPrompt）

视觉 prompt = 一张 refer 图 + 若干示例；每个示例可以是 **bbox** 或 **mask**（每类可有
多个）。掩码在 **模型输入坐标** 的 1/8 分辨率（P3 网格）上生成：

- bbox：按 letterbox 的 ratio/pad 变换到输入坐标，再乘 1/8，按半开区间光栅化；
- mask：最近邻缩放到 letterbox 后的输入尺寸，padding 置 0，再最近邻降到 1/8；
- 同类别多个示例按 OR 合并。

JSON 契约：
    {"image": "refer.jpg",
     "prompts": [{"class": "person", "bbox": [x1, y1, x2, y2]},
                 {"class": "person", "mask": "person.png"}]}

类别顺序 = 首次出现顺序（vpe 行序与之对应）；bbox 为 refer 图像素坐标 xyxy；mask 为
与 refer 图同尺寸的二值图（非 0 即前景），相对路径按 JSON 所在目录解析。每个示例
必须且只能给 bbox / mask 之一。
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["load_visual_prompt", "visual_masks", "collect_visual_embeddings"]


def _load_mask(mask_path, json_path):
    p = Path(mask_path)
    if not p.is_absolute():  # 相对路径按 JSON 所在目录解析
        p = Path(json_path).resolve().parent / p
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read visual prompt mask: {p}")
    return mask > 0


def load_visual_prompt(path):
    """读视觉 prompt JSON -> (image_path, names, prompts_by_class)

    names 按首次出现排序；prompts_by_class[q] 是第 q 类的示例列表，元素为
    `{"kind": "bbox", "box": [x1,y1,x2,y2]}` 或 `{"kind": "mask", "mask": bool HxW}`。
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

    names, grouped = [], []
    for i, item in enumerate(prompts):
        name, bbox, mask_path = item.get("class"), item.get("bbox"), item.get("mask")
        if not name or not isinstance(name, str):
            raise ValueError(f"visual prompt JSON {p}: prompts[{i}] needs a non-empty 'class'")
        if (bbox is None) == (mask_path is None):
            raise ValueError(f"visual prompt JSON {p}: prompts[{i}] needs exactly one of 'bbox' / 'mask'")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"visual prompt JSON {p}: prompts[{i}] 'bbox' must be [x1,y1,x2,y2]")
            box = [float(v) for v in bbox]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"visual prompt JSON {p}: prompts[{i}] bbox has no area: {box}")
            prompt = {"kind": "bbox", "box": box}
        else:
            if not isinstance(mask_path, str) or not mask_path:
                raise ValueError(f"visual prompt JSON {p}: prompts[{i}] 'mask' must be a path")
            prompt = {"kind": "mask", "mask": _load_mask(mask_path, p)}
        if name not in names:
            names.append(name)
            grouped.append([])
        grouped[names.index(name)].append(prompt)
    return image, names, grouped


def visual_masks(names, prompts_by_class, ratio, pad, imgsz, src_shape):
    """原始图像示例（bbox / mask）-> (Q, imgsz/8, imgsz/8) 二值掩码（float32）

    ratio/pad 来自 `data.preprocess.preprocess` 的 letterbox 结果；src_shape 是 refer 图
    原始 (H, W)（mask 尺寸校验与 letterbox 缩放需要）。口径与官方 `LoadVisualPrompt`
    一致；同类别多个示例 OR 合并。Q = len(names)，行序与 names 一致。
    """
    if len(names) != len(prompts_by_class):
        raise ValueError(f"names ({len(names)}) and prompts ({len(prompts_by_class)}) length mismatch")
    h8 = w8 = int(imgsz) // 8
    xs = np.arange(w8, dtype=np.float32)
    ys = np.arange(h8, dtype=np.float32)
    pad_top, pad_left = pad
    src_h, src_w = src_shape
    new_h, new_w = round(src_h * ratio), round(src_w * ratio)
    out = np.zeros((len(names), h8, w8), dtype=np.float32)

    for q, items in enumerate(prompts_by_class):
        for item in items:
            kind = item.get("kind")
            if kind == "bbox":
                x1, y1, x2, y2 = item["box"]
                x1 = (x1 * ratio + pad_left) / 8
                y1 = (y1 * ratio + pad_top) / 8
                x2 = (x2 * ratio + pad_left) / 8
                y2 = (y2 * ratio + pad_top) / 8
                mask = ((xs[None, :] >= x1) & (xs[None, :] < x2) &
                        (ys[:, None] >= y1) & (ys[:, None] < y2)).astype(np.float32)
            elif kind == "mask":
                raw = item["mask"]
                if raw.shape != (src_h, src_w):
                    raise ValueError(f"visual prompt mask shape {raw.shape} != refer image {(src_h, src_w)}")
                resized = cv2.resize(raw.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                canvas = np.zeros((imgsz, imgsz), dtype=np.uint8)
                canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
                m = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(m, size=(h8, w8), mode="nearest")[0, 0].numpy()
            else:
                raise ValueError(f"unknown visual prompt kind: {kind!r}")
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
        prompts = [[{"kind": "bbox", "box": b} for b in by_cls[j]] for j in ids]
        masks = visual_masks([names[j] for j in ids], prompts, ratio, pad, imgsz, image.shape[:2])
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

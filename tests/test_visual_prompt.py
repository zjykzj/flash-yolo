"""视觉 prompt：JSON 契约 + 掩码光栅化（官方 LoadVisualPrompt 1/8 口径）"""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.visual_prompt import load_visual_prompt, visual_masks


def _write(tmp_path, payload):
    p = tmp_path / "prompt.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_visual_prompt_groups_boxes(tmp_path):
    """类别按首次出现排序；同类多个 bbox 归并；bbox 原图 xyxy"""
    p = _write(tmp_path, {
        "image": "refer.jpg",
        "prompts": [
            {"class": "person", "bbox": [10, 20, 30, 40]},
            {"class": "car", "bbox": [1, 2, 3, 4]},
            {"class": "person", "bbox": [50, 60, 70, 80]},
        ],
    })
    image, names, prompts = load_visual_prompt(p)
    assert image == "refer.jpg"
    assert names == ["person", "car"]
    assert prompts == [
        [{"kind": "bbox", "box": [10.0, 20.0, 30.0, 40.0]},
         {"kind": "bbox", "box": [50.0, 60.0, 70.0, 80.0]}],
        [{"kind": "bbox", "box": [1.0, 2.0, 3.0, 4.0]}],
    ]


def test_load_visual_prompt_mask(tmp_path):
    """mask 为路径（相对 JSON 目录），读成 bool HxW"""
    import cv2

    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    cv2.imwrite(str(tmp_path / "m.png"), mask)
    p = _write(tmp_path, {"image": "refer.jpg",
                          "prompts": [{"class": "person", "mask": "m.png"}]})
    _, names, prompts = load_visual_prompt(p)
    assert names == ["person"]
    assert prompts[0][0]["kind"] == "mask"
    assert prompts[0][0]["mask"].shape == (16, 16)
    assert prompts[0][0]["mask"].sum() == 64


def test_load_visual_prompt_rejects_bad_schema(tmp_path):
    with pytest.raises(ValueError, match="missing 'image'"):
        load_visual_prompt(_write(tmp_path, {"prompts": [{"class": "a", "bbox": [0, 0, 1, 1]}]}))
    with pytest.raises(ValueError, match="non-empty list"):
        load_visual_prompt(_write(tmp_path, {"image": "x", "prompts": []}))
    with pytest.raises(ValueError, match="needs a non-empty 'class'"):
        load_visual_prompt(_write(tmp_path, {"image": "x", "prompts": [{"bbox": [0, 0, 1, 1]}]}))
    with pytest.raises(ValueError, match="bbox has no area"):
        load_visual_prompt(_write(tmp_path, {"image": "x", "prompts": [
            {"class": "a", "bbox": [2, 2, 2, 3]}]}))
    with pytest.raises(ValueError, match="exactly one of 'bbox' / 'mask'"):
        load_visual_prompt(_write(tmp_path, {"image": "x", "prompts": [{"class": "a"}]}))
    with pytest.raises(ValueError, match="exactly one of 'bbox' / 'mask'"):
        load_visual_prompt(_write(tmp_path, {"image": "x", "prompts": [
            {"class": "a", "bbox": [0, 0, 1, 1], "mask": "m.png"}]}))


def test_visual_masks_rasterize_and_merge():
    """imgsz=64 -> 8×8；[8,8,16,16] 落在网格 (1,1)；同类两框 OR 合并"""
    prompts = [[{"kind": "bbox", "box": [8, 8, 16, 16]},
                {"kind": "bbox", "box": [24, 24, 32, 32]}]]
    masks = visual_masks(["person"], prompts, 1.0, (0, 0), 64, (64, 64))
    assert tuple(masks.shape) == (1, 8, 8)
    assert masks[0].sum() == 2
    assert masks[0, 1, 1] == 1 and masks[0, 3, 3] == 1


def test_visual_masks_letterbox_transform():
    """ratio/pad 先变换到输入坐标，再乘 1/8（官方 LoadVisualPrompt 口径）"""
    # x: 20*0.5+20=30 -> 3.75; 40*0.5+20=40 -> 5.0；y: 20*0.5+10=20 -> 2.5; 40*0.5+10=30 -> 3.75
    prompts = [[{"kind": "bbox", "box": [20, 20, 40, 40]}]]
    masks = visual_masks(["a"], prompts, 0.5, (10, 20), 64, (100, 100))
    assert masks[0, 3, 4] == 1
    assert masks[0].sum() == 1


def test_visual_masks_input():
    """mask 最近邻 letterbox 到输入尺寸，再降到 1/8；与同区域 mask 的 bbox 一致"""
    mask = np.zeros((64, 64), np.uint8)
    mask[8:16, 8:16] = 1
    masks = visual_masks(["a"], [[{"kind": "mask", "mask": mask.astype(bool)}]],
                         1.0, (0, 0), 64, (64, 64))
    assert masks[0].sum() == 1 and masks[0, 1, 1] == 1
    with pytest.raises(ValueError, match="mask shape"):
        visual_masks(["a"], [[{"kind": "mask", "mask": mask.astype(bool)}]],
                     1.0, (0, 0), 64, (32, 32))


def test_mask_rasterization_matches_ultralytics():
    """mask 1/8 光栅化与官方 LoadVisualPrompt 逐位一致（dev 依赖 ultralytics）"""
    import torch
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LoadVisualPrompt

    mask = np.zeros((64, 64), np.uint8)
    mask[8:16, 8:16] = 255
    official = LoadVisualPrompt().get_visuals(
        [0], (64, 64), masks=torch.from_numpy(mask.astype(np.float32))[None])
    ours = visual_masks(["a"], [[{"kind": "mask", "mask": mask > 0}]],
                        1.0, (0, 0), 64, (64, 64))
    assert torch.equal(official, ours)


def test_collect_visual_embeddings_orders_and_normalizes():
    """refer 数据集逐类聚合：已知类归一化、缺失类零行、返回未覆盖的 refer 类"""
    import torch

    from utils.visual_prompt import collect_visual_embeddings

    class Head:
        embed = 4

        def get_vpe(self, x, vp):
            return torch.ones(1, vp.shape[1], self.embed, device=vp.device)

    class Model:
        def __init__(self):
            self.model = [Head()]

        def forward_feats(self, x):
            return [x]

    class DS:
        names = {0: "a", 1: "b", 2: "z"}
        cat_id_to_idx = {0: 0, 1: 1, 9: 2}  # 原始 category_id -> names 索引
        annotations = {
            1: [{"category_id": 0, "bbox": [1, 1, 10, 10]},
                {"category_id": 0, "bbox": [2, 2, 5, 5]}],
            2: [{"category_id": 1, "bbox": [3, 3, 8, 8]}],
        }

        def __len__(self):
            return 2

        def image_id(self, i):
            return i + 1

        def load_image(self, i):
            return np.zeros((32, 32, 3), np.uint8)

    vpe, missing = collect_visual_embeddings(Model(), DS(), ["a", "b", "c"], 64, "cpu")
    assert vpe.shape == (1, 3, 4)
    assert torch.allclose(vpe[0, 0].norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(vpe[0, 1].norm(), torch.tensor(1.0), atol=1e-6)
    assert vpe[0, 2].abs().sum() == 0  # 没有 refer 样本的类别
    assert missing == ["z"]  # refer 数据里存在、评估集没有的类别

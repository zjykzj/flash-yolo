"""TAL + STAL 分配器验收：已知答案 / STAL 代理框 / 多 GT 竞争 / o2o 一对一 / 空 GT"""

import numpy as np
import torch

from train.assigner import TaskAlignedAssigner, bbox_iou_torch


def _grid_anchors(shape, stride=8):
    """手工构造 (2, N) anchors 与 (1, N) strides（meshgrid ij 行主序）"""
    h, w = shape
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    anchors = torch.tensor(np.stack([xs.ravel(), ys.ravel()]), dtype=torch.float32)
    strides = torch.full((1, h * w), stride, dtype=torch.float32)
    return anchors, strides


def _pred(anchors, ltrb=1.0):
    """所有 anchor 同一 ltrb 预测（pred 框 = 以 anchor 为中心的 2x2 框）"""
    n = anchors.shape[1]
    return torch.full((4, n), ltrb)


def test_bbox_iou_known_values():
    """IoU 已知值 + CIoU 自反性"""
    a = torch.tensor([[0.0, 0, 10, 10]])
    b = torch.tensor([[5.0, 5, 15, 15]])
    iou = bbox_iou_torch(a, b, ciou=False)
    np.testing.assert_allclose(iou.item(), 25 / 175, atol=1e-6)  # 5x5 交 / 175 并
    assert bbox_iou_torch(a, a, ciou=True).item() == 1.0, "同框 CIoU 应为 1"
    print("  IoU 已知值正确")


def test_bbox_iou_degenerate_boxes():
    """回归：零面积退化框 CIoU 前向有限、反向无 NaN（0/0 根因修复）"""
    box = torch.tensor([[5.0, 5, 5, 5]], requires_grad=True)  # w=h=0（ltrb->0 的预测框）
    gt = torch.tensor([[0.0, 0, 10, 10]])
    ciou = bbox_iou_torch(box, gt, ciou=True)
    assert torch.isfinite(ciou).all(), f"退化框 CIoU 应有限: {ciou}"
    ciou.sum().backward()
    assert torch.isfinite(box.grad).all(), f"退化框反向应有限: {box.grad}"
    print(f"  退化框 CIoU 有限: {ciou.item():.4f}")


def test_tal_known_answer():
    """4x4 网格双 GT：候选/竞争/软标签/回归目标逐项验证"""
    anchors, strides = _grid_anchors((4, 4))
    # GT1 cls0 [8,8,24,24]px -> grid [1,1,3,3]（4 候选）; GT2 cls1 [16,16,40,40]px -> grid [2,2,5,5]（4 候选）
    gt = torch.tensor([[0, 8, 8, 24, 24], [1, 16, 16, 40, 40]], dtype=torch.float32)
    scores = torch.full((2, 16), -10.0)
    scores[0, 5] = 10.0   # anchor (1.5,1.5) 高分 cls0
    scores[0, 10] = 10.0  # anchor (2.5,2.5) 高分 cls0
    scores[1, 10] = 10.0  # anchor (2.5,2.5) 高分 cls1（竞争）
    scores[1, 15] = 10.0  # anchor (3.5,3.5) 高分 cls1

    out = TaskAlignedAssigner(nc=2).forward(_pred(anchors), scores, anchors, strides, gt)
    # GT1 候选 {5,6,9,10} + GT2 候选 {10,11,14,15}，共享 idx10 竞争后归 GT1 -> n_pos=7
    assert out["n_pos"] == 7, f"n_pos={out['n_pos']}"
    fg = {5, 6, 9, 10, 11, 14, 15}
    assert set(out["fg_mask"].nonzero().flatten().tolist()) == fg
    assert not out["fg_mask"][0].item(), "角点 anchor 不在任何 GT 内"
    # 竞争：anchor(2.5,2.5) 与 GT1 的 CIoU 更高 -> 归 GT1
    assert out["target_scores"][10].argmax().item() == 0, "竞争 anchor 应归 CIoU 更高的 GT1"
    assert out["target_scores"][5].argmax().item() == 0
    assert out["target_scores"][15].argmax().item() == 1
    # 软标签 per-GT 归一化
    assert abs(out["target_scores"][[5, 6, 9, 10], 0].sum().item() - 1.0) < 1e-5
    assert abs(out["target_scores"][[11, 14, 15], 1].sum().item() - 1.0) < 1e-5
    # 回归目标（逐 anchor grid 单位，原始框）
    np.testing.assert_allclose(out["target_ltrb"][10].tolist(), [1.5, 1.5, 0.5, 0.5], atol=1e-5)  # GT1 [1,1,3,3]
    np.testing.assert_allclose(out["target_boxes"][10].tolist(), [1, 1, 3, 3], atol=1e-5)
    print("  TAL 已知答案全部正确")


def test_stal_surrogate():
    """STAL：3x3px 小 GT 经代理框扩宽后获得正样本；回归仍用原始框"""
    anchors, strides = _grid_anchors((1, 1))  # 单 anchor (0.5,0.5), stride 8
    gt = torch.tensor([[0, 0, 0, 3, 3]], dtype=torch.float32)  # grid [0,0,0.375,0.375]
    scores = torch.full((1, 1), 10.0)
    boxes = torch.full((4, 1), 0.1)

    # 无 STAL（s_min=0）：中心 0.5 > 0.375 不在框内 -> 无正样本
    out_none = TaskAlignedAssigner(nc=1, s_min=0.0).forward(boxes, scores, anchors, strides, gt)
    assert out_none["n_pos"] == 0, "无代理框时小 GT 不应有候选"

    # 有 STAL：短边 0.375 < s_min/8=1 -> 扩到 2 grid 单位 -> 中心落入 -> 正样本
    out = TaskAlignedAssigner(nc=1).forward(boxes, scores, anchors, strides, gt)
    assert out["n_pos"] == 1, f"n_pos={out['n_pos']}"
    # 回归目标基于原始框 [0,0,0.375,0.375]：l=0.5, t=0.5, r=0.375-0.5, b=0.375-0.5
    np.testing.assert_allclose(out["target_ltrb"][0].tolist(), [0.5, 0.5, -0.125, -0.125], atol=1e-5)
    np.testing.assert_allclose(out["target_boxes"][0].tolist(), [0, 0, 0.375, 0.375], atol=1e-5)
    print("  STAL 代理框行为正确（回归仍用原始框）")


def test_o2o_one_to_one():
    """o2o 二次 top-k：单 GT 全覆盖 4 候选 -> 二次 top-1 只留 1 个 anchor（真一对一）"""
    anchors, strides = _grid_anchors((2, 2))
    gt = torch.tensor([[0, 0, 0, 16, 16]], dtype=torch.float32)  # grid [0,0,2,2] 覆盖全部 4 anchor
    scores = torch.full((1, 4), 10.0)
    boxes = torch.full((4, 4), 0.5)
    boxes[:, 3] = 0.6  # idx3 预测框更大 -> CIoU 唯一最大（打破对称平局）

    out_m = TaskAlignedAssigner(nc=1).forward(boxes, scores, anchors, strides, gt, one2one=False)
    assert out_m["n_pos"] == 4, f"o2m 一次 top-k 应取全部 4 候选: n_pos={out_m['n_pos']}"

    out = TaskAlignedAssigner(nc=1).forward(boxes, scores, anchors, strides, gt, one2one=True)
    assert out["n_pos"] == 1, f"二次 top-1 应只剩 1 个: n_pos={out['n_pos']}"
    # 唯一胜出者 = CIoU 最高的 anchor（pred 框 [1,1,2,2] vs GT [0,0,2,2] 中心重合者为 idx3）
    assert out["fg_mask"][3].item(), "CIoU 最高的 anchor (1.5,1.5) 应胜出"
    print("  o2o 二次 top-k 一对一语义正确")


def test_empty_gt():
    anchors, strides = _grid_anchors((2, 2))
    out = TaskAlignedAssigner(nc=2).forward(_pred(anchors), torch.zeros(2, 4), anchors, strides,
                                            torch.zeros(0, 5))
    assert out["n_pos"] == 0 and not out["fg_mask"].any()
    assert out["target_scores"].shape == (4, 2)
    print("  空 GT 安全返回")

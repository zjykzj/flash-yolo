"""anchor 生成与框解码（E2E / NMS 两条路径共用）"""

import torch

__all__ = ["make_anchors", "dist2bbox"]


def make_anchors(feats, strides, offset=0.5):
    """生成 anchor 点（grid 单位）与逐点 stride

    Args:
        feats: 各检测层特征图列表 [(B, C, H, W), ...]
        strides: 各层 stride [8, 16, 32]
        offset: 网格偏移，0.5 即格点中心

    Returns:
        anchors: (2, N) 格点坐标（未乘 stride），转置形式便于与 (B, 4, N) 的
            预测张量按最后一维广播（与 dist2bbox 配合）
        stride_tensor: (1, N) 每点 stride
    """
    anchors, stride_tensor = [], []
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[2:]
        # 先在 fp32 建格再 cast 到 feat.dtype：half 的 arange 会被 TorchScript ONNX 导出器拒绝
        # （"tensor does not have a device"），挡住 model.half() -> fp16 onnx 的整条路。
        # 格点值（≤639.5）在 fp16 可精确表示，fp32 路径数值不变。
        sx = (torch.arange(w, dtype=torch.float32, device=feat.device) + offset).to(feat.dtype)
        sy = (torch.arange(h, dtype=torch.float32, device=feat.device) + offset).to(feat.dtype)
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchors.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(feat.new_full((h * w, 1), stride))
    return torch.cat(anchors).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)


def dist2bbox(distance, anchor_points, xywh=False, dim=1):
    """ltrb 距离 -> 框坐标

    Args:
        distance: (B, 4, N) 左/上/右/下距离（4 在 dim 1）
        anchor_points: (1, 2, N) anchor 中心
        xywh: True 输出中心宽高，False 输出 xyxy
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)

"""YOLOv3-tiny 检测头：anchor-based 双尺度（P4/16 + P5/32）、每级 3 锚 × (5+nc) 通道

设计要点（对照 darknet cfg/yolov3-tiny.cfg）：
- 两颗 1×1→3*(5+nc) 输出卷积在此（darknet 里是独立层；折进来与 yolo26 的 Detect 持有
  cv2/cv3 的惯例一致）：bias=True、无 BN/激活（darknet linear 卷积口径）。
- anchors/strides 是 yaml 数据（ref_imgsz=416 的像素单位），构建时按 imgsz/ref_imgsz 线性
  缩放；注册为 persistent=False 的 buffer——不进 state_dict（转换器键集对账、EMA 键断言、
  strict load 三处零特判），且随 .to(device) 自动搬运。
- 解码唯一实现见 decode_level（darknet get_yolo_box 口径；训练损失与推理共用，不得各写一份）。
"""

import torch
import torch.nn as nn

__all__ = ["V3Detect", "decode_level"]


def decode_level(raw, anchors, stride, no, na=3):
    """单级原始输出 -> (boxes xyxy 像素, obj logits, cls logits)

    raw: (B, na*no, H, W)，通道序 = 逐锚 [x, y, w, h, obj, cls...]。
    darknet 口径：xy = (σ(t) + 格) · stride、wh = 锚 × exp(t)（t 为原始 logits），网格偏移 0。
    tw/th 先温和 clamp(±8) 防训练初期 exp 溢出（e^8 ≈ 2981 倍锚框，正常权重不触发）。
    """
    b, _, h, w = raw.shape
    r = raw.reshape(b, na, no, h, w).permute(0, 1, 3, 4, 2)  # (B, na, H, W, no)
    tx, ty = r[..., 0].sigmoid(), r[..., 1].sigmoid()
    tw, th = r[..., 2].clamp(-8.0, 8.0), r[..., 3].clamp(-8.0, 8.0)
    gx = torch.arange(w, device=raw.device, dtype=raw.dtype).view(1, 1, 1, w)
    gy = torch.arange(h, device=raw.device, dtype=raw.dtype).view(1, 1, h, 1)
    cx = (tx + gx) * stride
    cy = (ty + gy) * stride
    bw = anchors[:, 0].view(1, na, 1, 1) * tw.exp()
    bh = anchors[:, 1].view(1, na, 1, 1) * th.exp()
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
    return boxes, r[..., 4], r[..., 5:]


class V3Detect(nn.Module):
    """YOLOv3-tiny 检测头（anchor-based）

    输出：
        train: 每级原始 (B, 3*(5+nc), H, W) 列表（损失消费）
        eval : 解码拼接 (B, NA, 5+nc) = [x1,y1,x2,y2, σ(obj), σ(cls)...]，NA = 3·ΣH·W
               （640 输入 = 6000、416 时 2535；扁平序 = 级 → 槽 → y → x）
    无 end2end/cv2/cv3 属性；不需要 bias_init（darknet 无此先验）。
    """

    na = 3

    def __init__(self, nc=80, ch=(), anchors=None, strides=None, ref_imgsz=416, imgsz=640):
        super().__init__()
        if anchors is None or strides is None:
            raise ValueError("V3Detect requires anchors and strides (from the model yaml)")
        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 5
        self.m = nn.ModuleList(nn.Conv2d(c, self.no * self.na, 1) for c in ch)
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32) * (imgsz / ref_imgsz),
                             persistent=False)  # (nl, na, 2) 输入像素单位
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32), persistent=False)

    def _forward_train(self, x):
        """训练口径：每级原始输出（validator 与损失共用同一入口）"""
        return [self.m[i](x[i]) for i in range(self.nl)]

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        outs = []
        for lvl, raw in enumerate(self._forward_train(x)):
            boxes, obj, cls = decode_level(raw, self.anchors[lvl], self.stride[lvl], self.no, self.na)
            b = boxes.shape[0]
            outs.append(torch.cat([
                boxes.reshape(b, -1, 4),
                obj.reshape(b, -1, 1).sigmoid(),
                cls.reshape(b, -1, self.nc).sigmoid(),
            ], dim=-1))
        return torch.cat(outs, dim=1)  # (B, NA, 5+nc)

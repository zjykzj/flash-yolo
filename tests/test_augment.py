"""增强管线验收：恒等变换不变量 / 镜像精确性 / mosaic 几何 / 框过滤 / 空标签存活

几何口径对照 ultralytics/data/augment.py + data/base.py（8.4.173）：
- 载图等比缩放到长边 = imgsz（不拉伸成方形）
- mosaic 每块按自然尺寸锚定在随机中心 (xc, yc)（不逐块随机缩放、不贴象限）
- 仿射后按官方 box_candidates 过滤碎片框
- HSV 在 HSV 空间（hue 加性、sat/val 乘性、lut_sat[0]=0）
"""

import numpy as np
import torch

from data.augment import augment, bgr_swap, flip_lr, mosaic, random_hsv, random_perspective, resize_long_side
from config.train import TrainConfig


def _identity_hyp(**over):
    """透视恒等、其余关闭的最小增强配置

    aug_scale 是增益幅度（官方 random_perspective 语义：g ~ U(1-s, 1+s)），
    恒等对应 0.0（旧实现为 [s, 1/s] 约定、用 1.0 表示恒等）。
    """
    hyp = TrainConfig(
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, shear=0.0, translate=0.0,
        aug_scale=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
        imgsz=640,
    )
    for k, v in over.items():
        setattr(hyp, k, v)
    return hyp


def _img(size=640, value=60):
    return np.full((size, size, 3), value, np.uint8)


def _loader(images):
    def load(i):
        return images[i][0].copy(), images[i][1].copy()

    return load


def test_perspective_identity_invariant():
    """恒等仿射参数：标签不变"""
    hyp = _identity_hyp()
    img = _img()
    labels = np.array([[0, 100, 50, 300, 250], [1, 10, 10, 20, 20]], np.float32)
    out_img, out_labels = random_perspective(img, labels, hyp, np.random.default_rng(0))
    np.testing.assert_allclose(out_labels, labels, atol=1e-3)
    print("  恒等透视标签不变:", out_labels.tolist())


def test_fliplr_exact_mirror():
    """水平镜像：x' = W - x 精确"""
    img = _img()
    labels = np.array([[0, 100, 50, 200, 300]], np.float32)
    out, out_labels = flip_lr(img, labels)
    np.testing.assert_array_equal(out_labels, [[0, 640 - 200, 50, 640 - 100, 300]])
    print("  fliplr 镜像精确:", out_labels.tolist())


def test_hsv_zero_gain_near_identity():
    """hsv 增益为 0：HLS 往返仅允许转换级舍入误差"""
    img = _img(value=120)
    out = random_hsv(img, 0.0, 0.0, 0.0, np.random.default_rng(0))
    assert int(np.abs(out.astype(int) - img.astype(int)).max()) <= 2
    print("  hsv(0) 近似恒等")


def test_mosaic_bounds():
    """mosaic 输出 (2S,2S)，标签全部在画布内"""
    S = 640
    hyp = _identity_hyp()
    images = [(_img(value=40), np.array([[0, 10, 10, 100, 100]], np.float32)) for _ in range(8)]
    img, labels = mosaic(_loader(images), 0, len(images), hyp, np.random.default_rng(1))
    assert img.shape == (S * 2, S * 2, 3)
    assert len(labels) >= 1
    assert labels[:, 1:].min() >= 0 and labels[:, 1].max() <= S * 2 and labels[:, 3].max() <= S * 2
    assert labels[:, 2].max() <= S * 2 and labels[:, 4].max() <= S * 2
    print(f"  mosaic 边界正确（{len(labels)} 框）")


def test_augment_full_pipeline_empty_labels():
    """空标签图像走完整管线存活；输出张量形状正确"""
    hyp = _identity_hyp(mosaic=1.0)
    images = [(_img(value=40), np.zeros((0, 5), np.float32)) for _ in range(4)]
    tensor, labels = augment(_img(value=40), np.zeros((0, 5), np.float32), _loader(images), hyp,
                             np.random.default_rng(2), 0, 4)
    assert tensor.shape == (3, 640, 640) and tensor.dtype == torch.float32
    assert labels.shape == (0, 5)
    print("  空标签管线存活")


# ---- 几何口径回归（官方对齐新增）----
class _StubRNG:
    """确定性 RNG：uniform 取区间中点、integers 轮流取 seq、random 固定 0（不触发概率分支）"""

    def __init__(self, center=True, seq=(0, 1, 2, 3), rand=0.0):
        self.center = center
        self.seq = list(seq)
        self.rand = rand
        self.k = 0

    def uniform(self, lo, hi=None, size=None):
        if size is not None:  # numpy 风格 uniform(low, high, size)：取上界
            return np.full(size, hi)
        if hi is None:
            lo, hi = 0.0, lo
        return (lo + hi) / 2 if self.center else lo

    def integers(self, lo, hi):
        v = self.seq[self.k % len(self.seq)]
        self.k += 1
        return v

    def random(self, size=None):
        return np.full(size, self.rand) if size else self.rand

    def beta(self, a, b):
        return 0.5


def test_resize_long_side_preserves_aspect():
    """等比缩放：长边 -> size，短边按比例（不拉伸成方形）"""
    img = np.zeros((480, 1280, 3), np.uint8)  # 1280x480 -> 640x240
    labels = np.array([[0, 0, 0, 1280, 480]], np.float32)
    out, lbs = resize_long_side(img, labels, 640)
    assert out.shape[:2] == (240, 640), out.shape
    np.testing.assert_allclose(lbs[0, 1:], [0, 0, 640, 240])
    out2, _ = resize_long_side(np.zeros((480, 640, 3), np.uint8), np.zeros((0, 5), np.float32), 640)
    assert out2.shape[:2] == (480, 640), "长边已等于 size 时不应改写"
    print("  resize_long_side 保纵横比:", out.shape[:2], out2.shape[:2])


def test_mosaic_tiles_anchor_at_random_center():
    """mosaic 几何：每块按自然尺寸锚定在 (xc,yc)，不拉伸、不逐块缩放

    4 块 64x48 纯色图 + 居中 rng：画布行 16-64 为上下两块（高 48 而非 64），
    行 0-16 / 112-128 为灰边。
    """
    S = 64
    hyp = _identity_hyp(imgsz=S)
    colors = [(10, 10, 10), (40, 40, 40), (70, 70, 70), (100, 100, 100)]
    images = [(np.full((48, 64, 3), c, np.uint8), np.array([[0, 10, 5, 30, 25]], np.float32)) for c in colors]
    canvas, labels = mosaic(_loader(images), 0, 4, hyp, _StubRNG(seq=(1, 2, 3)))  # xc = yc = S
    assert canvas.shape == (S * 2, S * 2, 3)
    np.testing.assert_array_equal(canvas[16:64, 0:64, 0], colors[0][0])  # 左上
    np.testing.assert_array_equal(canvas[16:64, 64:128, 0], colors[1][0])  # 右上
    np.testing.assert_array_equal(canvas[64:112, 0:64, 0], colors[2][0])  # 左下
    np.testing.assert_array_equal(canvas[64:112, 64:128, 0], colors[3][0])  # 右下
    np.testing.assert_array_equal(canvas[0:16], 114)  # 灰边（未被 48 高的块覆盖）
    np.testing.assert_array_equal(canvas[112:128], 114)
    # 标签随块平移：块 0 的框 (10,5,30,25) -> 画布 (10, 21, 30, 41)
    np.testing.assert_allclose(labels[0, 1:], [10, 21, 30, 41])
    print(f"  mosaic 中心锚定/保纵横比正确（{len(labels)} 框）")


def test_affine_drops_fragment_boxes():
    """仿射后碎片框过滤（官方 box_candidates）：面积保留率 <=10% / 宽高 <=2px 丢弃"""
    hyp = _identity_hyp(imgsz=64)  # translate=0 -> 恒等映射
    img = _img(size=64, value=30)
    labels = np.array([
        [0, 10, 10, 50, 50],    # 完整框：保留
        [1, 58, 10, 118, 50],   # 裁剪后 6x40，面积保留 240/2400 = 0.10：丢弃
        [2, 63, 10, 123, 50],   # 裁剪后 1x40：宽 <= 2px：丢弃
        [3, 55, 10, 115, 50],   # 裁剪后 9x40，面积保留 0.15：保留
    ], np.float32)
    _, out = random_perspective(img, labels, hyp, _StubRNG(center=False))
    kept = out[:, 0].astype(int).tolist()
    assert kept == [0, 3], kept
    np.testing.assert_allclose(out[1, 1:], [55, 10, 64, 50])
    print("  仿射碎片框过滤:", kept)


def test_hsv_applies_gain_in_hsv_space():
    """HSV 口径：仅 s_gain 时 V 通道不动、饱和度提升（旧实现把增益乘到 HLS 的 L 上）"""
    img = np.full((4, 4, 3), (128, 128, 255), np.uint8)  # 浅红 BGR：V=255, S=127
    out = random_hsv(img, 0.0, 0.5, 0.0, _StubRNG(center=False))
    assert out[0, 0, 2] == 255, "V 不应改变"
    assert out[0, 0, 0] == out[0, 0, 1] < 128, f"饱和度应提升（min 通道下降），实际 {out[0, 0]}"
    white = np.full((4, 4, 3), 255, np.uint8)
    np.testing.assert_array_equal(random_hsv(white, 0.0, 0.9, 0.9, _StubRNG(center=False)), white)  # lut_sat[0]=0
    print("  HSV 增益落在 HSV 空间:", out[0, 0].tolist())


def test_bgr_swap_reverses_channels():
    """bgr 增强 = RGB<->BGR 通道序互换（官方 Format 的 img[::-1]）"""
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0], img[..., 2] = 200, 10  # B=200, R=10
    out = bgr_swap(img)
    assert out[0, 0, 0] == 10 and out[0, 0, 2] == 200
    print("  bgr 通道互换正确")

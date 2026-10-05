"""增强管线验收：恒等变换不变量 / 镜像精确性 / mosaic 边界 / 空标签存活"""

import numpy as np
import torch

from data.augment import augment, flip_lr, mosaic, random_hsv, random_perspective
from config.train import TrainConfig


def _identity_hyp(**over):
    """透视恒等、其余关闭的最小增强配置"""
    hyp = TrainConfig(
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, shear=0.0, translate=0.0,
        aug_scale=1.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
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

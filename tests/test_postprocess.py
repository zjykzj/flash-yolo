"""nms_per_image 验收：与逐类参考实现等价（含早停/垃圾期极值）/ 类隔离 / 边界

参考实现 = 重写前的逐类全量贪心（本文件保留副本），两者输出须一致——允许同分块内部
排列不同（重写前是"逐类 NMS 后合并再排序"，重写后是"全局分数序贪心"，同分才可能异序）。
"""

import numpy as np

from utils.iou import box_iou
from utils.postprocess import nms_per_image


def _reference_nms(boxes, score_map, conf_thres, iou_thres, max_det):
    """逐类全量贪心（旧实现口径）：类内贪心 -> 合并 -> 全局 top-max_det"""
    out = []
    for c in range(score_map.shape[1]):
        mask = score_map[:, c] > conf_thres
        b, s = boxes[mask], score_map[mask, c]
        if len(b) == 0:
            continue
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            order = order[1:][box_iou(b[i], b[order[1:]]) < iou_thres]
        out.append(np.concatenate([b[keep], s[keep, None], np.full((len(keep), 1), c, np.float32)], axis=1))
    if not out:
        return np.zeros((0, 6), dtype=np.float32)
    merged = np.concatenate(out)
    return merged[merged[:, 4].argsort()[::-1][:max_det]]


def _canon(a):
    """按 (分数降序, 框坐标, 类) 全序规范化——同分块内部排列差异不算差异"""
    return a[np.lexsort((a[:, 5], a[:, 3], a[:, 2], a[:, 1], a[:, 0], -a[:, 4]))]


def _random_case(rng, n, nc, scale=100.0):
    centers = rng.uniform(0, scale, (n, 2))
    wh = rng.uniform(2, 0.6 * scale, (n, 2))
    boxes = np.concatenate([centers - wh / 2, centers + wh / 2], 1).astype(np.float32)
    scores = (rng.random((n, nc)) ** 3).astype(np.float32)
    return boxes, scores


def test_matches_per_class_reference():
    """随机簇状数据（含大量重叠）：规范化后逐位等价"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        boxes, scores = _random_case(rng, 500, 4)
        got = nms_per_image(boxes, scores, 0.05, 0.7, 300)
        ref = _reference_nms(boxes, scores, 0.05, 0.7, 300)
        assert np.array_equal(_canon(got), _canon(ref))
    print("  NMS 与逐类参考实现等价（20 组随机簇）")


def test_all_pass_conf_early_stop():
    """垃圾期极值：几乎全部候选通过 conf——早停必须给出与全量 NMS 相同的 top-max_det"""
    rng = np.random.default_rng(1)
    boxes, scores = _random_case(rng, 500, 10, scale=200.0)
    scores = np.clip(scores, 0.3, 1.0)  # 模拟 sigmoid(0)² ≈ 0.25 的未训练状态
    got = nms_per_image(boxes, scores, 0.001, 0.7, 100)
    ref = _reference_nms(boxes, scores, 0.001, 0.7, 100)
    assert len(got) == 100 and len(ref) == 100
    assert np.array_equal(_canon(got), _canon(ref))
    print("  全候选通过 conf 时早停与全量等价（top-100）")


def test_class_isolation():
    """同一框不同类互不抑制（类偏移隔离）——两类的最高分都可保留"""
    boxes = np.tile(np.array([[10, 10, 20, 20]], np.float32), (2, 1))
    scores = np.array([[0.9, 0.0], [0.0, 0.9]], np.float32)
    det = nms_per_image(boxes, scores, 0.1, 0.7, 300)
    assert len(det) == 2 and set(det[:, 5].astype(int)) == {0, 1}
    print("  类隔离正确（同框不同类均保留）")


def test_max_det_and_empty():
    """max_det 截断 / 按分降序 / 全部低于 conf / max_det=0 的边界"""
    rng = np.random.default_rng(2)
    centers = rng.uniform(0, 1000, (50, 2))  # 大间距小框：互不抑制，保底 50 个检出
    wh = rng.uniform(5, 20, (50, 2))
    boxes = np.concatenate([centers - wh / 2, centers + wh / 2], 1).astype(np.float32)
    scores = rng.random((50, 3)).astype(np.float32)

    det = nms_per_image(boxes, scores, 0.0, 0.7, 7)
    assert len(det) == 7, "被截断到 max_det"
    assert np.all(np.diff(det[:, 4]) <= 0), "输出按分降序"

    empty = nms_per_image(boxes, scores, 1.1, 0.7, 300)
    assert empty.shape == (0, 6) and empty.dtype == np.float32
    assert nms_per_image(boxes, scores, 0.0, 0.7, 0).shape == (0, 6)
    print("  max_det 截断 / 空输入 / 边界正确")

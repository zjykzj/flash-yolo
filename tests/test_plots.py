"""训练报告图：results.csv -> results.png；FastMetrics -> PR / 混淆矩阵 PNG

matplotlib 缺失时跳过（utils.plots 自身也会告警并降级）。
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from train.metrics import FastMetrics
from utils.plots import plot_confusion_matrix, plot_pr_curves, plot_results, write_training_report


def _metrics():
    m = FastMetrics(nc=2, task="segment")
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    cls = np.array([0, 1])
    gts = boxes.copy()
    gt_cls = np.array([0, 1])
    m.update(boxes, scores, cls, gts, gt_cls)
    masks = np.zeros((2, 16, 16), np.uint8)
    masks[0, 1:5, 1:5] = 1
    masks[1, 8:12, 8:12] = 1
    m.update_mask(masks, scores, cls, masks, gt_cls)
    return m


def test_results_plot(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,box,seg,cls,mAP,mAP@[.5:.95](M)\n"
                        "1,2.0,3.0,4.0,0.1,0.05\n2,1.0,2.0,3.0,0.2,0.1\n", encoding="utf-8")
    out = tmp_path / "results.png"
    assert plot_results(csv_path, out) and out.exists() and out.stat().st_size > 0


def test_pr_and_confusion_plots(tmp_path):
    m = _metrics()
    pr = tmp_path / "PR_curve.png"
    assert plot_pr_curves({"Box": m.pr_curve("detect"), "Mask": m.pr_curve("segment")}, pr)
    assert pr.exists() and pr.stat().st_size > 0
    cm = tmp_path / "cm.png"
    assert plot_confusion_matrix(m.confusion_matrix(), ["a", "b"], cm)
    assert cm.exists() and cm.stat().st_size > 0


def test_write_training_report(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,box\n1,2.0\n", encoding="utf-8")
    files = write_training_report(tmp_path, csv_path, _metrics(), ["a", "b"], "segment")
    assert {"results.png", "PR_curve.png", "confusion_matrix.png",
            "confusion_matrix_normalized.png"} <= set(files)

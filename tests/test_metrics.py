"""FastMetrics 验收：完美检测 / 空检测 / 手算 AP / IoU 阈值行为 / 多类均值"""

import numpy as np

from train.metrics import FastMetrics


def test_perfect_detections():
    """单 GT 完美检测：全部指标为 1"""
    m = FastMetrics(nc=1)
    m.update(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([0]),
             np.array([[0, 0, 10, 10]]), np.array([0]))
    r = m.compute()
    assert r["mAP@50"] == 1.0 and r["mAP@[.5:.95]"] == 1.0 and r["AR@100"] == 1.0
    assert r["P"] == 1.0 and r["R"] == 1.0
    assert r["images"] == 1 and r["instances"] == 1
    print("  完美检测全指标为 1")


def test_no_detections():
    """有 GT 无检测：全部指标为 0"""
    m = FastMetrics(nc=1)
    m.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64),
             np.array([[0, 0, 10, 10]]), np.array([0]))
    r = m.compute()
    assert r["mAP@50"] == 0.0 and r["mAP@[.5:.95]"] == 0.0 and r["AR@100"] == 0.0
    assert r["P"] == 0.0 and r["R"] == 0.0
    print("  空检测全指标为 0")


def test_tp_fp_hand_computed():
    """2 GT：1 TP + 1 FP -> AP50 = 51/101（101 点插值手算），AR = 0.5"""
    m = FastMetrics(nc=1)
    # 检测 1 完美命中 GT1，检测 2 完全脱靶（分数相同，TP 在前保持顺序稳定）
    m.update(
        np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), np.array([0.9, 0.9]), np.array([0, 0]),
        np.array([[0, 0, 10, 10], [0, 0, 20, 20]]), np.array([0, 0]),
    )
    r = m.compute()
    # recall=[0.5,0.5], precision=[1.0,0.5] -> r<=0.5 处 precision 1.0（51/101 个点），r>0.5 处 0
    assert abs(r["mAP@50"] - 51 / 101) < 1e-3, r["mAP@50"]
    assert abs(r["AR@100"] - 0.5) < 1e-3
    # max-F1 在首点（TP 在前）：f1=[0.667, 0.5] -> P=1.0, R=0.5
    assert abs(r["P"] - 1.0) < 1e-3, r["P"]
    assert abs(r["R"] - 0.5) < 1e-3, r["R"]
    print(f"  手算 AP50 = {r['mAP@50']:.4f} (=51/101), P={r['P']:.2f}, R={r['R']:.2f}")


def test_iou_threshold_behavior():
    """IoU=0.6 的检测：AP50=1；AP@[.5:.95] 只在 0.5/0.55 阈值命中 -> 0.2"""
    m = FastMetrics(nc=1)
    w = np.sqrt(100 / 0.6)  # 使 IoU 恰为 0.6
    m.update(np.array([[0, 0, w, w]]), np.array([0.9]), np.array([0]),
             np.array([[0, 0, 10, 10]]), np.array([0]))
    r = m.compute()
    assert r["mAP@50"] == 1.0, r["mAP@50"]
    assert abs(r["mAP@[.5:.95]"] - 0.2) < 1e-3, r["mAP@[.5:.95]"]
    assert r["P"] == 1.0 and r["R"] == 1.0  # IoU=0.6 在 0.5 阈值下仍为 TP
    print(f"  IoU=0.6: AP50=1.0, AP[.5:.95]={r['mAP@[.5:.95]']:.4f}")


def test_global_score_order_across_images():
    """跨图检测必须按分数全局排序累积 PR（COCOeval 口径）——图序拼接是错的

    图1: GT1；检测 [TP(0.5), FP(0.1)]；图2: GT2；检测 [FP(0.9), TP(0.8)]。
    全局分数序 = [FP.9, TP.8, TP.5, FP.1] -> recall [0,.5,1,1]、precision [0,.5,2/3,.5]
    -> 精度包络 [2/3,2/3,2/3,.5] -> AP50 = 2/3（图序累积会得到 0.752，回归闸门）。
    """
    m = FastMetrics(nc=1)
    m.update(np.array([[0, 0, 10, 10], [100, 100, 110, 110]]), np.array([0.5, 0.1]), np.array([0, 0]),
             np.array([[0, 0, 10, 10]]), np.array([0]))
    m.update(np.array([[100, 100, 110, 110], [0, 0, 10, 10]]), np.array([0.9, 0.8]), np.array([0, 0]),
             np.array([[0, 0, 10, 10]]), np.array([0]))
    r = m.compute()
    assert abs(r["mAP@50"] - 2 / 3) < 1e-3, r["mAP@50"]
    assert abs(r["P"] - 2 / 3) < 1e-3 and abs(r["R"] - 1.0) < 1e-3, (r["P"], r["R"])
    print(f"  跨图全局排序: AP50={r['mAP@50']:.4f} (=2/3), P={r['P']:.2f}, R={r['R']:.2f}")


def test_multi_class_mean():
    """类 0 完美、类 1 无检测 -> mAP50 = 0.5"""
    m = FastMetrics(nc=2)
    m.update(
        np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([0]),
        np.array([[0, 0, 10, 10], [0, 0, 20, 20]]), np.array([0, 1]),
    )
    r = m.compute()
    assert abs(r["mAP@50"] - 0.5) < 1e-3, r["mAP@50"]
    assert abs(r["mAP@[.5:.95]"] - 0.5) < 1e-3
    assert abs(r["AR@100"] - 0.5) < 1e-3
    assert abs(r["P"] - 0.5) < 1e-3 and abs(r["R"] - 0.5) < 1e-3  # 类 1 无检测 -> P=R=0
    print("  多类均值正确")

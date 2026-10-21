"""训练报告图：results.png / PR_curve.png / confusion_matrix(.normalized).png

matplotlib 是**可选依赖**：缺失时只告警并跳过报告图，绝不阻塞训练。报告口径：
- results.png：直接读 results.csv（训练/验证损失、mAP 曲线）；
- PR_curve.png：FastMetrics 的逐类分数与 tp/fp 记录，IoU=0.5 的 PR 曲线（检测/分割各一组）；
- confusion_matrix：conf 0.25 / IoU 0.45 的框匹配（官方同口径），行为 GT、列为预测，
  最后一行为/列为背景。

这些图是**训练产物**，与 `scripts/eval.py` 的 pycocotools 正式指标独立。
"""

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["write_training_report", "plot_results", "plot_pr_curves", "plot_confusion_matrix"]


def _pyplot():
    """延迟导入 matplotlib（Agg 后端）；缺失返回 None"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib is not installed — training report plots are skipped "
                       "(pip install matplotlib)")
        return None


def _read_results(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return [], []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _series(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def plot_results(csv_path, out_path):
    """results.csv -> 训练/验证曲线网格（缺失列自动跳过）"""
    plt = _pyplot()
    if plt is None:
        return False
    cols, rows = _read_results(csv_path)
    if not rows:
        return False
    candidates = [("box", "box loss"), ("seg", "seg loss"), ("cls", "cls loss"), ("l1", "l1 loss"),
                  ("sem", "sem loss"), ("loss", "total loss"), ("mAP", "box mAP50-95"),
                  ("mAP50", "box mAP50"), ("mAP@[.5:.95](M)", "mask mAP50-95"),
                  ("mAP50(M)", "mask mAP50"), ("val_box", "val box loss"), ("val_seg", "val seg loss")]
    present = [(k, label) for k, label in candidates if k in cols and np.isfinite(_series(rows, k)).any()]
    if not present:
        return False
    epochs = _series(rows, "epoch")
    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    for ax, (key, label) in zip(axes.ravel(), present):
        ax.plot(epochs, _series(rows, key), linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(present):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_pr_curves(curves_by_task, out_path):
    """{task: {class_id: (recall, precision, ap)}} -> 每任务一张 PR 子图"""
    plt = _pyplot()
    if plt is None or not curves_by_task:
        return False
    tasks = [(k, v) for k, v in curves_by_task.items() if v]
    if not tasks:
        return False
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2), squeeze=False)
    for ax, (task, curves) in zip(axes.ravel(), tasks):
        aps = []
        for cid, (recall, precision, ap) in curves.items():
            ax.plot(recall, precision, linewidth=0.9, alpha=0.35)
            aps.append(ap)
        m_ap = float(np.mean(aps)) if aps else 0.0
        ax.set_title(f"{task} PR (IoU 0.5, mAP50 {m_ap:.4f})", fontsize=10)
        ax.set_xlabel("recall", fontsize=8)
        ax.set_ylabel("precision", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_confusion_matrix(matrix, names, out_path, normalized=False):
    """混淆矩阵热图（行 = GT，列 = 预测，末行/列为背景）"""
    plt = _pyplot()
    if plt is None:
        return False
    m = np.asarray(matrix, dtype=np.float64)
    if normalized:
        row_sum = m.sum(axis=1, keepdims=True)
        m = m / np.maximum(row_sum, 1)
    labels = list(names) + ["background"]
    n = m.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, interpolation="nearest", cmap="Blues")
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("true", fontsize=9)
    ax.set_title("confusion matrix" + (" (normalized)" if normalized else ""), fontsize=10)
    step = max(1, n // 15)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def write_training_report(run_dir, results_csv, metrics, names, task):
    """训练结束时的全部报告图；metrics 为最后一次 FastMetrics（无验证时传 None）"""
    run_dir = Path(run_dir)
    ok = []
    if plot_results(results_csv, run_dir / "results.png"):
        ok.append("results.png")
    if metrics is not None:
        curves = {"Box": metrics.pr_curve(task="detect")}
        if task == "segment":
            curves["Mask"] = metrics.pr_curve(task="segment")
        if plot_pr_curves(curves, run_dir / "PR_curve.png"):
            ok.append("PR_curve.png")
        matrix = metrics.confusion_matrix()
        if plot_confusion_matrix(matrix, names, run_dir / "confusion_matrix.png"):
            ok.append("confusion_matrix.png")
        if plot_confusion_matrix(matrix, names, run_dir / "confusion_matrix_normalized.png", normalized=True):
            ok.append("confusion_matrix_normalized.png")
    if ok:
        logger.info(f"plots: {', '.join(ok)} -> {run_dir}")
    return ok

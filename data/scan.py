"""数据集扫描的共享类型与排版（COCO / YOLO 两种加载器共用，避免两处漂移）

data/coco.py 与 data/yolo.py 的扫描都产出 ScanStats/ScanResult 口径；
`└` 续行由 scan_summary 统一渲染（训练与评估共用同一排版）。
"""

from collections import namedtuple

__all__ = ["ScanStats", "ScanResult", "scan_summary"]

# 数据集扫描统计（训练/验证共用口径）
#   n_missing: 磁盘上不存在（已从图片清单剔除）；n_backgrounds: 无框图片
#   n_crowd_excluded: 被剔除的 iscrowd=1 标注数（仅 COCO 训练侧非 0；YOLO 恒 0）
ScanStats = namedtuple("ScanStats", "n_missing n_backgrounds n_crowd_excluded "
                                    "n_instances parse_time scan_time first_missing")

# 扫描结果：labels 仅训练侧非 None（build_labels=True / YOLO 训练集恒建）
ScanResult = namedtuple("ScanResult", "images labels annotations cat_id_to_idx names stats")


def scan_summary(ds, extra=""):
    """扫描行下的 `└` 续行（训练与评估共用同一排版，避免两处漂移）

    形如 `       └ 849949 instances · 80 categories · crowd 10052 excluded · parse 12.6s + scan 6.6s`；
    YOLO 无解析段（parse_time=0）时省略 `parse Xs + `。
    """
    crowd = f"crowd {ds.n_crowd_excluded} excluded · " if ds.n_crowd_excluded else ""
    parse = f"parse {ds.parse_time:.1f}s + " if ds.parse_time else ""
    return (f"       └ {ds.n_instances} instances · {ds.n_categories} categories · {crowd}"
            f"{parse}scan {ds.scan_time:.1f}s{extra}")

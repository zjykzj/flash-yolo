"""fmt_elapsed / fmt_rate 档位边界 + 进度条渲染（短条 10 宽、pct、节流）"""

import io

from utils.progress import ProgressBar, fmt_elapsed, fmt_rate


def test_fmt_elapsed_seconds():
    assert fmt_elapsed(0) == "0.0s"
    assert fmt_elapsed(45.2) == "45.2s"
    assert fmt_elapsed(59.9) == "59.9s"


def test_fmt_elapsed_minutes():
    assert fmt_elapsed(60) == "1 min 00 s"
    assert fmt_elapsed(266.9) == "4 min 27 s"
    assert fmt_elapsed(3599) == "59 min 59 s"


def test_fmt_elapsed_hours():
    assert fmt_elapsed(3600) == "1 h 00 min"
    assert fmt_elapsed(3661) == "1 h 01 min"


def test_progress_bar_render():
    """短条渲染：10 宽、计数、速度、耗时齐全"""
    buf = io.StringIO()
    bar = ProgressBar(10, desc="test", file=buf)
    bar.update(5, speed=2.0)
    out = buf.getvalue()
    assert "█" * 5 in out and "░" * 5 in out
    assert "5/10" in out and "2.0it/s" in out
    bar.close()
    print("  进度条渲染正常（10 宽短条）")


def test_fmt_rate_scaling():
    """<1000 与旧的 f"{speed:.1f}it/s" 逐字符一致；>=1000 走 K 前缀"""
    assert fmt_rate(2.0) == "2.0it/s"
    assert fmt_rate(0.4) == "0.4it/s"
    assert fmt_rate(999.9) == "999.9it/s"
    assert fmt_rate(18000) == "18.0Kit/s"
    assert fmt_rate(1200, unit="img") == "1.2Kimg/s"


def test_progress_bar_pct():
    """pct=True: 条前加百分比记号（数据集扫描行口径）"""
    buf = io.StringIO()
    bar = ProgressBar(10, desc="scan", file=buf, pct=True)
    bar.update(5, speed=2.0)
    assert "scan 50% [█████░░░░░] 5/10" in buf.getvalue()
    bar.close()


def test_progress_bar_throttle():
    """min_interval 节流：中途帧被丢弃，末帧（n == total）永远重画"""
    buf = io.StringIO()
    bar = ProgressBar(10, file=buf, min_interval=3600)  # 大到中途绝不会到点
    bar.update(1)  # 首帧必画（last_draw = -inf）
    bar.update(2)
    bar.update(9)
    assert buf.getvalue().count("\r") == 1, "中途帧应被节流"
    bar.update(10)  # 末帧
    assert buf.getvalue().count("\r") == 2
    bar.close()
    print("  节流与末帧重画正确")

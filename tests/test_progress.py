"""fmt_elapsed 档位边界 + 进度条渲染（短条 10 宽）"""

import io

from utils.progress import ProgressBar, fmt_elapsed


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

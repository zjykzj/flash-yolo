"""轻量进度条（tqdm 风格外观，零依赖，控制台专用）

用法:
    bar = ProgressBar(total, desc="val")
    for i in range(total):
        ... 处理 ...
        if (i + 1) % 100 == 0:
            bar.update(i + 1, speed=窗口瞬时速度)
    bar.update(total, speed)
    bar.close()

进度行用 \\r 刷新，不经过 logger（控制台 UI 元素，不落日志文件）。
"""

import sys
import time

__all__ = ["ProgressBar", "fmt_elapsed"]


def fmt_elapsed(seconds):
    """耗时人性化：<1min '45.2s' / <1h '4 min 27 s' / 其余 '1 h 05 min'（进度条与定格行统一）"""
    s = max(seconds, 0.0)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(round(s), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


class ProgressBar:
    """单行进度条：百分比条 + 计数 + 瞬时速度 + ETA"""

    def __init__(self, total, desc="", width=10, file=None):
        self.total = total
        self.desc = desc
        self.width = width
        self.file = file or sys.stdout
        self.start = time.monotonic()  # 单调时钟：不受墙钟跳变影响（WSL2 时钟同步会回拨 time.time()）
        self.last_len = 0

    def update(self, n, speed=None, desc=None):
        """更新进度

        Args:
            n: 已完成数
            speed: 瞬时速度（单位/秒），None 则不显示
            desc: 行前缀描述（如指标列），None 保持上次值；前缀随每次调用原地刷新
        """
        if desc is not None:
            self.desc = desc
        frac = min(n / self.total, 1.0) if self.total else 1.0
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        speed_str = f"{speed:.1f}it/s" if speed is not None else "----it/s"
        # 耗时单调递增显示（ultralytics 风格）；结束后自然停在总耗时，无需额外处理
        elapsed = time.monotonic() - self.start
        line = f"\r{self.desc} [{bar}] {n}/{self.total} · {speed_str} · {fmt_elapsed(elapsed)}"
        self.file.write(line + " " * max(0, self.last_len - len(line)))
        self.file.flush()
        self.last_len = len(line)

    def close(self, clear=False):
        """结束进度条

        clear=True: 清掉当前行（配合调用方随后打印定格记录行，避免视觉重复）
        否则仅换行。
        """
        if clear:
            self.file.write("\r" + " " * self.last_len + "\r")
        else:
            self.file.write("\n")
        self.file.flush()

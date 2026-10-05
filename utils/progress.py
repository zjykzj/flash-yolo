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

__all__ = ["ProgressBar"]


class ProgressBar:
    """单行进度条：百分比条 + 计数 + 瞬时速度 + ETA"""

    def __init__(self, total, desc="", width=30, file=None):
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
        # ETA 优先用瞬时速度（平均速度含数据加载等启动开销，早期严重失真）
        if speed is not None and speed > 0:
            eta = (self.total - n) / speed
        else:
            elapsed = time.monotonic() - self.start
            eta = (self.total - n) / (n / elapsed) if n > 0 else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(max(eta, 0)))
        line = f"\r{self.desc} [{bar}] {n}/{self.total} · {speed_str} · ETA {eta_str}"
        self.file.write(line + " " * max(0, self.last_len - len(line)))
        self.file.flush()
        self.last_len = len(line)

    def close(self):
        """结束进度条（换行）"""
        self.file.write("\n")
        self.file.flush()

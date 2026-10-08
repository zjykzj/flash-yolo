"""轻量进度条（tqdm 风格外观，零依赖，控制台专用）

用法:
    bar = ProgressBar(total, desc="val")
    for i in range(total):
        ... 处理 ...
        if (i + 1) % 100 == 0:
            bar.update(i + 1, speed=窗口瞬时速度)
    bar.update(total, speed)
    bar.close()

    # 数据集扫描（data/coco.py::scan_split）：min_interval 节流重绘（11.8 万次调用）、
    # start 把时钟起点提到 ann json 解析之前、pct=True 在条前加百分比记号


进度行用 \\r 刷新，不经过 logger（控制台 UI 元素，不落日志文件）。
行宽按终端宽度钳制（_fit_line）：折行后 \\r 回不到行首、逐帧刷新会留叠影，
超宽时优先截 desc（计数/速度/耗时后缀永远保留）。
"""

import shutil
import sys
import time

__all__ = ["ProgressBar", "fmt_elapsed", "fmt_rate"]


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


def fmt_rate(rate, unit="it"):
    """速率人性化：>=1000 加 K 前缀（数据集扫描 18.0Kit/s）

    <1000 时与历史输出逐字符一致（2.0it/s），因此训练/验证条改用本函数后外观零变化。
    """
    return f"{rate / 1000:.1f}K{unit}/s" if rate >= 1000 else f"{rate:.1f}{unit}/s"


def _fit_line(desc, suffix, width, back=12):
    """desc + suffix 钳制到单行（len <= width - 1）：超宽时 desc 右截断加 …，≤back 字符内退到词边界

    \\r 只能回到**当前物理行**行首——一旦折行，逐帧刷新就盖不干净、留叠影（实测带完整
    ann 路径的扫描行 180+ 字）。后缀（条/计数/速度/耗时）永远保留，先牺牲 desc。
    中文路径按字符数近似（显示列宽会略低估，极窄终端属已知边界）。
    """
    budget = width - 1 - len(suffix)
    if len(desc) <= budget:
        return desc + suffix
    if budget <= 1:
        return "…"[:max(budget, 0)] + suffix
    cut = desc[:budget - 1]
    sp = cut.rfind(" ")
    if 0 <= budget - 1 - sp <= back:
        cut = cut[:sp]
    return cut.rstrip() + "…" + suffix


def fmt_num(v, width=11, prec=4):
    """定宽数值文本：常规用定点（与历史输出逐字符一致）；超宽回退科学计数

    训练/验证损失在退化状态可能到 1e12 量级——定点会把 11 宽列撑破、串列错位；
    回退 2 位小数的科学计数（含 11 字符内）保证列永远对齐。
    """
    s = f"{v:{width}.{prec}f}"
    return s if len(s) <= width else f"{v:{width}.2e}"


class ProgressBar:
    """单行进度条：百分比条 + 计数 + 瞬时速度 + ETA"""

    def __init__(self, total, desc="", width=10, file=None, unit="it", start=None, min_interval=0.0, pct=False):
        self.total = total
        self.desc = desc
        self.width = width
        self.file = file or sys.stdout
        self.unit = unit
        self.pct = pct  # True: 条前加 "100% " 记号（数据集扫描行，对齐 ultralytics 观感）
        self.min_interval = min_interval  # >0 时节流重绘（数据集扫描 11.8 万次调用）
        self.last_draw = -float("inf")
        # 单调时钟：不受墙钟跳变影响（WSL2 时钟同步会回拨 time.time()）
        # start 非 None 时由调用方指定起点：数据集扫描条从 ann json 解析开始计时，
        # 最终帧 elapsed = 解析 + 扫描（解析期那段由静态提示行占位，见 data/coco.py）
        self.start = start if start is not None else time.monotonic()
        self.last_len = 0

    def update(self, n, speed=None, desc=None):
        """更新进度

        Args:
            n: 已完成数
            speed: 瞬时速度（单位/秒），None 则不显示
            desc: 行前缀描述（如指标列），None 保持上次值；前缀随每次调用原地刷新
        """
        if desc is not None:
            self.desc = desc  # 先落 desc：被节流跳过的帧不会让下次重绘拿到旧值
        now = time.monotonic()
        if self.min_interval and n < self.total and now - self.last_draw < self.min_interval:
            return  # 节流（末帧永远重画）
        self.last_draw = now
        frac = min(n / self.total, 1.0) if self.total else 1.0
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        speed_str = fmt_rate(speed, self.unit) if speed is not None else f"----{self.unit}/s"
        pct_str = f"{int(frac * 100)}% " if self.pct else ""
        # 耗时单调递增显示（ultralytics 风格）；结束后自然停在总耗时，无需额外处理
        elapsed = now - self.start
        suffix = f" {pct_str}[{bar}] {n}/{self.total} · {speed_str} · {fmt_elapsed(elapsed)}"
        # 按终端宽度钳制（COLUMNS/ioctl；非 TTY 回落 80）：折行后 \r 失效会留叠影。
        # 补空格（清上一帧残留）同样受宽度约束——终端中途变窄时补空格自己也不能触发折行
        width = shutil.get_terminal_size().columns
        line = "\r" + _fit_line(self.desc, suffix, width)
        pad = max(0, min(self.last_len, width - 1) - len(line))
        self.file.write(line + " " * pad)
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

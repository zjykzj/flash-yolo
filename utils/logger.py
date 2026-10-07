"""日志模块：控制台 + 文件双输出（stdlib logging 封装，零依赖）

用法（两种接入方式）:

    # ① 有 run 目录的脚本（train / eval / infer）：日志落进 run 目录，产物自包含
    setup_logging(to_file=False)                 # 控制台即刻可用，先不落文件
    ...
    attach_file_log(run_dir / "run.log")         # run 目录确定后挂载（含 resume 复用同一目录）

    # ② 没有 run 目录的脚本（bench_io / convert_weights / ...）：兜底文件
    setup_logging()                              # -> logs/<脚本名>_<时间戳>.log

设计约定:
    - 运行日志跟 run 目录走（`run.log`）：归档/对比/删除 run 就是一个目录；resume 追加同一文件，
      一次训练只有一份完整日志
    - 兜底：没有 run 目录的脚本走 `logs/<脚本名>_<时间戳>.log`，多程序并发天然隔离（无跨进程锁）
    - 多卡训练时仅 rank 0 落文件：setup_logging(to_file=(rank == 0))
    - 文件轮转 10MB x 5 份，防长跑写爆
    - 控制台纯消息 + 级别着色（无时间戳，与进度条等 UI 元素对齐；ultralytics 风格）；
      文件带完整时间戳前缀且永远纯文本
"""

import contextlib
import io
import logging
import logging.handlers  # RotatingFileHandler 所在子模块，需显式导入
import re
import sys
from datetime import datetime
from pathlib import Path

__all__ = ["setup_logging", "attach_file_log", "get_logger", "bold", "redirect_prints", "log_file_only"]

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# 控制台级别颜色（仅控制台 handler 使用）
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # 青
    logging.INFO: "\033[32m",  # 绿
    logging.WARNING: "\033[33m",  # 黄
    logging.ERROR: "\033[31m",  # 红
    logging.CRITICAL: "\033[35m",  # 紫
}
_RESET = "\033[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_FMT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"  # 文件格式（完整前缀）
_CONSOLE_FMT = "%(message)s"  # 控制台纯消息（与进度条从第 0 列对齐）
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def bold(text):
    """加粗文本（控制台显示用；文件 handler 会自动剥离 ANSI）"""
    return f"\033[1m{text}\033[0m"


class _ColoredFormatter(logging.Formatter):
    """控制台专用：按级别着色，其余与标准格式一致"""

    def format(self, record):
        msg = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{msg}{_RESET}" if color else msg


class _PlainFormatter(logging.Formatter):
    """文件专用：剥离消息里的 ANSI 颜色码，保证纯文本"""

    def format(self, record):
        return _ANSI_RE.sub("", super().format(record))


def _console_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColoredFormatter(_CONSOLE_FMT))
    return handler


def _file_handler(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(_PlainFormatter(_FMT, datefmt=_DATEFMT))
    return handler


def setup_logging(log_name=None, level="INFO", to_file=True):
    """配置 root logger（幂等：重复调用不叠加 handler）

    Args:
        log_name: 兜底文件名前缀，默认取调用脚本名（sys.argv[0].stem）
        level: 日志级别
        to_file: 是否先落兜底文件；有 run 目录的脚本传 False，稍后用 attach_file_log 挂到 run 目录
    """
    root = logging.getLogger()
    if root.handlers:  # 已配置过，幂等返回
        return
    root.setLevel(level)
    root.addHandler(_console_handler())
    if to_file:
        name = log_name or Path(sys.argv[0]).stem
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root.addHandler(_file_handler(_LOG_DIR / f"{name}_{stamp}.log"))


def _find_file_handler():
    """root 上当前的文件 handler（RotatingFileHandler）"""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return handler
    return None


def attach_file_log(path):
    """把文件日志切到 path（有 run 目录的脚本在 run 目录确定后调用）

    - 幂等：同一路径重复调用直接返回；换路径则先摘掉旧 handler 再挂新的
    - 若旧的是 setup_logging 建的兜底文件且内容为空，顺手删掉（不留空壳）
    - 未 setup_logging 的进程（如测试）会自动补一个控制台 handler
    - 落两行头部（run 目录 + 命令行）：挂载前只进了控制台的那几行由此补齐
    """
    path = Path(path).resolve()  # 头部/日志里统一显示绝对路径
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel("INFO")
        root.addHandler(_console_handler())
    old = _find_file_handler()
    if old is not None:
        if Path(old.baseFilename) == path:
            return path
        base = Path(old.baseFilename)
        root.removeHandler(old)
        old.close()
        if base.exists() and base.stat().st_size == 0:  # 空的兜底文件没有信息，别留在 logs/
            base.unlink(missing_ok=True)
    root.addHandler(_file_handler(path))
    log_file_only(f"run dir: {path.parent}", name=__name__)
    log_file_only(f"argv: {' '.join(sys.argv)}", name=__name__)
    return path


def get_logger(name):
    """带命名空间的 logger（建议传 __name__，日志可溯源到模块）"""
    return logging.getLogger(name)


def log_file_only(msg, name="flash_yolo"):
    """只写日志文件（控制台行已由进度条定格行承担时使用，避免双行）

    无文件 handler（如测试进程未 setup_logging）时静默跳过。
    """
    record = logging.LogRecord(name, logging.INFO, "", 0, msg, (), None)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.handle(record)
            break


@contextlib.contextmanager
def redirect_prints(logger, level=logging.DEBUG):
    """把范围内的 print 输出捕获为日志（压制第三方库的裸 print，如 pycocotools）

    注意：logger 的控制台 handler 绑定的是原始 stdout 对象，不受重定向影响，
    只有第三方库的 print 会被捕获。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    for line in buf.getvalue().splitlines():
        if line.strip():
            logger.log(level, line)

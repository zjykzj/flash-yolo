"""日志模块：控制台 + 文件双输出（stdlib logging 封装，零依赖）

用法（脚本里两行接入）:
    from utils.logger import get_logger, setup_logging
    setup_logging()                 # 文件自动命名 logs/<脚本名>_<时间戳>.log
    logger = get_logger(__name__)

设计约定:
    - 每次运行独立的时间戳文件，多程序并发天然隔离（不做跨进程锁）
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

__all__ = ["setup_logging", "get_logger", "bold", "redirect_prints", "log_file_only"]

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
        log_name: 文件名前缀，默认取调用脚本名（sys.argv[0].stem）
        level: 日志级别
        to_file: 是否写文件（多卡训练时非主进程传 False）
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

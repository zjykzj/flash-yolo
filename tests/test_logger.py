"""日志模块验收：参数预览行（log_params）格式 + attach_file_log 的 run.log 头部两行"""

import logging
from pathlib import Path

from utils.logger import log_params

ROOT = Path(__file__).resolve().parent.parent


def test_log_params_format(caplog):
    """`scripts/xxx.py: k=v, k=v`（仓库相对路径前缀；与 model/summary.py 同风格）"""
    with caplog.at_level(logging.INFO):
        log_params(logging.getLogger("t"), ROOT / "scripts" / "train.py",
                   data="coco-tiny", batch=4, dry=True)
    assert caplog.records[-1].getMessage() == "scripts/train.py: data=coco-tiny, batch=4, dry=True"

    with caplog.at_level(logging.INFO):  # 仓库外的脚本（被拷走）：退回文件名前缀
        log_params(logging.getLogger("t"), "/opt/other/tool.py", a=1)
    assert caplog.records[-1].getMessage() == "tool.py: a=1"
    print("  log_params 行格式正确")


def test_attach_file_log_headers(tmp_path):
    """run.log 头部两行（仅文件）：run dir + argv"""
    from utils.logger import attach_file_log

    root = logging.getLogger()
    before_handlers, before_level = list(root.handlers), root.level
    root.setLevel("INFO")  # 入口脚本都会 setup_logging()（INFO）；测试进程模拟同等配置
    attach_file_log(tmp_path / "run.log")
    try:
        text = (tmp_path / "run.log").read_text(encoding="utf-8")
        assert "run dir:" in text and "argv: " in text, text
        assert "[utils.logger]" in text  # 文件格式带 [name] 前缀
    finally:  # 复原 handler 与级别，避免污染其它测试
        for h in list(root.handlers):
            if h not in before_handlers:
                root.removeHandler(h)
                h.close()
        root.setLevel(before_level)
    print("  run.log 头部（run dir + argv）正确")

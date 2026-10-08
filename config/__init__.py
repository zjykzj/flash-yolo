"""config 包：项目元数据与默认配置

__version__ 为版本号单一事实源（scripts 通过 `from config import __version__` 引用）。
"""

__version__ = "0.2.0"

from config.defaults import *  # noqa: F401,F403  # 默认超参与 COCO 类别名

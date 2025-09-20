# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/20 14:46
@File    : gitutil.py
@Author  : zj
@Description: 
"""

from pathlib import Path
from subprocess import check_output

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''

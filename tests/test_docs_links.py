"""文档下载契约：公开文档不出现 release 直链与 tag 细节

权重获取的单一入口是 `scripts/download_weights.py`（--self 模式）；公开文档只允许
出现命令与该脚本常量，不允许出现 `releases/download/...` 直链（官方 ultralytics
资产链接除外）或权重 release 的 tag 名——升级 release 时只需改脚本，文档零改动。
"""

from pathlib import Path

import pytest

import scripts.download_weights as dw

ROOT = Path(__file__).resolve().parent.parent
DOC_ROOTS = (ROOT / "docs", ROOT / "README.md", ROOT / "README.zh-CN.md")


def _iter_doc_files():
    for root in DOC_ROOTS:
        if root.is_dir():
            yield from root.rglob("*.md")
        else:
            yield root


_DOC_ITEMS = [(p, p.read_text(encoding="utf-8")) for p in _iter_doc_files()]
_DOC_IDS = [str(p.relative_to(ROOT)) for p, _ in _DOC_ITEMS]


@pytest.mark.parametrize("path,text", _DOC_ITEMS, ids=_DOC_IDS)
def test_no_own_release_download_links(path, text):
    # 官方 ultralytics 资产链接是文档事实，允许；本仓库 release 直链禁止
    assert "zjykzj/flash-yolo/releases/download" not in text


@pytest.mark.parametrize("path,text", _DOC_ITEMS, ids=_DOC_IDS)
def test_no_release_tag_in_docs(path, text):
    # tag 名只活在脚本常量里；文档不含 weights-v1 / v1.0.0 资产口径
    assert dw.RELEASE_TAG not in text
    assert "releases/download/v1.0.0" not in text

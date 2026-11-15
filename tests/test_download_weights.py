"""scripts/download_weights.py: 本仓库 release 权重下载（--self）契约

覆盖：weights-v1 资产清单与稳定 URL、单份/全部下载、--self 与 --model 互斥、
下载失败（404 / 连接错误）的退出行为。网络调用全部 mock，不触网。
"""

import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from scripts import download_weights as dw


def test_release_constants():
    assert dw.RELEASE_TAG == "weights-v1"
    assert dw.RELEASE_REPO == "https://github.com/zjykzj/flash-yolo"
    assert "all" not in dw.RELEASE_ASSETS
    assert len(dw.RELEASE_ASSETS) == 8
    # 三任务齐全：检测 / 分割 / 姿态（yolo26n 与 flash-yolo 家族）+ v3 两版
    assert "flash-yolo-pose-e300-640" in dw.RELEASE_ASSETS
    assert "yolo26n-pose-from-scratch-e300-640" in dw.RELEASE_ASSETS


def test_official_url_dst_branches(tmp_path):
    # yolo26 .pt
    url, dst = dw.official_url_dst("yolo26n", "v8.4.0", tmp_path)
    assert url.endswith("/v8.4.0/yolo26n.pt") and dst == tmp_path / "yolo26n.pt"
    # darknet .weights
    url, dst = dw.official_url_dst("yolov3-tiny", "v8.4.0", tmp_path)
    assert url.startswith("https://pjreddie.com/") and dst == tmp_path / "yolov3-tiny.weights"
    # mobileclip2_b.ts（与官方 .pt 同源同 tag，目的地为 .ts）
    url, dst = dw.official_url_dst("mobileclip2_b", "v8.4.0", tmp_path)
    assert url.endswith("/v8.4.0/mobileclip2_b.ts") and dst == tmp_path / "mobileclip2_b.ts"


def test_self_url_is_stable():
    url = dw.self_url(dw.RELEASE_TAG, "flash-yolo-e300-640")
    assert url == ("https://github.com/zjykzj/flash-yolo/releases/download/"
                   "weights-v1/flash-yolo-e300-640.safetensors")


def test_self_single_download(tmp_path):
    logger = mock.Mock()
    with mock.patch.object(dw.urllib.request, "urlretrieve") as m:
        dw.download_self(logger, dw.RELEASE_TAG, "flash-yolo-e300-640", tmp_path)
    assert m.call_count == 1
    url, dst = m.call_args.args
    assert url.endswith("/weights-v1/flash-yolo-e300-640.safetensors")
    assert dst == tmp_path / "flash-yolo-e300-640.safetensors"


def test_self_all_downloads_every_asset(tmp_path):
    logger = mock.Mock()
    with mock.patch.object(dw.urllib.request, "urlretrieve") as m:
        dw.download_self_all(logger, dw.RELEASE_TAG, tmp_path)
    assert m.call_count == len(dw.RELEASE_ASSETS)
    names = [Path(call.args[1]).stem for call in m.call_args_list]
    assert names == list(dw.RELEASE_ASSETS)


def test_download_http_404_exits(tmp_path):
    logger = mock.Mock()
    err = urllib.error.HTTPError("url", 404, "Not Found", None, None)
    with mock.patch.object(dw.urllib.request, "urlretrieve", side_effect=err):
        with pytest.raises(SystemExit):
            dw.download_self(logger, dw.RELEASE_TAG, "missing-e300-640", tmp_path)
    assert logger.error.called


def test_parser_self_choices_and_defaults():
    parser = dw.build_parser()
    args = parser.parse_args([])
    assert args.model is None and args.self_name is None
    assert args.tag == dw.RELEASE_TAG
    assert parser.parse_args(["--self", "all"]).self_name == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["--self", "bogus-name"])


def test_parser_self_model_mutually_exclusive():
    parser = dw.build_parser()
    with pytest.raises(SystemExit):
        dw.check_args(parser, parser.parse_args(["--self", "all", "--model", "yolo26n"]))
    # 单独给出时均不报错
    dw.check_args(parser, parser.parse_args(["--self", "all"]))
    dw.check_args(parser, parser.parse_args(["--model", "yolo26n"]))

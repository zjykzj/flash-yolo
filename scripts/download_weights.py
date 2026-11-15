"""下载权重：官方权重（--model）与本仓库发布权重（--self，weights-v1 release）

官方模型文件许可边界见 README 的 License 一节，本仓库不直接分发，仅提供下载脚本
供用户自行获取（个人研究/数值对照用途）。yolov3-tiny 为 darknet 官方权重（pjreddie.com，
33.1 AP50 / 220 FPS 谱系），下载后需 scripts/convert_weights.py 转 safetensors。

本仓库从零训练的权重统一发布在 GitHub Release `weights-v1`（稳定 tag，URL 不随代码
版本变化，可直接 curl）；`--self` 一条命令下载，无需访问网页。RELEASE_ASSETS 是公开
资产清单，发布新权重时与 release 资产同步更新。

mobileclip2_b.ts（Apple MobileCLIP2-B 文本编码器 TorchScript，AMLR 研究用途限定）与
官方 .pt 同源托管在 ultralytics assets release；`--model mobileclip2_b` 下载后需
scripts/convert_mobileclip_ts.py 转 safetensors（YOLOE 文本模式使用）。

用法:
    python scripts/download_weights.py --model yolo26n
    python scripts/download_weights.py --model yolov3-tiny
    python scripts/download_weights.py --model mobileclip2_b
    python scripts/download_weights.py --model yolo26n --version v8.4.0 --dir weights
    python scripts/download_weights.py --self flash-yolo-e300-640
    python scripts/download_weights.py --self all
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录入 sys.path

from config import __repo__
from utils.logger import get_logger, log_params, setup_logging

BASE_URL = "https://github.com/ultralytics/assets/releases/download/{version}/{model}"
DARKNET_URL = "https://pjreddie.com/media/files/{model}.weights"

# 本仓库权重 release（资产清单与 weights-v1 的 assets 一一对应）
RELEASE_TAG = "weights-v1"
RELEASE_REPO = __repo__.removesuffix(".git")
RELEASE_ASSETS = (
    "flash-yolo-e300-640",
    "flash-yolo-seg-e300-640",
    "flash-yolo-pose-e300-640",
    "yolo26n-from-scratch-e300-640",
    "yolo26n-seg-from-scratch-e300-640",
    "yolo26n-pose-from-scratch-e300-640",
    "yolov3-tiny-e300-416",
    "yolov3-tiny-silu-e300-416",
)


def self_url(tag, name):
    """本仓库 release 资产的稳定下载 URL"""
    return f"{RELEASE_REPO}/releases/download/{tag}/{name}.safetensors"


def _download(logger, url, dst):
    logger.info(f"downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dst)
    except urllib.error.HTTPError as e:
        logger.error(f"download failed: HTTP {e.code} {e.reason} ({url})")
        if e.code == 404:
            logger.error("asset not found — check the release tag (--tag) and the asset name")
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error(f"download failed: {e.reason} ({url})")
        sys.exit(1)
    logger.info(f"saved -> {dst}")


def download_self(logger, tag, name, out_dir):
    """下载一份本仓库 release 权重到 out_dir"""
    _download(logger, self_url(tag, name), out_dir / f"{name}.safetensors")


def official_url_dst(model, version, out_dir):
    """官方/外部模式（--model）的 (url, dst)：yolo26 .pt、darknet .weights、mobileclip2_b.ts"""
    if model == "yolov3-tiny":
        return DARKNET_URL.format(model=model), out_dir / "yolov3-tiny.weights"
    if model == "mobileclip2_b":
        return f"{BASE_URL.format(version=version, model=model)}.ts", out_dir / "mobileclip2_b.ts"
    return f"{BASE_URL.format(version=version, model=model)}.pt", out_dir / f"{model}.pt"


def download_self_all(logger, tag, out_dir):
    """按 RELEASE_ASSETS 顺序下载全部本仓库权重"""
    for name in RELEASE_ASSETS:
        download_self(logger, tag, name, out_dir)


def check_args(parser, args):
    if args.self_name and args.model:
        parser.error("--self and --model are mutually exclusive")


def build_parser():
    parser = argparse.ArgumentParser(
        description="download weights: official ultralytics/darknet (--model) or "
                    "this repo's release assets (--self)")
    parser.add_argument("--model", default=None,
                        help="official weight name (yolo26n/s/m/l/x | "
                             "yolo26n/s/m/l/x-seg | yolo26n/s/m/l/x-pose | "
                             "yolo26n/s/m/l/x-objv1-seg | yolov3-tiny | "
                             "mobileclip2_b); defaults to yolo26n when --self "
                             "is not given")
    parser.add_argument("--self", dest="self_name", default=None,
                        choices=RELEASE_ASSETS + ("all",),
                        help="this repo's release weight name (see --help "
                             "choices; 'all' downloads every asset); "
                             "mutually exclusive with --model")
    parser.add_argument("--tag", default=RELEASE_TAG,
                        help="release tag for --self weights")
    parser.add_argument("--version", default="v8.4.0",
                        help="ultralytics release version tag (yolo26 .pt only)")
    parser.add_argument("--dir", default="weights", help="save directory")
    return parser


def main():
    setup_logging()
    logger = get_logger(__name__)
    parser = build_parser()
    args = parser.parse_args()
    check_args(parser, args)
    log_params(logger, __file__, model=args.model, self=args.self_name, tag=args.tag,
               dir=args.dir, version=args.version)

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_name == "all":
        download_self_all(logger, args.tag, out_dir)
    elif args.self_name:
        download_self(logger, args.tag, args.self_name, out_dir)
    else:
        model = args.model or "yolo26n"
        url, dst = official_url_dst(model, args.version, out_dir)
        _download(logger, url, dst)
        if model == "yolov3-tiny":
            logger.info(f"next: python scripts/convert_weights.py --src {dst} "
                        f"--dst weights/yolov3-tiny.safetensors")
        elif model == "mobileclip2_b":
            logger.info(f"next: python scripts/convert_mobileclip_ts.py --input {dst} "
                        f"--output weights/mobileclip2_b.safetensors")


if __name__ == "__main__":
    main()

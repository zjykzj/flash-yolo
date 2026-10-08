"""下载官方权重（YOLO26 的 .pt / YOLOv3-tiny 的 darknet .weights）

注意：官方模型文件许可边界见 README 的 License 一节，本仓库不直接分发，仅提供下载脚本
供用户自行获取（个人研究/数值对照用途）。yolov3-tiny 为 darknet 官方权重（pjreddie.com，
33.1 AP50 / 220 FPS 谱系），下载后需 scripts/convert_weights.py 转 safetensors。

用法:
    python scripts/download_weights.py --model yolo26n
    python scripts/download_weights.py --model yolov3-tiny
    python scripts/download_weights.py --model yolo26n --version v8.4.0 --dir weights
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录入 sys.path

from utils.logger import get_logger, setup_logging

BASE_URL = "https://github.com/ultralytics/assets/releases/download/{version}/{model}.pt"
DARKNET_URL = "https://pjreddie.com/media/files/{model}.weights"


def main():
    setup_logging()
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description="download official weights (AGPL-3.0)")
    parser.add_argument("--model", default="yolo26n", help="model name (yolo26n/s/m/l/x | yolov3-tiny)")
    parser.add_argument("--version", default="v8.4.0", help="ultralytics release version tag (yolo26 .pt only)")
    parser.add_argument("--dir", default="weights", help="save directory")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "yolov3-tiny":
        url = DARKNET_URL.format(model=args.model)
        dst = out_dir / "yolov3-tiny.weights"
    else:
        url = BASE_URL.format(version=args.version, model=args.model)
        dst = out_dir / f"{args.model}.pt"
    logger.info(f"downloading: {url}")
    urllib.request.urlretrieve(url, dst)
    logger.info(f"saved -> {dst}")
    if args.model == "yolov3-tiny":
        logger.info(f"next: python scripts/convert_weights.py --src {dst} "
                    f"--dst weights/yolov3-tiny.safetensors")


if __name__ == "__main__":
    main()

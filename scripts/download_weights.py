"""下载官方 YOLO26 权重

注意：官方模型文件为 AGPL-3.0 许可（见 docs/license.md），本仓库不直接分发，
仅提供下载脚本供用户自行获取（个人研究/数值对照用途）。

用法:
    python scripts/download_weights.py --model yolo26n
    python scripts/download_weights.py --model yolo26n --version v8.4.0 --dir weights
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录入 sys.path

from utils.logger import get_logger, setup_logging

BASE_URL = "https://github.com/ultralytics/assets/releases/download/{version}/{model}.pt"


def main():
    setup_logging()
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description="download official YOLO26 weights (AGPL-3.0)")
    parser.add_argument("--model", default="yolo26n", help="model name (yolo26n/s/m/l/x)")
    parser.add_argument("--version", default="v8.4.0", help="ultralytics release version tag")
    parser.add_argument("--dir", default="weights", help="save directory")
    args = parser.parse_args()

    url = BASE_URL.format(version=args.version, model=args.model)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{args.model}.pt"
    logger.info(f"downloading: {url}")
    urllib.request.urlretrieve(url, dst)
    logger.info(f"saved -> {dst}")


if __name__ == "__main__":
    main()

"""从 COCO 抽取子集（coco-tiny）：子集 json + 图片硬链接 + 描述符 + dataflow-cv 转换命令

用途：本机分钟级验证（启动/训练/评估），并作为 YOLO 格式转换的输入：
dataflow-cv 的 `analyse sample/split` 不支持 COCO，所以子集抽取在这里做；
转换用它的 CLI（`dataflow-cv convert coco2yolo <json> <out>`）——注意它只写 labels/ 与
classes.txt、不拷图片，且标签按 COCO image id 命名（本工程 YOLO 读取器有数字 stem 回退，可直接读）。

用法:
    python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \\
        --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

from utils.logger import get_logger, log_params, setup_logging

setup_logging(to_file=False)
logger = get_logger(__name__)


def _subset(ann_file, n, seed):
    """按文件名排序 + 种子随机抽 n 张 -> (子集 ann dict, [file_name, ...])（保留原始 id）"""
    anns = json.loads(Path(ann_file).read_text())
    names = sorted(img["file_name"] for img in anns["images"])
    keep = set(random.Random(seed).sample(names, min(n, len(names))))
    images = [im for im in anns["images"] if im["file_name"] in keep]
    ids = {im["id"] for im in images}
    annotations = [a for a in anns["annotations"] if a["image_id"] in ids]
    return {"images": images, "annotations": annotations, "categories": anns["categories"]}, sorted(keep)


def _link(src, dst, link):
    """硬链接（同盘零成本）；跨设备/不支持时退回复制"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _descriptor_text(fmt, root, names, paths):
    """生成描述符 yaml 文本（相对角色路径，便于整体搬迁）"""
    lines = [f"# 生成物：{root} 的数据集描述符（{fmt}）——不经手改，重生成请重跑 make_coco_subset.py",
             f"format: {fmt}", f"path: {root}", "names:"]
    lines += [f"  - {n}" for n in names]
    for role, block in paths.items():
        lines.append(f"{role}:")
        for k, v in block.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="extract a tiny COCO subset + descriptors + conversion commands")
    parser.add_argument("--src", required=True, help="source COCO root (annotations/ + images/)")
    parser.add_argument("--dst", required=True, help="output root for the subset (e.g. .../coco-tiny)")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-images", action="store_true",
                        help="hard-link images instead of copying (same filesystem: free)")
    parser.add_argument("--emit-descriptors", action="store_true",
                        help="write coco/yolo descriptors + copy the coco one into config/datasets/local/")
    parser.add_argument("--yolo-dst", default=None, help="target root for the YOLO conversion (default <dst>-yolo)")
    args = parser.parse_args()
    log_params(logger, __file__, src=args.src, dst=args.dst, seed=args.seed,
               **{"n-train": args.n_train, "n-val": args.n_val, "link-images": args.link_images})

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    yolo_dst = Path(args.yolo_dst).resolve() if args.yolo_dst else Path(str(dst) + "-yolo")
    if not (src / "annotations").is_dir():
        parser.error(f"{src}: not a COCO root (no annotations/)")

    names = None
    for split, n in (("train2017", args.n_train), ("val2017", args.n_val)):
        subset, files = _subset(src / "annotations" / f"instances_{split}.json", n, args.seed)
        if names is None:
            names = [c["name"] for c in sorted(subset["categories"], key=lambda c: c["id"])]
        (dst / "annotations").mkdir(parents=True, exist_ok=True)
        (dst / "annotations" / f"instances_{split}.json").write_text(json.dumps(subset))
        for name in files:
            _link(src / "images" / split / name, dst / "images" / split / name, args.link_images)
        logger.info(f"{split}: {len(files)} images · {len(subset['annotations'])} annotations -> {dst}")

    if args.emit_descriptors:
        (dst / "coco-tiny.yaml").write_text(_descriptor_text("coco", dst, names, {
            "train": {"images": "images/train2017", "ann": "annotations/instances_train2017.json"},
            "val": {"images": "images/val2017", "ann": "annotations/instances_val2017.json"},
        }))
        (dst / "coco-tiny-yolo.yaml").write_text(_descriptor_text("yolo", yolo_dst, names, {
            "train": {"images": "train/images"},  # labels 按 images -> labels 镜像（转换产物布局）
            "val": {"images": "val/images"},
        }))
        local = ROOT / "config" / "datasets" / "local"
        local.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst / "coco-tiny.yaml", local / "coco-tiny.yaml")
        shutil.copy2(dst / "coco-tiny-yolo.yaml", local / "coco-tiny-yolo.yaml")
        logger.info(f"descriptors -> {dst}/coco-tiny[-yolo].yaml · {local}/ (gitignored，可 --data coco-tiny 寻址)")

        logger.info("")
        logger.info("转 YOLO 格式（dataflow-cv CLI；只写 labels/ 与 classes.txt，不拷图片）：")
        for split in ("train2017", "val2017"):
            role = "train" if split.startswith("train") else "val"
            logger.info(f"  dataflow-cv convert coco2yolo {dst}/annotations/instances_{split}.json "
                        f"{yolo_dst / role} --log-dir {dst}/logs")
        logger.info("转换后把空 images/ 换成指向 COCO 子集图片的符号链接：")
        for split, role in (("train2017", "train"), ("val2017", "val")):
            logger.info(f"  rmdir {yolo_dst / role}/images && "
                        f"ln -s {dst}/images/{split} {yolo_dst / role}/images")
        logger.info(f"然后：python scripts/train.py --data {local / 'coco-tiny-yolo.yaml'}")


if __name__ == "__main__":
    main()

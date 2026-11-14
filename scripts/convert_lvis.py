"""LVIS v1 annotation JSON -> compat COCO JSON

Official LVIS annotations are COCO-like but omit two fields this repo's loader needs:

- images have no ``file_name`` (only ``coco_url``)
- annotations have no ``iscrowd`` (LVIS has no crowd concept)

The raw category names also use underscores, while ``config/datasets/lvis.yaml`` uses the
official first-synonym prompt names. This script writes the compat shape:

1. images: ``file_name`` = basename of ``coco_url`` (fallback ``%012d.jpg``)
2. annotations: ``iscrowd = 0``
3. categories: renumbered ``0..nc-1`` in sorted-id order, renamed to the descriptor names
4. image-level ``neg_category_ids`` / ``not_exhaustive_category_ids`` renumbered with the same map

``--minival-ref`` optionally keeps only the images listed in a reference file (an LVIS/COCO
JSON with ``images``, or a text file with one image id / file name per line). This lets a
minival compat be derived from the val annotation without a second source file.

Usage:
    python scripts/convert_lvis.py --src annotations/lvis_v1_train.json \\
        --dst annotations/lvis_v1_train_compat.json
    python scripts/convert_lvis.py --src annotations/lvis_v1_val.json \\
        --dst annotations/lvis_v1_minival_compat.json \\
        --minival-ref annotations/lvis_v1_minival.json
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

from config.datasets import load_names
from utils.logger import get_logger, log_params, setup_logging

setup_logging()
logger = get_logger(__name__)

DEFAULT_NAMES = ROOT / "config/datasets/lvis.yaml"
IMAGE_ID_KEYS = ("neg_category_ids", "not_exhaustive_category_ids")


def _resolve_names(names):
    """descriptor path / {id: name} / list -> 0..nc-1 的 name list"""
    if names is None:
        names = DEFAULT_NAMES
    if isinstance(names, (str, Path)):
        mapping = load_names(names)
        names = [mapping[i] for i in range(len(mapping))]
    elif isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    else:
        names = list(names)
    if not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("names must be a non-empty list of strings")
    return names


def _image_file_name(img):
    """compat file_name: existing > coco_url basename > COCO %012d.jpg fallback"""
    name = img.get("file_name")
    if name:
        return name
    url = img.get("coco_url")
    if url:
        name = Path(urlparse(url).path).name
        if name:
            return name
    return f"{int(img['id']):012d}.jpg"


def _read_minival_ref(path):
    """reference -> (image ids, file names, reference images, reference annotations)

    JSON references can materialize images missing from the source (LVIS minival has 191
    background images that are absent from the LVIS val annotations); text references only
    filter, because they carry no image metadata.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path}: empty minival reference")
    if text.startswith("{"):
        data = json.loads(text)
        images = data.get("images")
        if not images:
            raise ValueError(f"{path}: minival reference JSON has no 'images'")
        return ({int(img["id"]) for img in images}, set(), images,
                data.get("annotations", []))
    ids, names = set(), set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.isdigit():
            ids.add(int(line))
        else:
            names.add(Path(line).name)
    if not ids and not names:
        raise ValueError(f"{path}: minival reference has no image ids or file names")
    return ids, names, None, None


def _filter_minival(data, ref):
    """keep source images in the reference; append reference-only images/annotations"""
    ids, names, ref_images, ref_annotations = _read_minival_ref(ref)
    source_kept = [img for img in data["images"]
                   if int(img["id"]) in ids or img["file_name"] in names]
    if not source_kept:
        raise ValueError(f"{ref}: minival reference matches none of the source images")
    kept_ids = {int(img["id"]) for img in source_kept}
    annotations = [a for a in data["annotations"] if int(a["image_id"]) in kept_ids]
    if ref_images is not None:
        missing = [img for img in ref_images if int(img["id"]) not in kept_ids]
        missing_ids = {int(img["id"]) for img in missing}
        for img in missing:
            img = dict(img)
            img["file_name"] = _image_file_name(img)
            source_kept.append(img)
        annotations.extend(dict(a) for a in ref_annotations
                           if int(a["image_id"]) in missing_ids)
    data["images"] = source_kept
    data["annotations"] = annotations


def _remap_categories(data, names):
    """sorted-id categories -> ids 0..nc-1 with descriptor names; return old_id -> new_id"""
    categories = sorted(data["categories"], key=lambda c: c["id"])
    if len(categories) != len(names):
        raise ValueError(f"descriptor defines {len(names)} classes but the annotation file has "
                         f"{len(categories)} — `--names-from` must match the LVIS categories")
    cat_map = {}
    for new_id, cat in enumerate(categories):
        old_id = cat["id"]
        if old_id in cat_map:
            raise ValueError(f"duplicate category id {old_id} in {data.get('info', {}).get('description', 'annotation')}")
        cat_map[old_id] = new_id
        cat["id"] = new_id
        cat["name"] = names[new_id]
    data["categories"] = categories
    return cat_map


def _remap_annotations(annotations, cat_map):
    for ann in annotations:
        old_id = ann["category_id"]
        if old_id not in cat_map:
            raise ValueError(f"annotation {ann.get('id')} uses unknown category_id {old_id}")
        ann["category_id"] = cat_map[old_id]
        ann.setdefault("iscrowd", 0)


def _remap_image_fields(images, cat_map):
    for img in images:
        for key in IMAGE_ID_KEYS:
            if key not in img or img[key] is None:
                continue
            mapped = []
            for old_id in img[key]:
                if old_id not in cat_map:
                    raise ValueError(f"image {img.get('id')} {key} uses unknown category_id {old_id}")
                mapped.append(cat_map[old_id])
            img[key] = mapped


def convert(src, dst, names=None, minival_ref=None):
    """Convert one LVIS annotation JSON; return image / annotation / category counts."""
    src, dst = Path(src), Path(dst)
    data = json.loads(src.read_text(encoding="utf-8"))
    for img in data["images"]:
        img["file_name"] = _image_file_name(img)
    if minival_ref is not None:
        _filter_minival(data, minival_ref)

    names = _resolve_names(names)
    cat_map = _remap_categories(data, names)
    _remap_annotations(data["annotations"], cat_map)
    _remap_image_fields(data["images"], cat_map)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(dst)
    return {"images": len(data["images"]), "annotations": len(data["annotations"]),
            "categories": len(data["categories"])}


def main():
    parser = argparse.ArgumentParser(description="LVIS annotation JSON -> compat COCO JSON")
    parser.add_argument("--src", required=True, help="raw LVIS annotation json")
    parser.add_argument("--dst", required=True, help="output compat json")
    parser.add_argument("--names-from", default=str(DEFAULT_NAMES),
                        help="descriptor yaml whose names define class order (default: lvis.yaml)")
    parser.add_argument("--minival-ref", default=None,
                        help="optional LVIS/COCO json or image-id/file-name list; keep only those images")
    args = parser.parse_args()
    log_params(logger, __file__, src=args.src, dst=args.dst, names_from=args.names_from,
               minival_ref=args.minival_ref)
    stats = convert(args.src, args.dst, names=args.names_from, minival_ref=args.minival_ref)
    logger.info(f"converted {stats['images']} images · {stats['annotations']} annotations · "
                f"{stats['categories']} categories -> {args.dst}")


if __name__ == "__main__":
    main()

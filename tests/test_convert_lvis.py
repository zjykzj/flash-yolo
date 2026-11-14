"""scripts/convert_lvis.py: raw LVIS -> compat COCO contracts

Covers the four compat operations (file_name / iscrowd / category remap / image-level id
remap), the optional minival filter (JSON and text reference), and the two failure modes
(unknown category id, descriptor class-count mismatch).
"""

import json

import pytest

from data.coco import parse_coco
from scripts.convert_lvis import convert


def _raw_lvis(tmp_path):
    """3 classes (ids deliberately unsorted), 2 images, 2 annotations."""
    data = {
        "info": {"description": "synthetic LVIS"},
        "licenses": [],
        "images": [
            {"id": 11, "coco_url": "http://images.cocodataset.org/val2017/000000000011.jpg",
             "width": 64, "height": 64, "neg_category_ids": [2],
             "not_exhaustive_category_ids": [3]},
            {"id": 12, "width": 64, "height": 64, "neg_category_ids": [1]},
        ],
        "annotations": [
            {"id": 1, "image_id": 11, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100,
             "segmentation": []},
            {"id": 2, "image_id": 12, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400},
        ],
        "categories": [
            {"id": 2, "name": "cat_b", "frequency": "c", "synonyms": ["cat_b"]},
            {"id": 1, "name": "cat_a", "frequency": "r", "synonyms": ["cat_a"]},
            {"id": 3, "name": "cat_c", "frequency": "f", "synonyms": ["cat_c"]},
        ],
    }
    src = tmp_path / "raw.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    return src


NAMES = ["cat a", "cat b", "cat c"]


def _run(tmp_path, src, **kwargs):
    dst = tmp_path / "compat.json"
    stats = convert(src, dst, names=NAMES, **kwargs)
    return dst, json.loads(dst.read_text(encoding="utf-8")), stats


def test_convert_adds_compat_fields_and_remaps(tmp_path):
    src = _raw_lvis(tmp_path)
    dst, data, stats = _run(tmp_path, src)

    assert stats == {"images": 2, "annotations": 2, "categories": 3}
    assert [(c["id"], c["name"], c["frequency"]) for c in data["categories"]] == [
        (0, "cat a", "r"), (1, "cat b", "c"), (2, "cat c", "f")]
    assert [a["category_id"] for a in data["annotations"]] == [1, 2]
    assert all(a["iscrowd"] == 0 for a in data["annotations"])
    assert data["images"][0]["file_name"] == "000000000011.jpg"
    assert data["images"][1]["file_name"] == "000000000012.jpg"
    assert data["images"][0]["neg_category_ids"] == [1]
    assert data["images"][0]["not_exhaustive_category_ids"] == [2]
    assert data["images"][1]["neg_category_ids"] == [0]

    # The loader's own reconciliation: sorted ids -> 0..nc-1 names must match the descriptor.
    _, _, cat_id_to_idx, names = parse_coco(dst)
    assert cat_id_to_idx == {0: 0, 1: 1, 2: 2}
    assert names == {0: "cat a", 1: "cat b", 2: "cat c"}


def test_names_from_descriptor_path(tmp_path):
    src = _raw_lvis(tmp_path)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("names:\n  - cat a\n  - cat b\n  - cat c\n", encoding="utf-8")
    dst = tmp_path / "compat.json"
    convert(src, dst, names=descriptor)
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert [c["name"] for c in data["categories"]] == NAMES


def test_minival_ref_json_filters_images_and_annotations(tmp_path):
    src = _raw_lvis(tmp_path)
    ref = tmp_path / "minival.json"
    ref.write_text(json.dumps({"images": [{"id": 11}]}), encoding="utf-8")
    dst, data, stats = _run(tmp_path, src, minival_ref=ref)

    assert stats == {"images": 1, "annotations": 1, "categories": 3}
    assert [img["id"] for img in data["images"]] == [11]
    assert [a["id"] for a in data["annotations"]] == [1]
    # Categories stay complete so the class-id map is independent of the subset.
    assert [c["id"] for c in data["categories"]] == [0, 1, 2]


def test_minival_ref_text_filters_by_file_name(tmp_path):
    src = _raw_lvis(tmp_path)
    ref = tmp_path / "minival.txt"
    ref.write_text("# one file name\n000000000011.jpg\n", encoding="utf-8")
    _, data, stats = _run(tmp_path, src, minival_ref=ref)
    assert stats["images"] == 1
    assert data["images"][0]["id"] == 11


def test_minival_ref_materializes_missing_reference_images(tmp_path):
    src = _raw_lvis(tmp_path)
    ref = tmp_path / "minival.json"
    ref.write_text(json.dumps({
        "images": [
            {"id": 11, "coco_url": "http://images.cocodataset.org/val2017/000000000011.jpg"},
            {"id": 13, "coco_url": "http://images.cocodataset.org/val2017/000000000013.jpg",
             "neg_category_ids": [1]},
        ],
        "annotations": [{"id": 3, "image_id": 13, "category_id": 1,
                         "bbox": [0, 0, 5, 5], "area": 25}],
    }), encoding="utf-8")
    _, data, stats = _run(tmp_path, src, minival_ref=ref)
    assert stats["images"] == 2
    assert [img["id"] for img in data["images"]] == [11, 13]
    assert [a["id"] for a in data["annotations"]] == [1, 3]
    assert data["annotations"][1]["category_id"] == 0
    assert data["annotations"][1]["iscrowd"] == 0
    assert data["images"][1]["file_name"] == "000000000013.jpg"
    assert data["images"][1]["neg_category_ids"] == [0]


def test_unknown_annotation_category_raises(tmp_path):
    src = _raw_lvis(tmp_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    data["annotations"][0]["category_id"] = 99
    src.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown category_id 99"):
        convert(src, tmp_path / "compat.json", names=NAMES)


def test_descriptor_class_count_mismatch_raises(tmp_path):
    src = _raw_lvis(tmp_path)
    with pytest.raises(ValueError, match="descriptor defines 2 classes"):
        convert(src, tmp_path / "compat.json", names=["cat a", "cat b"])


def test_minival_ref_without_matches_raises(tmp_path):
    src = _raw_lvis(tmp_path)
    ref = tmp_path / "minival.json"
    ref.write_text(json.dumps({"images": [{"id": 999}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="matches none"):
        convert(src, tmp_path / "compat.json", names=NAMES, minival_ref=ref)

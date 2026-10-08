"""数据集描述符 schema：解析 / 校验 / 名字解析与 local 覆盖"""

import pytest

from config.datasets import load_dataset, load_names
from config.datasets.spec import normalize_names, parse_descriptor

COCO_LIKE = """
format: coco
path: {path}
names: [cat, dog]
train:
  images: images/train2017
  ann: annotations/instances_train2017.json
val:
  images: images/val2017
  ann: annotations/instances_val2017.json
"""


def _descriptor(tmp_path, text, name="ds.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_template_needs_path():
    """包内 coco.yaml 是模板：path 空 + 相对角色路径 -> actionable 报错（仓库不存机器路径）

    按路径直读模板（不按名字）：本机可能已有 config/datasets/local/coco.yaml 覆盖（gitignored）。
    """
    from config.datasets import DATASETS_DIR
    with pytest.raises(ValueError, match="path:"):
        load_dataset(str(DATASETS_DIR / "coco.yaml"))


def test_load_names_reads_template():
    """load_names 只读 names 字段，不受描述符完整性影响（可视化默认名仍可用）"""
    names = load_names("coco")
    assert len(names) == 80 and names[0] == "person" and names[79] == "toothbrush"


def test_parse_full_coco_descriptor(tmp_path):
    """完整描述符：角色路径相对 `path:`（相对 yaml 所在目录）解析为绝对路径"""
    spec = load_dataset(str(_descriptor(tmp_path, COCO_LIKE.format(path="."))))
    assert spec.name == "ds" and spec.format == "coco" and spec.nc == 2
    assert spec.root == tmp_path.resolve()
    train = spec.role("train")
    assert train.images == tmp_path / "images/train2017"
    assert train.ann == tmp_path / "annotations/instances_train2017.json"
    assert train.labels is None
    print("  描述符解析：format/roles/names/路径全部就位")


def test_names_forms():
    """names 列表与 {index: name} 映射等价；非法形态报错"""
    assert normalize_names(["a", "b"]) == {0: "a", 1: "b"}
    assert normalize_names({1: "b", 0: "a"}) == {0: "a", 1: "b"}  # 键序无关，归一化为 0..n-1
    with pytest.raises(ValueError, match="indices must be"):
        normalize_names({0: "a", 2: "b"})
    with pytest.raises(ValueError, match="keys must be integers"):
        normalize_names({"x": "a"})
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_names("nope")
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_names(["a", ""])
    print("  names 两种形态与非法键报错正确")


def test_unknown_name_and_missing_path(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("definitely-not-a-dataset")
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / "nope.yaml"))
    print("  未知名字 / 缺文件报错正确")


def test_directory_arg_gets_migration_hint(tmp_path):
    """--data 不再收数据集目录：报错要给出迁移提示"""
    with pytest.raises(ValueError, match="not a directory"):
        load_dataset(str(tmp_path))


def test_role_missing_lists_available(tmp_path):
    spec = load_dataset(str(_descriptor(tmp_path, COCO_LIKE.format(path="."))))
    with pytest.raises(ValueError, match="no 'test' role.*available"):
        spec.role("test")
    print("  缺失角色报错并列出 available")


def test_unknown_role_block_errors_and_scalar_warns(tmp_path, caplog):
    """dict 值且不是角色名 -> 报错（疑似旧式 split 名）；散字段 -> warning 不拦"""
    with pytest.raises(ValueError, match="unknown role"):
        parse_descriptor(_descriptor(tmp_path, "format: coco\npath: .\nnames: [a]\ntrain2017: {images: x}\n"))
    with caplog.at_level("WARNING"):
        parse_descriptor(_descriptor(tmp_path, COCO_LIKE.format(path=".") + "\nnote: hello\n"))
    assert "note" in caplog.text
    print("  未知角色块报错 / 散字段告警正确")


def test_format_required_and_unknown(tmp_path):
    with pytest.raises(ValueError, match="'format' must be one of"):
        parse_descriptor(_descriptor(tmp_path, "path: .\nnames: [a]\ntrain: {images: i}\n"))
    with pytest.raises(ValueError, match="'format' must be one of"):
        parse_descriptor(_descriptor(tmp_path, "format: voc\npath: .\nnames: [a]\ntrain: {images: i}\n"))
    print("  format 必填且枚举校验正确")


def test_coco_requires_ann_and_forbids_labels(tmp_path):
    with pytest.raises(ValueError, match="requires 'ann'"):
        parse_descriptor(_descriptor(tmp_path, "format: coco\npath: .\nnames: [a]\ntrain: {images: imgs}\n"))
    with pytest.raises(ValueError, match="only valid for format yolo"):
        parse_descriptor(_descriptor(tmp_path,
                                     "format: coco\npath: .\nnames: [a]\ntrain: {images: i, ann: a.json, labels: l}\n"))


def test_yolo_labels_mirror_and_explicit(tmp_path):
    p = _descriptor(tmp_path, """
format: yolo
path: .
names: [a, b]
train: {images: train/images}
val: {images: val/images, labels: mylabels}
""")
    spec = load_dataset(str(p))
    assert spec.role("train").labels == tmp_path / "train/labels"  # images 段镜像
    assert spec.role("val").labels == tmp_path / "mylabels"
    with pytest.raises(ValueError, match="only valid for format coco"):
        parse_descriptor(_descriptor(tmp_path, "format: yolo\npath: .\nnames: [a]\ntrain: {images: i, ann: x.json}\n"))
    with pytest.raises(ValueError, match="set `labels:` explicitly"):
        parse_descriptor(_descriptor(tmp_path, "format: yolo\npath: .\nnames: [a]\ntrain: {images: mydata}\n"))
    print("  yolo 标签目录镜像 / 显式 / 报错正确")


def test_empty_path_blocks_relative_role_paths(tmp_path):
    """path 为空时相对角色路径报错；全绝对路径合法"""
    with pytest.raises(ValueError, match="`path:` is empty"):
        parse_descriptor(_descriptor(tmp_path, "format: coco\nnames: [a]\ntrain: {images: imgs, ann: a.json}\n"))
    spec = parse_descriptor(_descriptor(
        tmp_path, f"format: coco\nnames: [a]\ntrain: {{images: {tmp_path}/i, ann: {tmp_path}/a.json}}\n"))
    assert spec.root is None and spec.role("train").ann == tmp_path / "a.json"
    print("  空 path + 相对路径报错、绝对路径放行")


def test_no_roles_defined(tmp_path):
    with pytest.raises(ValueError, match="no roles defined"):
        parse_descriptor(_descriptor(tmp_path, "format: coco\npath: .\nnames: [a]\n"))


def test_local_override_wins(monkeypatch, tmp_path):
    """config/datasets/local/<name>.yaml 优先于包内模板（本机机器路径的唯一落点）"""
    monkeypatch.setattr("config.datasets.spec.LOCAL_DATASETS_DIR", tmp_path)
    (tmp_path / "myds.yaml").write_text(COCO_LIKE.format(path=str(tmp_path)), encoding="utf-8")
    spec = load_dataset("myds")
    assert spec.name == "myds" and spec.source == (tmp_path / "myds.yaml").resolve()
    print("  local/ 覆盖优先于包内模板")

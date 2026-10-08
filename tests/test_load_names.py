"""数据集元数据加载器：类别名 yaml -> {class_id: name}（错误契约对齐 resolve_recipe）"""

import pytest
import yaml

from config.datasets import DATASETS_DIR, load_names
from model.build import YOLO26_CONFIG_PATH


def test_coco_names():
    """80 类、键恰为 0..79、首末项与类别名唯一性"""
    names = load_names("coco")
    assert len(names) == 80
    assert list(names) == list(range(80))
    assert names[0] == "person" and names[79] == "toothbrush"
    assert all(isinstance(v, str) and v for v in names.values())
    assert len(set(names.values())) == 80, "类别名应互不相同"
    print("  COCO 80 类加载正确：", names[0], "...", names[79])


def test_names_match_model_nc():
    """类别数与模型结构 yaml 的 nc 一致（模型与数据集元数据不脱节）"""
    with open(YOLO26_CONFIG_PATH) as f:
        nc = yaml.safe_load(f)["nc"]
    assert len(load_names("coco")) == nc, f"names 80 类 vs yolo26.yaml nc={nc}"
    print(f"  names 与 yolo26.yaml nc={nc} 一致")


def test_returned_dict_is_a_copy():
    """每次返回新 dict（缓存的是不可变 tuple）：改坏一次不影响下一次"""
    first = load_names("coco")
    first[0] = "MUTATED"
    assert load_names("coco")[0] == "person"
    print("  返回值是拷贝，缓存不被调用方污染")


def test_load_by_path():
    """显式 .yaml 路径直接读取（与 resolve_recipe 同一约定）"""
    assert load_names(str(DATASETS_DIR / "coco.yaml"))[0] == "person"
    print("  路径形式加载正确")


def test_unknown_dataset_raises():
    """未知名字 -> ValueError（列 available）；缺路径 -> FileNotFoundError"""
    with pytest.raises(ValueError, match="unknown dataset"):
        load_names("nope")
    with pytest.raises(FileNotFoundError):
        load_names("/no/such/dataset.yaml")
    print("  未知数据集 / 缺文件报错正确")


def test_malformed_yaml_raises(tmp_path):
    """`names` 必须是按类别 id 顺序的非空字符串列表"""
    p = tmp_path / "bad.yaml"
    p.write_text("names: not-a-list\n")
    with pytest.raises(ValueError, match="must be a non-empty list"):
        load_names(str(p))
    print("  畸形 yaml 报错正确")

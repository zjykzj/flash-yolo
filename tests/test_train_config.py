"""训练配置验收：默认配方 / official 配方分档合并 / 未知配方报错 / CLI 覆盖优先"""

from types import SimpleNamespace

import pytest

from config.defaults import TRAIN_CONFIG_PATH
from config.train import apply_cli, load_train_config


def test_default_recipe():
    """默认 = 通用训练配方（100 轮档，对齐 ultralytics 默认）"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    assert cfg.recipe == "default"
    assert cfg.epochs == 100 and cfg.lr0 == 0.01 and cfg.lrf == 0.01
    assert cfg.nbs == 64 and cfg.momentum == 0.937 and cfg.weight_decay == 0.0005
    assert cfg.box_gain == 7.5 and cfg.cls_gain == 0.5 and cfg.dfl_gain == 1.5
    assert cfg.mosaic == 1.0 and cfg.mixup == 0.0 and cfg.copy_paste == 0.0
    print("  默认配方(100 轮/通用增强)正确")


def test_official_recipe_n():
    """official + n：官方 COCO 段全量配方"""
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe="official", scale="n")
    assert cfg.recipe == "official"
    assert cfg.epochs == 245 and cfg.nbs == 128
    assert cfg.lr0 == 0.0054 and cfg.lrf == 0.0495 and cfg.momentum == 0.947
    assert cfg.weight_decay == 0.00064 and cfg.warmup_epochs == 0.98
    assert cfg.box_gain == 5.63 and cfg.cls_gain == 0.56 and cfg.dfl_gain == 9.04
    assert cfg.mosaic == 0.909 and cfg.copy_paste == 0.075 and cfg.aug_scale == 0.562
    print("  official n 档配方正确")


def test_official_recipe_per_scale():
    """official + s/m/l/x：档位增量覆盖，未列字段继承 base；epochs 与官方表一致"""
    expect = {"s": 70, "m": 80, "l": 60, "x": 40}
    for scale, epochs in expect.items():
        cfg = load_train_config(TRAIN_CONFIG_PATH, recipe="official", scale=scale)
        assert cfg.epochs == epochs, f"{scale}: {cfg.epochs} != {epochs}"
        assert cfg.lr0 == 0.00038 and cfg.lrf == 0.882
        assert cfg.momentum == 0.948 and cfg.weight_decay == 0.00027 and cfg.warmup_epochs == 0.99
        # 未列字段继承 base（n 档全量）
        assert cfg.box_gain == 5.63 and cfg.mosaic == 0.909 and cfg.nbs == 128
    print("  official s/m/l/x 分档配方正确（增量覆盖 + base 继承）")


def test_unknown_recipe_or_scale():
    """未知配方 / official 未覆盖的档位 -> 报错而非静默回退"""
    with pytest.raises(ValueError, match="unknown recipe"):
        load_train_config(TRAIN_CONFIG_PATH, recipe="nope")
    with pytest.raises(ValueError, match="scale"):
        load_train_config(TRAIN_CONFIG_PATH, recipe="official", scale="zz")
    print("  未知配方/档位报错正确")


def test_cli_override_wins():
    """优先序：recipe 覆盖基础值，CLI 再覆盖 recipe"""
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe="official", scale="n")
    apply_cli(cfg, SimpleNamespace(epochs=50))
    assert cfg.epochs == 50, "CLI 应覆盖 recipe"
    assert cfg.lr0 == 0.0054, "未提供的字段不应被 CLI 触碰"
    print("  CLI 覆盖优先序正确")

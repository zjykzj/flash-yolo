"""训练配置验收：内置 default / 配方文件分档合并 / 路径加载 / 未知配方报错 / CLI 覆盖优先"""

from types import SimpleNamespace

import pytest

from config.train_config import (TRAIN_CONFIG_PATH, RECIPES_DIR, apply_cli, load_train_config,
                                 resolve_recipe)

RECIPE = "yolo26-coco-ft"


def test_default_recipe():
    """default = 内置基线（train.yaml 基础值，100 轮档，对齐 ultralytics 默认）"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    assert cfg.recipe == "default"
    assert cfg.model == "yolo26"
    assert cfg.epochs == 100 and cfg.lr0 == 0.01 and cfg.lrf == 0.01
    assert cfg.nbs == 64 and cfg.momentum == 0.937 and cfg.weight_decay == 0.0005
    # released ultralytics code 的 MuSGD 默认；官方 recipes 按 ckpt 记录覆盖
    assert cfg.muon_w == 0.2 and cfg.sgd_w == 1.0
    assert cfg.box_gain == 7.5 and cfg.cls_gain == 0.5 and cfg.dfl_gain == 1.5 and cfg.obj_gain == 1.0
    assert cfg.mosaic == 1.0 and cfg.mixup == 0.0 and cfg.copy_paste == 0.0
    assert cfg.mask_ratio == 4 and cfg.overlap_mask is True
    print("  内置 default（100 轮/通用增强）正确")


def test_product_defaults_on():
    """训练产物默认开启（关掉等于放弃事后分析能力）"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    assert cfg.save_period == 20 and cfg.diag_interval == 50 and cfg.aug_samples == 8
    print("  产物默认值开启正确")


def test_recipe_file_n():
    """yolo26-coco-ft + n：官方 COCO 阶段全量配方"""
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe=RECIPE, scale="n")
    assert cfg.recipe == RECIPE
    assert cfg.epochs == 245 and cfg.nbs == 128
    assert cfg.lr0 == 0.0054 and cfg.lrf == 0.0495 and cfg.momentum == 0.947
    assert cfg.weight_decay == 0.00064 and cfg.warmup_epochs == 0.98
    assert cfg.box_gain == 5.63 and cfg.cls_gain == 0.56 and cfg.dfl_gain == 9.04
    assert cfg.mosaic == 0.909 and cfg.copy_paste == 0.075 and cfg.aug_scale == 0.562
    assert cfg.muon_w == 0.528 and cfg.sgd_w == 0.674 and cfg.topk == 8
    print("  yolo26-coco-ft n 档配方正确")


def test_seg_recipe():
    """yolo26-seg-coco-ft n 档：数值取自官方 yolo26n-seg.pt 的 train_args"""
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe="yolo26-seg-coco-ft", scale="n")
    assert cfg.model == "yolo26-seg"
    assert cfg.mask_ratio == 1 and cfg.overlap_mask is True
    assert cfg.copy_paste == 0.075 and cfg.copy_paste_mode == "flip"
    assert cfg.epochs == 245 and cfg.nbs == 128
    assert cfg.lr0 == 0.0054 and cfg.lrf == 0.05 and cfg.momentum == 0.937
    assert cfg.warmup_epochs == 1.0 and cfg.warmup_momentum == 0.658
    assert cfg.box_gain == 5.628 and cfg.cls_gain == 0.561 and cfg.dfl_gain == 9.039
    # n-seg ckpt 与检测表不同：mixup/degrees/shear/bgr 都是 0
    assert cfg.mixup == 0.0 and cfg.degrees == 0.0 and cfg.shear == 0.0 and cfg.bgr == 0.0
    assert cfg.muon_w == 0.528 and cfg.sgd_w == 0.674 and cfg.topk == 8
    assert cfg.prog_alpha_init == 1.0 and cfg.cls_w == 2.7371
    assert cfg.cutmix == 0.0 and cfg.perspective == 0.0
    print("  yolo26-seg-coco-ft n 档配方正确")


def test_seg_recipe_per_scale():
    """s/m/l/x：各自官方 seg ckpt 的 train_args（s 是另一提交的早期基线）"""
    s = load_train_config(TRAIN_CONFIG_PATH, recipe="yolo26-seg-coco-ft", scale="s")
    assert s.epochs == 50 and s.lr0 == 0.00125 and s.lrf == 0.5
    assert s.momentum == 0.937 and s.weight_decay == 0.0007
    assert s.topk == 10 and s.box_gain == 7.5 and s.cls_gain == 0.5 and s.dfl_gain == 6.0
    assert s.mixup == 0.05 and s.copy_paste == 0.15 and s.aug_scale == 0.9
    assert s.translate == 0.1 and s.fliplr == 0.5 and s.hsv_s == 0.7 and s.close_mosaic == 5
    assert s.prog_alpha_init == 1.0 and s.cls_w == 2.0 and s.cutmix == 0.0 and s.perspective == 0.0
    for scale, epochs, cp in (("m", 80, 0.304), ("l", 60, 0.404), ("x", 40, 0.404)):
        cfg = load_train_config(TRAIN_CONFIG_PATH, recipe="yolo26-seg-coco-ft", scale=scale)
        assert cfg.epochs == epochs and cfg.copy_paste == cp, f"{scale}: {cfg.epochs}/{cfg.copy_paste}"
        assert cfg.lr0 == 0.00038 and cfg.lrf == 0.882
        assert cfg.momentum == 0.948 and cfg.weight_decay == 0.00027
        assert cfg.topk == 5 and cfg.box_gain == 9.832 and cfg.cls_gain == 0.649 and cfg.dfl_gain == 0.958
        assert cfg.mixup == 0.427 and cfg.aug_scale == 0.95
        assert cfg.translate == 0.275 and cfg.fliplr == 0.304
        assert cfg.prog_alpha_init == 0.70518 and cfg.cls_w == 3.48357
        assert cfg.cutmix == (0.00082 if scale == "m" else 0.0)
        assert cfg.perspective == (0.00074 if scale == "m" else 0.0)
        assert cfg.mask_ratio == 1 and cfg.overlap_mask is True and cfg.copy_paste_mode == "flip"
    print("  yolo26-seg-coco-ft s/m/l/x 分档配方正确")


def test_recipe_file_per_scale():
    """s/m/l/x：档位增量覆盖，未列字段继承 base；epochs 与官方表一致"""
    expect = {"s": 70, "m": 80, "l": 60, "x": 40}
    for scale, epochs in expect.items():
        cfg = load_train_config(TRAIN_CONFIG_PATH, recipe=RECIPE, scale=scale)
        assert cfg.epochs == epochs, f"{scale}: {cfg.epochs} != {epochs}"
        assert cfg.lr0 == 0.00038 and cfg.lrf == 0.882
        assert cfg.momentum == 0.948 and cfg.weight_decay == 0.00027 and cfg.warmup_epochs == 0.99
        # 各档的 loss/aug/MuSGD 也不一样（官方表逐档给出）
        assert cfg.box_gain == 9.83 and cfg.cls_gain == 0.65 and cfg.dfl_gain == 0.96
        assert cfg.mosaic == 0.992 and cfg.fliplr == 0.304 and cfg.bgr == 0.0
        assert cfg.muon_w == 0.436 and cfg.sgd_w == 0.479 and cfg.topk == 5
        # 未列字段继承：配方 base（nbs）+ 本仓库 train.yaml 基础值（tal_beta/imgsz）
        assert cfg.nbs == 128 and cfg.tal_beta == 6.0 and cfg.imgsz == 640
    print("  yolo26-coco-ft s/m/l/x 分档配方正确（增量覆盖 + base 继承）")


def test_recipe_by_path():
    """--recipe 支持直接给 .yaml 路径"""
    path = RECIPES_DIR / f"{RECIPE}.yaml"
    assert resolve_recipe(RECIPE) == path
    assert resolve_recipe(str(path)) == path
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe=str(path), scale="n")
    assert cfg.epochs == 245 and cfg.muon_w == 0.528
    print("  配方路径加载正确")


def test_unknown_recipe_or_scale():
    """未知配方 / 未覆盖的档位 -> 报错而非静默回退"""
    with pytest.raises(ValueError, match="unknown recipe"):
        load_train_config(TRAIN_CONFIG_PATH, recipe="nope")
    with pytest.raises(FileNotFoundError):
        resolve_recipe("/no/such/recipe.yaml")
    with pytest.raises(ValueError, match="scale"):
        load_train_config(TRAIN_CONFIG_PATH, recipe=RECIPE, scale="zz")
    print("  未知配方/档位报错正确")


def test_cli_override_wins():
    """优先序：配方覆盖基础值，CLI 再覆盖配方"""
    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe=RECIPE, scale="n")
    apply_cli(cfg, SimpleNamespace(epochs=50))
    assert cfg.epochs == 50, "CLI 应覆盖配方"
    assert cfg.lr0 == 0.0054, "未提供的字段不应被 CLI 触碰"
    print("  CLI 覆盖优先序正确")


def test_data_field():
    """数据集走描述符：train.yaml 不写死（空串），--data 经 apply_cli 写入 cfg.data"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    assert cfg.data == "", "train.yaml 不应夹带任何数据集"
    assert not hasattr(cfg, "train_split"), "冒烟/调试切 split 已由描述符角色取代（train_split 已删）"
    apply_cli(cfg, SimpleNamespace(data="coco"))
    assert cfg.data == "coco"
    print("  数据集字段（data）接线正确")


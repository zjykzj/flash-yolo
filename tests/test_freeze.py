"""--freeze：按层冻结参数 / 冻结层 BN 保持 eval / 优化器只收可训练参数"""

from config.train_config import TrainConfig
from model.build import build_model
from train.optimizer import build_param_groups
from train.trainer import freeze_layers, set_frozen_bn_eval


def test_freeze_layers_params_and_bn():
    model = build_model("flash-yolo", None, 320)
    assert freeze_layers(model, 3) == 3
    for name, p in model.named_parameters():
        layer = int(name.split(".")[1])
        assert (not p.requires_grad) == (layer < 3), name
    bn0 = model.model[0].bn
    assert not bn0.training, "冻结层 BN 应保持 eval"
    model.train()  # 每个 epoch 会重新置 train
    assert bn0.training
    set_frozen_bn_eval(model, 3)
    assert not bn0.training, "重放后冻结层 BN 回到 eval"
    # 越界 N 自动收敛到模型层数
    assert freeze_layers(model, 10_000) == len(model.model)


def test_freeze_excluded_from_optimizer():
    model = build_model("flash-yolo", None, 320)
    freeze_layers(model, 3)
    groups = build_param_groups(model, TrainConfig(), model.model[-1])
    params = [p for g in groups for p in g["params"]]
    assert params and all(p.requires_grad for p in params)
    assert len(params) < sum(1 for _ in model.parameters())

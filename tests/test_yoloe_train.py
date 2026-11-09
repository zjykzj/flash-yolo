"""YOLOE-26 训练前向契约（P0：头部训练输出 + 损失分派 + PE fused 路径）

不依赖官方权重：结构从 yaml 构建、tpe 随机生成。覆盖：
- 训练模式双分支原始输出（boxes ltrb / scores logits / mask 系数 / proto）
- o2o detach 梯度隔离（backbone 梯度只经 o2m 回流）
- fused 文本布局（PE 产物）与未 fused 前向数值一致（BN 折进口径）
- pf 布局训练显式拒绝；PE fused 头走检测损失、文本布局走分割损失
"""

import copy
from pathlib import Path

import pytest
import torch

from model.build import build_model
from config.train_config import TrainConfig

NL = 3


def _build(prompt_free=False, nc=80, imgsz=640):
    return build_model("yoloe-26", "n", imgsz=imgsz, nc=nc, prompt_free=prompt_free)


def _tpe(k=5, embed=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, k, embed, generator=g)
    return t / t.norm(dim=-1, keepdim=True)


def test_train_mode_shapes_text():
    """文本布局训练输出：双分支 boxes/scores/系数 + 原型"""
    model = _build().train()
    head = model.model[-1]
    head.set_tpe(_tpe())
    head.end2end = True
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {"one2many", "one2one"}
    for key in ("one2many", "one2one"):
        b = out[key]
        assert b["boxes"].shape == (1, 4, 8400), f"{key} boxes: {b['boxes'].shape}"
        assert b["scores"].shape == (1, 5, 8400), f"{key} scores: {b['scores'].shape}"
        assert b["mask_coefficient"].shape == (1, 32, 8400), f"{key} coeff: {b['mask_coefficient'].shape}"
        assert b["proto"].shape == (1, 32, 160, 160), f"{key} proto: {b['proto'].shape}"
    # nc 随 prompt 数量动态变（与 forward_text 同口径）
    assert head.nc == 5


def test_o2o_feature_detach():
    """o2o 分支梯度隔离：backbone 梯度只经 o2m 回流"""
    model = _build().train()
    head = model.model[-1]
    head.set_tpe(_tpe())
    x = torch.zeros(1, 3, 640, 640)

    out = model(x)
    out["one2one"]["boxes"].sum().backward()
    assert model.model[0].conv.weight.grad is None, "o2o-only 损失不应回传 backbone"
    assert head.cv2[0][0].conv.weight.grad is None, "o2o-only 损失不应回传 o2m 头"
    assert head.one2one_cv2[0][0].conv.weight.grad is not None, "o2o 头自身应有梯度"

    model.zero_grad(set_to_none=True)
    out = model(x)
    out["one2many"]["boxes"].sum().backward()
    assert model.model[0].conv.weight.grad is not None, "o2m 损失应回传 backbone"
    assert head.one2one_cv2[0][0].conv.weight.grad is None, "detach 应阻断 o2m 损失进入 o2o 头"


def test_fused_train_matches_unfused_scores():
    """fuse_text 后 cv3 尾卷积直接出 logits（PE 训练路径）与未 fused 前向数值一致

    BN 在训练模式用批统计、fused 折叠的是 running stats——比较前把未 fused 的 BN 置
    eval（PE 训练同样冻结 BN 在 eval）。
    """
    model = _build().eval()
    feats = model.forward_feats(torch.randn(1, 3, 640, 640))
    tpe = _tpe()

    ref = copy.deepcopy(model)
    ref.train()
    ref_head = ref.model[-1]
    ref_head.set_tpe(tpe)
    for i in range(NL):
        ref_head.cv4[i].norm.eval()
        ref_head.one2one_cv4[i].norm.eval()
    want = ref_head._forward_train([f.clone() for f in feats])

    model.model[-1].fuse_text(tpe)
    model.train()
    got = model.model[-1]._forward_train([f.clone() for f in feats])

    for key in ("one2many", "one2one"):
        torch.testing.assert_close(
            got[key]["scores"].sigmoid(), want[key]["scores"].sigmoid(),
            atol=2e-5, rtol=0, msg=f"{key} fused/unfused 分数不一致")
        # 框与系数来自未改动的分支，逐位一致
        assert torch.equal(got[key]["boxes"], want[key]["boxes"])
        assert torch.equal(got[key]["mask_coefficient"], want[key]["mask_coefficient"])


def test_pf_training_rejected():
    """pf 布局（lrpc 烘焙词汇）没有训练路径——显式报错而非静默走错分支"""
    model = _build(prompt_free=True).train()
    x = torch.zeros(1, 3, 640, 640)
    with pytest.raises(NotImplementedError, match="prompt-free"):
        model(x)


def test_pe_all_scales_fuse_backward():
    """PE 单卡功能覆盖 n/s/m/l/x：官方文本权重 strict load -> fuse -> 只训 6 个尾卷积

    不依赖文本编码器（随机 tpe 即可验证结构/梯度）；权重缺失的档位跳过。
    """
    from model.weights import load_weights
    from train.trainer import freeze_pe

    root = Path(__file__).resolve().parent.parent
    tested = []
    for scale in ("n", "s", "m", "l", "x"):
        path = root / "weights" / f"yoloe-26{scale}-seg.safetensors"
        if not path.is_file():
            continue
        model = build_model("yoloe-26", scale, 320, nc=80)
        load_weights(model, path, strict=True)
        model.eval()
        head = model.model[-1]
        head.fuse_text(_tpe(k=5))
        n = freeze_pe(model)
        assert n == 2 * NL * 2, f"{scale}: trainable tensors {n}"
        model.train()
        out = model(torch.randn(1, 3, 320, 320))
        (out["one2many"]["scores"].sum() + out["one2one"]["scores"].sum()).backward()
        grads = {name for name, p in model.named_parameters() if p.grad is not None}
        assert len(grads) == n, f"{scale}: unexpected grads {sorted(grads)[:5]}"
        assert all("cv3" in name and name.endswith((".weight", ".bias")) for name in grads), scale
        tested.append(scale)
    assert tested, "no official text weights found for any scale"


def test_loss_dispatch_fused_vs_text():
    """PE fused 头 -> ComputeLoss（检测）；未 fused 文本布局 -> ComputeLossSeg"""
    from train.loss import ComputeLoss, ComputeLossSeg, build_loss

    cfg = TrainConfig()
    fused = _build()
    fused.eval()
    fused.model[-1].fuse_text(_tpe())
    assert isinstance(build_loss("yoloe-26", cfg, fused.model[-1], torch.device("cpu")), ComputeLoss)

    text = _build()
    assert isinstance(build_loss("yoloe-26", cfg, text.model[-1], torch.device("cpu")), ComputeLossSeg)


def test_pe_freezes_all_but_cv3_tails():
    """PE 参数边界：fuse_text 产出的 6 个尾卷积可训练，其余全部 requires_grad=False"""
    from train.trainer import freeze_pe, set_all_bn_eval

    model = _build().eval()
    head = model.model[-1]
    head.fuse_text(_tpe(k=7))
    n = freeze_pe(model)
    assert n == 2 * NL * 2  # weight + bias
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == n
    assert all(("cv3." in name and name.endswith((".weight", ".bias"))) for name in trainable)
    assert head.nc == 7
    # 尾卷积输出维度 = prompt 数（类锁定）
    assert head.cv3[0][-1].out_channels == 7

    model.train()  # 恢复训练模式会把所有 BN 置 train；PE 必须逐轮重放 eval
    set_all_bn_eval(model)
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))


def test_pe_param_groups_no_lr_boost():
    """PE 的 6 个尾卷积不走 cv3 的 3× 学习率提升（全量微调专用）"""
    from train.optimizer import build_param_groups
    from train.trainer import freeze_pe

    cfg = TrainConfig(pe=True)
    model = _build().eval()
    head = model.model[-1]
    head.fuse_text(_tpe(k=5))
    freeze_pe(model)
    groups = build_param_groups(model, cfg, head)
    assert all(g["lr_mult"] == 1.0 for g in groups)
    assert sum(len(g["params"]) for g in groups) == 12


def test_fused_checkpoint_pt_engine_roundtrip(tmp_path):
    """PE fused 权重的读取契约：按 struct_nc 重建 + 再 fuse + strict load 后与训练态逐位一致"""
    from model.weights import save_weights
    from utils.engines.pt import PtEngine

    model = _build().eval()
    head = model.model[-1]
    head.fuse_text(_tpe(k=3))
    path = tmp_path / "yoloe-pe.safetensors"
    save_weights(model, path, metadata={
        "arch": "yoloe-26", "scale": "n", "nc": 3, "struct_nc": 80, "imgsz": 640,
        "names": ["a", "b", "c"], "prompt_mode": "text", "pe": True, "fused": True,
    })

    engine = PtEngine(str(path), model="yoloe-26", scale="n", device="cpu")
    assert engine.fused and engine.text_mode
    assert engine.prompt_names == ["a", "b", "c"]
    assert engine.model.model[-1].nc == 3
    for k, v in model.state_dict().items():
        assert torch.equal(v, engine.model.state_dict()[k]), f"tensor mismatch: {k}"
    feats = [torch.randn(1, c, 640 // 2 ** (i + 3), 640 // 2 ** (i + 3))
             for i, c in enumerate((64, 128, 256))]
    with torch.no_grad():
        want = head._forward_train(feats)
        got = engine.model.model[-1]._forward_train(feats)
    for branch in ("one2many", "one2one"):
        assert torch.equal(got[branch]["scores"], want[branch]["scores"])


def test_full_finetune_raw_tpe_trains_reprta():
    """全量微调：缓存原始嵌入（reprta 之前），前向在图内过 get_tpe，reprta 拿到梯度"""
    model = _build().train()
    head = model.model[-1]
    head.set_tpe_raw(_tpe(k=5))
    assert head._tpe is None and head._tpe_raw is not None
    out = model(torch.zeros(1, 3, 640, 640))
    (out["one2many"]["scores"].sum() + out["one2one"]["scores"].sum()).backward()
    assert head.reprta.m.w3.weight.grad is not None, "reprta w3 应有梯度（文本侧参与训练）"
    assert head.reprta.m.w12.weight.grad is not None, "reprta w12 应有梯度"
    assert head.cv3[0][-1].weight.grad is not None, "视觉侧 cv3 尾卷积应有梯度"


def test_set_tpe_clears_raw_cache():
    """显式常量 tpe 优先于全量微调的原始嵌入缓存，避免两套缓存同时生效"""
    head = _build().model[-1]
    head.set_tpe_raw(_tpe(k=2))
    head.set_tpe(_tpe(k=2, seed=1))
    assert head._tpe_raw is None and head._tpe is not None


def test_full_finetune_param_groups_boost_cv3():
    """全量微调保留官方 cv3 3× 学习率提升（PE 线性探测才关闭）"""
    from train.optimizer import build_param_groups

    cfg = TrainConfig(full_finetune=True)
    model = _build()
    head = model.model[-1]
    groups = build_param_groups(model, cfg, head)
    boosted = [g for g in groups if g["lr_mult"] == 3.0]
    assert boosted, "full fine-tuning should keep the cv3 lr x3 boost"
    ids = {id(p) for g in boosted for p in g["params"]}
    assert id(head.cv3[0][-1].weight) in ids


def test_profile_flops_restores_raw_tpe_cache():
    """summary 的 FLOPs 统计会临时塞 dummy tpe：必须原样恢复全量微调的 _tpe_raw 缓存"""
    from model.summary import profile_flops

    model = _build()
    head = model.model[-1]
    raw = _tpe(k=3)
    head.set_tpe_raw(raw)
    profile_flops(model, imgsz=160, device="cpu")
    assert head._tpe is None and head._tpe_raw is not None
    assert torch.equal(head._tpe_raw, raw)


def test_full_finetune_checkpoint_pt_engine_roundtrip(tmp_path):
    """全量微调产物读取契约：未 fused 文本布局按 struct_nc 重建 + strict load 后前向一致"""
    from model.weights import load_meta, save_weights
    from utils.engines.pt import PtEngine

    model = _build().eval()
    head = model.model[-1]
    tpe = _tpe(k=3)
    head.set_tpe(tpe)
    path = tmp_path / "yoloe-ft.safetensors"
    save_weights(model, path, metadata={
        "arch": "yoloe-26", "scale": "n", "nc": 3, "struct_nc": 80, "imgsz": 640,
        "names": ["a", "b", "c"], "prompt_mode": "text", "full_finetune": True,
    })
    meta = load_meta(path)
    assert meta["full_finetune"] is True and meta["struct_nc"] == 80, meta

    engine = PtEngine(str(path), model="yoloe-26", scale="n", device="cpu")
    assert engine.text_mode and not engine.fused
    engine.model.model[-1].set_tpe(tpe)
    feats = [torch.randn(1, c, 640 // 2 ** (i + 3), 640 // 2 ** (i + 3))
             for i, c in enumerate((64, 128, 256))]
    with torch.no_grad():
        want = head._forward_train(feats)
        got = engine.model.model[-1]._forward_train(feats)
    for branch in ("one2many", "one2one"):
        assert torch.equal(got[branch]["scores"], want[branch]["scores"]), branch

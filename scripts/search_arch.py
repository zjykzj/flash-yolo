"""flash-yolo architecture design / search.

设计命题：v3-tiny 的廉价下采样哲学 × yolo26 的 CSP 容量分配；头（Detect, E2E）与
损失/训练管线全复用，改动集中在 backbone/neck（yolo26n 的 ~89% 算力所在）。

维度：
    ds    : L3/L5/L7/L17/L20 的 stride-2 3×3 Conv -> PoolConv（maxpool + 1×1）
    stem1 : L1 -> PoolConv
    stem2 : L2 C3k2 -> Conv 1×1（channel 投影）| dw1x1（DW 3×3 + 1×1，插入一行）
    hr    : 高分辨段（L3/L4/L5）通道因子
    nk    : neck 通道因子 | "lite"（L13/16/19/22 C3k2 -> LiteBlock）

变体 yaml 写到 /tmp/flash_yolo_search/，经 cfg_path 构建（不改 config/、不注册 ARCHS）。
PoolConv / LiteBlock 为正式算子（model/basic，已注册进 MODULES）。
FLOPs = conv/linear/pool 解析口径（anchor yolo26n = 5.36 ≡ summary.py 口径 5.5）。
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

torch.set_num_threads(4)  # 训练占用 CPU 时让路

from model.build import YOLO26_CONFIG_PATH, DetectionModel, make_divisible  # noqa: E402

OUT_DIR = Path("/tmp/flash_yolo_search")
CHANNEL_BLOCKS = {"Conv", "C3k2", "SPPF", "C2PSA", "LeakyConv", "DWConv", "PoolConv", "LiteBlock"}
POOL_DS_LAYERS = (3, 5, 7, 17, 20)     # yolo26n 的全部 stride-2 3×3 下采样（L0/L1 单列）
HR_LAYERS = (3, 4, 5)                  # 高分辨段（P3/8、P4/16 入口）
NK_LAYERS = (13, 16, 17, 19, 20, 22)   # neck（17/20 为下采样，13/16/19/22 为 C3k2）
NK_BLOCK_LAYERS = (13, 16, 19, 22)     # LiteBlock 替换目标（保留 17/20 的下采样职责）


# PoolConv / LiteBlock 已提升为正式算子（model/basic），MODULES 注册见 model/build.py


def _leaf_flops(mod, out):
    if isinstance(mod, nn.Conv2d):
        return (2 * out.shape[0] * out.shape[2] * out.shape[3] * mod.out_channels
                * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1])
    if isinstance(mod, nn.Linear):
        return 2 * out.numel() * mod.in_features
    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
        k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
        return out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3] * k * k
    return 0


def measure(model, imgsz=640):
    """Returns (total_flops, n_params, per_layer_flops)"""
    head = model.model[-1]
    saved = getattr(head, "end2end", None)
    if saved is not None:
        head.end2end = False
    model.eval()
    buckets = [0] * len(model.model)
    handles = []
    for i, layer in enumerate(model.model):
        def mk(i):
            def hook(mod, inp, out):
                buckets[i] += _leaf_flops(mod, out)
            return hook
        for sub in layer.modules():
            if isinstance(sub, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)):
                handles.append(sub.register_forward_hook(mk(i)))
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    for h in handles:
        h.remove()
    if saved is not None:
        head.end2end = saved
    return sum(buckets), sum(p.numel() for p in model.parameters()), buckets


def load_template():
    """yolo26n 的 yaml 行模板：repeats 与通道替换为 n 档解析值（flash 档 = [1.0, 1.0, 1024]）"""
    with open(YOLO26_CONFIG_PATH) as f:
        d = yaml.safe_load(f)
    rows = copy.deepcopy(d["backbone"]) + copy.deepcopy(d["head"])
    base = DetectionModel(cfg_path=YOLO26_CONFIG_PATH, scale="n", imgsz=640)
    for row, layer in zip(rows, base.model):
        row[1] = layer.repeats
        if row[2] in CHANNEL_BLOCKS:
            row[3][0] = layer.args[1]  # resolved c2
    return rows, len(d["backbone"])


def slim(row, f):
    row[3][0] = max(8, make_divisible(row[3][0] * f))


def to_poolconv(row):
    row[2], row[3] = "PoolConv", [row[3][0]]


def insert_after(rows, i, extra):
    """在 rows[i] 之后插入 extra 行；> i 的行索引与 from 引用整体右移（增层手术）"""
    n = len(extra)
    out = []
    for j, row in enumerate(rows):
        r = copy.deepcopy(row)
        if j == i:
            out.append(r)
            out.extend(copy.deepcopy(e) for e in extra)
        else:
            f = r[0]
            if isinstance(f, list):
                r[0] = [x + n if x > i else x for x in f]
            elif f > i:
                r[0] = f + n
            out.append(r)
    return out


def build_variant(tpl, ds, stem1, stem2, hr, nk):
    """Returns (rows, n_inserted)；顺序：ds/hr/nk/stem1 先做（按原索引），插入型 stem2 最后"""
    rows = copy.deepcopy(tpl)
    if ds == "pool":
        for i in POOL_DS_LAYERS:
            to_poolconv(rows[i])
    if hr != 1.0:
        for i in HR_LAYERS:
            slim(rows[i], hr)
    if nk == "lite":
        for i in NK_BLOCK_LAYERS:
            rows[i][2], rows[i][3] = "LiteBlock", [rows[i][3][0]]
    elif nk != 1.0:
        for i in NK_LAYERS:
            slim(rows[i], nk)
    if stem1 == "pool":
        to_poolconv(rows[1])
    n_extra = 0
    if stem2 == "conv1x1":
        rows[2][2], rows[2][3] = "Conv", [rows[2][3][0], 1, 1]
    elif stem2 == "dw1x1":
        c2 = rows[2][3][0]  # 64（n 档），c1 = L1 输出 = 32（stem1 不改通道）
        rows[2] = [-1, 1, "DWConv", [32, 3, 1]]
        rows = insert_after(rows, 2, [[-1, 1, "Conv", [c2, 1, 1]]])
        n_extra = 1
    return rows, n_extra


def dump(rows, path, n_backbone):
    d = {"nc": 80, "reg_max": 1, "end2end": True,
         "scales": {"flash": [1.0, 1.0, 1024]},
         "backbone": rows[:n_backbone], "head": rows[n_backbone:]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def layer_table(model):
    return [f"  {m.i:>3} {str(m.f):>9}  {type(m).__name__:<10} {m.args}" for m in model.model]


def top_layers(model, buckets, k=8):
    idx = sorted(range(len(buckets)), key=lambda i: -buckets[i])[:k]
    total = sum(buckets)
    out = []
    for i in idx:
        if buckets[i] / total < 0.01:
            break
        lay = model.model[i]
        out.append(f"    L{i:>2} {buckets[i] / 1e9:>5.2f}G {100 * buckets[i] / total:>4.1f}%  "
                   f"{type(lay).__name__:<10} {lay.args}")
    return out


PRESETS = {
    # v1 主设计：全池化下采样 + stem 1×1 投影
    "S1-v1": dict(ds="pool", stem1="pool", stem2="conv1x1", hr=1.0, nk=1.0),
    # v1 + DW stem（stem 是否要空间混合，A/B）
    "S2-v1dwstem": dict(ds="pool", stem1="pool", stem2="dw1x1", hr=1.0, nk=1.0),
    # v1 + neck 通道 ×0.75（等算力下"砍通道"路线）
    "S3-v1nk75": dict(ds="pool", stem1="pool", stem2="conv1x1", hr=1.0, nk=0.75),
    # v1 + neck C3k2 -> LiteBlock（等算力下"换 block"路线，与 S3 直接对照）
    "S4-v1lite": dict(ds="pool", stem1="pool", stem2="conv1x1", hr=1.0, nk="lite"),
}


def main():
    ap = argparse.ArgumentParser(description="flash-yolo architecture design/search")
    ap.add_argument("--grid", action="store_true", help="run the full 32-variant replacement grid")
    args = ap.parse_args()

    tpl, n_bb = load_template()
    anchor = DetectionModel(cfg_path=YOLO26_CONFIG_PATH, scale="n", imgsz=640)
    fl0, p0, _ = measure(anchor)
    print(f"anchor yolo26n: {fl0 / 1e9:.2f} GFLOPs, {p0 / 1e6:.2f}M params "
          f"(analytic; summary.py convention ≈ 5.5)")

    if args.grid:
        grid = [(ds, s1, s2, hr, nk)
                for ds in ("conv", "pool") for s1 in ("conv", "pool")
                for s2 in ("c3k2", "conv1x1") for hr in (1.0, 0.75) for nk in (1.0, 0.75)]
        print(f"\n== grid: {len(grid)} variants (sorted by GFLOPs) ==")
        rows_out = []
        for ds, s1, s2, hr, nk in grid:
            rows, n_extra = build_variant(tpl, ds, s1, s2, hr, nk)
            tag = f"ds{ds}_s1{s1}_s2{s2}_hr{hr}_nk{nk}".replace(".", "p")
            dump(rows, OUT_DIR / f"grid_{tag}.yaml", n_bb + n_extra)
            fl, prm, _ = measure(DetectionModel(cfg_path=OUT_DIR / f"grid_{tag}.yaml",
                                                scale="flash", imgsz=640))
            rows_out.append((fl, prm, tag))
        for fl, prm, tag in sorted(rows_out):
            print(f"{tag:<44}{fl / 1e9:>8.2f}{prm / 1e6:>9.2f}M")

    print("\n== v1 presets ==")
    results = {}
    for name, kw in PRESETS.items():
        rows, n_extra = build_variant(tpl, **kw)
        yml = OUT_DIR / f"{name}.yaml"
        dump(rows, yml, n_bb + n_extra)
        model = DetectionModel(cfg_path=yml, scale="flash", imgsz=640)
        fl, prm, buckets = measure(model)
        results[name] = (fl, prm, buckets, model)
        print(f"{name:<14} {fl / 1e9:>6.2f} GFLOPs   {prm / 1e6:>5.2f}M params   "
              f"({100 * (1 - fl / fl0):.0f}% below anchor)")

    fl, prm, buckets, model = results["S1-v1"]
    print(f"\n-- S1-v1 layer table ({fl / 1e9:.2f} GFLOPs, {prm / 1e6:.2f}M) --")
    print("\n".join(layer_table(model)))
    print(f"\n-- S1-v1 top layers --")
    print("\n".join(top_layers(model, buckets)))
    for name in ("S4-v1lite",):
        fl, prm, buckets, model = results[name]
        print(f"\n-- {name} top layers ({fl / 1e9:.2f} GFLOPs, {prm / 1e6:.2f}M) --")
        print("\n".join(top_layers(model, buckets)))
    print(f"\nyamls saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

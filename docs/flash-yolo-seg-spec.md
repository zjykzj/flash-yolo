# flash-yolo-seg 设计说明（分割变体）

> 定位：`flash-yolo` 的实例分割变体，模型名 `flash-yolo-seg`（`arch_task = segment`）。
> backbone/neck 与检测版**逐行相同**，只把最后一行 `Detect` 换成 `Segment26`——
> 与 `yolo26 -> yolo26-seg` 是同一种关系。结构事实见本文，训练/评估数字见 §4。

## 1. 命名

- 家族名在前、任务后缀在后：`flash-yolo-seg`（与 `yolo26-seg` 同构），**不是** `flash-yolo-seg-n`。
- 单档 `flash`：通道已材料化，没有 n/s/m/l/x 维度；`--scale` 对它不适用
  （`arch_has_scale("flash-yolo-seg") is False`，训练/评估提示都不会追加 `--scale`）。
- 权重文件名可反推架构（`flash-yolo-seg.safetensors` -> `flash-yolo-seg`）；
  正则里长名在前，`flash-yolo-seg` 不会被 `flash-yolo` 前缀误吸。

## 2. 结构（`config/models/flash-yolo-seg.yaml`）

25 行 yaml，与 `flash-yolo.yaml` 的差异**只有最后一行**：

```yaml
- [[17, 20, 23], 1, Segment26, [nc, 32, 64]]
```

`Segment26` = `Detect` + `cv4` 掩码系数分支 + `Proto26` 原型分支 + 训练期 semseg 辅助头
（`model/head/yolo26_seg.py`，与 yolo26-seg 共用一个实现）。三个检测层仍是 P3/P4/P5
（idx 17/20/23），P3/P4/P5 通道 64/128/256 与 yolo26n 相同，头参数直接复用。

**`npr = 64` 的由来**：`yolo26-seg.yaml` 里写的是缩放前的 `256`，n 档 `width=0.25` 经
`make_divisible(256*0.25, 8)` 得到 **64**（即官方 yolo26n-seg 的生效原型通道数）。flash 档
`width=1.0`，所以这里直接写生效值 `64`；写 `256` 会得到 4.68M 参数 / 47.8 GFLOPs 的错误结构。
`nm = 32` 是任务常数（掩码系数维度），与 yolo26-seg 一致。

## 3. 规模与同构性

| 指标 | flash-yolo（检测） | flash-yolo-seg | yolo26n-seg |
|---|---|---|---|
| 参数 | 1,974,248 | **2,528,248** | 3,126,280 |
| GFLOPs @640 | 3.80 | **7.6** | 9.3 |
| 头参数 | 309,656 | **863,656** | 863,656 |
| 其中 semseg 辅助头 | — | 79,184 | 79,184 |

- **backbone/neck 逐键同构**：`tests/test_flash_seg_model.py::test_backbone_neck_identical_to_detect`
  对同一 seed 下的两份 state_dict 逐键比对（`model.0`–`model.23` 键集与数值全等）。
- 与 yolo26n-seg 相比只省在 backbone/neck（−598,032 参数、−1.7 GFLOPs）：两者 P3/P4/P5 通道
  同为 64/128/256，`Segment26` 头逐键相同（863,656），所以分割头不是本次改动的变量。
- GFLOPs 相对检测版约翻倍来自原型分支：`Proto26` 在 P3/P4/P5 上卷积并输出 160×160×32 原型，
  与检测头无关，官方 seg 也一样（yolo26n 5.48 → yolo26n-seg 9.3）。

## 4. 复用的链路（零特判）

| 环节 | 复用方式 |
|---|---|
| 前向契约 | E2E `(B,300,6+nm)` + proto `(B,32,160,160)`；raw `(B,4+nc+nm,N)` + proto |
| 训练模式 | 双分支 dict（`one2many`/`one2one`，o2o 的 proto/semantic 均 detach） |
| 损失 | `build_loss` 分派到 `ComputeLossSeg`（检测三项/归一化/ProgLoss 完全复用 + 掩码 BCE） |
| 验证 | 训练中 Box+Mask（`val_seg`），best fitness = box + mask mAP50-95；正式口径走 `scripts/eval.py` |
| 权重 | safetensors metadata 写 `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz`，回载免显式参数 |
| 引擎 | 同一套 pt / onnx / trt 双输出路径（output0 检测+系数、output1 原型） |

除注册表（`ARCHS`）、损失分派表（`_LOSS_BY_ARCH`）与权重名正则外，没有新增分支：
数据、增强、损失、验证、导出、推理全部走既有分割链路。

## 5. 复现

```bash
# 结构自检（无需数据）
python model/summary.py --model flash-yolo-seg
# -> 311 layers, 2,528,248 parameters, 7.6 GFLOPs

# 训练（default 配方；COCO 分割标签见 docs/dataset.md）
python scripts/train.py --model flash-yolo-seg --data coco --epochs 20 --batch 32 --workers 8 --mask-ratio 4

# 正式评估（Box + Mask，权重 metadata 自动解析 arch/scale）
python scripts/eval.py --weights runs/train/train-flash-seg-20ep/weights/best.safetensors --data coco

# 推理 / 导出
python scripts/infer.py --weights runs/train/train-flash-seg-20ep/weights/best.safetensors --image assets/traffic.jpg
python scripts/export.py --weights runs/train/train-flash-seg-20ep/weights/best.safetensors
```

## 6. 验收

结构闸门（不依赖训练）：

- `tests/test_flash_seg_model.py`：参数数/头参数/semseg 参数、与 flash-yolo 的 body 同构、
  E2E/raw/训练三种前向契约、自定义 nc、命名与任务解析、损失分派；
- `tests/test_train_smoke.py::test_train_smoke_flash_seg`（`FLASH_YOLO_SMOKE=1`，CPU，2 epoch）：
  `cfg.scale == "flash"`、分割列名、权重 metadata、strict 回载。

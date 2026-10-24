# flash-yolo-seg 设计说明（分割变体）

> 定位：`flash-yolo` 的实例分割变体，模型名 `flash-yolo-seg`（`arch_task = segment`）。
> backbone/neck 与检测版**逐行相同**，只把最后一行 `Detect` 换成 `Segment26`——
> 与 `yolo26 -> yolo26-seg` 是同一种关系。结构事实见本文，训练/评估数字见 §7–§8。

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
python scripts/train.py --model flash-yolo-seg --data coco --epochs 20 --batch 32 --workers 8 \
    --mask-ratio 4 --name flash-seg-20ep

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

20 轮从零训练与同预算对照的正式数字见 §7。

## 7. 20 轮从零验收（2026-10-21）

设置：**default 配方、从零、640、20 epoch、batch 32、nbs 64、fp32、mask_ratio 4、seed 0**。
对照组 yolo26n-seg 用完全相同的配方与预算（唯一变量 = 架构），两组都从随机初始化开始。
本机产物为 `runs/train/train-flash-seg-20ep2`（`--name flash-seg-20ep` 被上一轮中止的 run
占用后自动加序号）与 `runs/train/train-seg-n-20ep-scratch`。

正式口径（`scripts/eval.py` + pycocotools，COCO val2017 5000 图）：

| 模型 | 参数 | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| flash-yolo-seg | 2,528,248 | 7.6 | 0.3821 | 0.2580 | 0.3588 | 0.2207 |
| yolo26n-seg（从零对照） | 3,126,280 | 9.3 | 0.4081 | 0.2780 | 0.3843 | 0.2395 |
| **Δ（flash − 对照）** | **−19.1%** | **−18.3%** | **−0.0260** | **−0.0200** | **−0.0255** | **−0.0188** |
| 相对 | | | −6.4% | −7.2% | −6.6% | −7.8% |

- **与检测版的筛选结论同量级**：flash-yolo 100 轮检测是 −2.45 box mAP（−7%，参数 −23% /
  GFLOPs −31%）；分割 20 轮是 −7.2% box / −7.8% mask（参数 −19% / GFLOPs −18%）。
  两个任务上的架构代价一致，说明 PoolConv/DW stem 的取舍不随检测头或分割头改变。
- **20 轮远未收敛**：两条曲线的最佳 fitness 都出现在最后一轮（测试集每轮 box +0.003~0.005、
  mask +0.002~0.003 稳定上升），绝对数字只用于同预算相对比较。
- **不要与官方 O365 初始化微调 20 轮（Box 0.3633 / Mask 0.3083）直接横比**：那一组起点是
  `yolo26n-objv1-seg` 预训练权重；本表第二行才是同起点对照。
- 训练成本：两组都约 **4.16 h / 20 轮**（每轮 11–13 min）；分割训练由 loader 与验证主导，
  架构差异在吞吐上不可辨。

**导出对齐（同权重）**：

- `scripts/export.py` 双输出 ONNX 正常；**pt(CPU) vs onnx(CPU)** 原始输出相对差 ~1e-6；
- **pt(CUDA) vs onnx(CPU)** 原始输出相对差 1e-3~4e-3，掩码出现边界像素翻转（最差实例
  IoU 0.993，其余 ≤0.1% 像素）。原因是非同设备运行时（cuDNN vs oneDNN）的舍入差被 DW stem
  的 BN 放大（`model.2.bn` running_var 最小 1.8e-3，eps=1e-3）；关掉 ONNX 常量折叠后差异不变，
  所以不是导出结构问题。跨设备对齐应在同运行时比较，或走 TensorRT。

## 8. 100 轮从零结果（2026-10-24）

设置：**default 配方、从零、640、100 epoch、batch 16、nbs 64、fp32、mask_ratio 4、seed 0**；
单卡 RTX 5090，**37 h 42 min**；产物 `runs/train/train-flash-seg-100ep2`。对照组
`runs/train/train-seg-n-100ep2` 使用完全相同的配方、数据与预算（唯一变量 = 架构）。

正式口径（`scripts/eval.py` + pycocotools，COCO val2017 5000 图）：

| 模型 | 参数 | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| flash-yolo-seg | 2,528,248 | 7.6 | 0.4796 | 0.3311 | 0.4534 | 0.2824 |
| yolo26n-seg（从零对照） | 3,126,280 | 9.3 | 0.5103 | 0.3569 | 0.4861 | 0.3052 |
| **Δ（flash − 对照）** | **−19.1%** | **−18.3%** | **−0.0307** | **−0.0258** | **−0.0327** | **−0.0228** |
| 相对 | | | −6.0% | −7.2% | −6.7% | −7.5% |

- **架构代价与检测版同量级**：flash-yolo 100 轮检测是 −2.45 box mAP50-95（−7%）；分割是
  −2.58 box / −2.28 mask（−7.2% / −7.5%），参数 −19.1%、GFLOPs −18.3%。两个任务上的结论一致，
  说明 PoolConv/DW stem 的取舍不随检测头或分割头改变。
- **best 在末轮（ep100），曲线仍在上升**：100 轮是预算点，不是收敛上限；绝对数字用于同预算
  相对比较。
- **不要与官方 O365 初始化权重直接横比**：官方 yolo26n-seg 的 39.67 / 33.67（Box / Mask
  mAP50-95）是预训练 + 微调口径，本表第二行才是同起点对照。
- 训练成本：两组分别 **37 h 42 min / 37 h 21 min**（batch 16）；分割训练由 loader 与验证
  主导，架构差异在吞吐上不可辨。

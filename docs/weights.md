# 发布权重档案（Released checkpoints）

> 权重与延迟的权威档案：4 份**从零自训**检测 checkpoint 的精度 / 速度 / 部署产物记录
> （2026-10-15 复测），另附分割参考数字。2026-10-26/28 补录了 v0.7.0 默认值的 300 轮 run
> （v3 与检测，本地、未分发，见 §1.1 / §1.2）。
> 权重本体随 GitHub [Releases](https://github.com/zjykzj/flash-yolo/releases) 分发（自训产物，
> 官方来源的转换权重不随附——用 `scripts/download_weights.py` + `convert_weights.py` 自行获取）。
> 训练过程与配方见 `yolov3-tiny-runs.md` / `yolo26-runs.md` / `flash-yolo-spec.md` /
> `flash-yolo-seg-spec.md`。README 只保留精度摘要，完整延迟与部署档案以本页为准。

## 1. 清单与精度（COCO val2017，pycocotools，`scripts/eval.py`）

| 权重（release 资产名） | 架构 / 配置 | 训练 | 输入 | mAP50-95 | mAP50 | P / R |
|---|---|---|---|---|---|---|
| `yolov3-tiny-silu-softobj-e300-416` | YOLOv3-tiny-SiLU + IoU-aware objectness（v0.6.0 推荐配方） | 300 ep 从零 · v0.6.0 默认（MuSGD 0.528/0.674） | 416 | **16.52** | **30.44** | 0.4612 / 0.3216 |
| `yolov3-tiny-softobj-e300-416` | YOLOv3-tiny（原版算子）+ IoU-aware objectness | 300 ep 从零 · v0.6.0 默认 | 416 | 16.07 | 29.56 | 0.4436 / 0.3150 |
| `yolo26n-from-scratch-e100-640` | YOLO26n · E2E（NMS-free） | 100 ep 从零 | 640 | 36.22 | 51.32 | 0.6342 / 0.4895 |
| 同上 · NMS（o2m） | | 同 run | 640 | 37.15 | 52.67 | 0.6554 / 0.4965 |
| `flash-yolo-e100-640` | Flash-YOLO（S2）· E2E（NMS-free） | 100 ep 从零 | 640 | 33.77 | 48.63 | 0.6220 / 0.4594 |
| 同上 · NMS（o2m） | | 同 run | 640 | 34.72 | 50.16 | 0.6254 / 0.4801 |

参考系：官方 darknet yolov3-tiny 权重同管线 @416 = **16.94 / 34.37**；官方 YOLO26n（O365 预训练
+ COCO 微调）= **40.27 / 55.80**（E2E）· **40.89 / 56.87**（NMS）。v3 两行与官方 v3 的 mAP50-95
差 0.42 / 0.87；mAP50 差 3.93 / 4.81。

### 1.1 v3 300 轮（v0.7.0 默认，未随 Release 分发）

2026-10-25/26 用 v0.7.0 原样默认（MuSGD 0.2/1.0）复跑了两条 300 轮，配方与上表两份 v3
权重逐项相同（batch 64 / nbs 64 / 416 / seed 0 / `--obj-iou-target`），唯一差异是 MuSGD
默认常数（v0.6.0 批次为 0.528/0.674）与随机流。产物只在本地 `runs/`：

| 权重（本地路径） | 架构 / 配置 | 训练 | 输入 | mAP50-95 | mAP50 | P / R |
|---|---|---|---:|---:|---:|---:|
| `runs/train/train-v1-yolov3-tiny-silu-e300-416-softobj3/weights/best.safetensors` | YOLOv3-tiny-SiLU + objectness | 300 ep 从零 · v0.7.0 默认 | 416 | 16.15 | 29.93 | 0.4450 / 0.3216 |
| `runs/train/train-v1-yolov3-tiny-e300-416-softobj2/weights/best.safetensors` | YOLOv3-tiny（原版）+ objectness | 300 ep 从零 · v0.7.0 默认 | 416 | 16.03 | 29.59 | 0.4383 / 0.3250 |

对照：v0.6.0 批次同配方 = 16.52 / 30.44（SiLU）与 16.07 / 29.56（原版）——原版基本复现，
SiLU 低 0.37 / 0.51；SiLU 相对原版的增益从 +0.45 / +0.88 缩小到 +0.12 / +0.34。分析见
`yolov3-tiny-runs.md` §4.6。要复现 v0.6.0 数字，训练时加 `--muon-w 0.528 --sgd-w 0.674`。

### 1.2 检测 300 轮（v0.7.0 默认，未随 Release 分发）

2026-10-25 → 10-28 从零 300 轮（default 配方、640、batch 32、nbs 64、fp32、seed 0、v0.7.0
默认 MuSGD 0.2/1.0），产物在本地 `runs/`：

| 权重（本地路径） | 输出 | 训练 | 输入 | mAP50-95 | mAP50 | P / R |
|---|---|---|---:|---:|---:|---:|
| `runs/train/train-v1-yolo26n-e3002/weights/best.safetensors` | YOLO26n · E2E | 300 ep · 63 h 12 min | 640 | **37.82** | **53.24** | 0.6652 / 0.4984 |
| 同上 · NMS（o2m） | | 同 run | 640 | 38.65 | 54.46 | 0.6752 / 0.5092 |
| `runs/train/train-v1-flash-yolo-e3002/weights/best.safetensors` | Flash-YOLO · E2E | 300 ep · 62 h 39 min | 640 | **35.14** | **50.15** | 0.6245 / 0.4799 |
| 同上 · NMS（o2m） | | 同 run | 640 | 36.09 | 51.56 | 0.6441 / 0.4910 |

- 100 轮参考（§1 主表）：yolo26n 36.22 / 51.32（NMS 37.15 / 52.67）、flash 33.77 / 48.63
  （NMS 34.72 / 50.16）→ 300 轮增益 +1.60 / +1.92 与 +1.37 / +1.52（E2E）。
- 官方 yolo26n（O365 + COCO）E2E 40.27 / 55.80 → 从零 300 轮差 2.45 / 2.56。
- 评估产物：`runs/val/val5`（yolo26n E2E）、`val7`（NMS）、`val6`（flash E2E）、`val8`（NMS）。

## 2. 分割参考（COCO val2017，Box + Mask，640）

口径同 `scripts/eval.py`（pycocotools，`iou_type="bbox"` / `"segm"`）。官方行是 O365 预训练 +
COCO 微调；从零行与官方**不是同预算**。

| 权重 / 来源 | 训练 | 参数 | GFLOPs | Box 50-95 | Box 50 | Mask 50-95 | Mask 50 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg 官方权重 | O365 + COCO | 3,126,280 | 9.3 | 39.67 | 55.48 | 33.67 | 52.76 |
| YOLO26n-seg 从零 | 100 ep | 3,126,280 | 9.3 | 35.69 | 51.03 | 30.52 | 48.61 |
| Flash-YOLO-seg 从零 | 100 ep | 2,528,248 | 7.6 | 33.11 | 47.96 | 28.24 | 45.34 |

- 官方训练期 validator 记录为 box 0.3975 / mask 0.3386，本管线 0.3967 / 0.3367，差 ~0.002
  属指标实现噪声（同检测侧 40.1 vs 40.27）。
- 分割三行中，官方权重需用 `scripts/download_weights.py` 自行获取；两个从零行是本地训练
  产物，已按 release 命名归档（`runs/new/`：`yolo26n-seg-from-scratch-e100-640` /
  `flash-yolo-seg-e100-640`），暂未随 Releases 分发。
- 同预算 100 轮对照（同配方、batch 16、nbs 64、fp32，唯一变量 = 架构）：Flash-YOLO-seg
  Box/Mask mAP50-95 = 33.11 / 28.24，YOLO26n-seg = 35.69 / 30.52（Δ −2.58 / −2.28，
  相对 −7.2% / −7.5%）。
- 同预算 20 轮对照（`flash-yolo-seg-spec.md` §7）：Flash-YOLO-seg Box 25.80 / Mask 22.07；
  YOLO26n-seg Box 27.80 / Mask 23.95（mAP50-95）。该组只用于架构相对比较，不代表收敛精度。
- Flash-YOLO-seg 100 轮训练 37 h 42 min，best 在末轮（ep100），曲线仍在上升；训练与评估记录见
  `runs/` 与 `flash-yolo-seg-spec.md` §8。

## 3. 延迟（model-only 推理段，不含前后处理；2026-10-15 复测）

> README 首页不再重复本表，只链接到这里。`b1` = batch 1 单图延迟；`b8` = batch 8 的**每图**
> 平均耗时（偏吞吐）。数字依赖机器与宿主负载，跨会话看趋势。

条件：**ORT-CPU** = Intel Xeon 8470Q 25 核配额（intra_op 16 / inter_op 1；b1 固定图 · b8 动态图；
每档 300 样本取 best-case `min`，host load ≈13）· **TensorRT** = RTX 5090 · TRT 11.4
（b1 固定引擎 · b8 batch-8 引擎；30 次单发 + 100 次流水线）。

| 权重 | ORT-CPU b1 | ORT-CPU b8 | TRT b1 fp32 | TRT b1 fp16 | TRT b8 fp32 | TRT b8 fp16 |
|---|---|---|---|---|---|---|
| `yolov3-tiny-silu-softobj-e300-416` | 4.82 ms | 3.74 ms/img | **0.68 ms** | 1.19 ms | 0.133 ms/img | **0.048 ms/img** |
| `yolov3-tiny-softobj-e300-416` | 4.52 ms | 3.61 ms/img | **0.68 ms** | 1.15 ms | 0.133 ms/img | 0.049 ms/img |
| `yolo26n-from-scratch-e100-640` | 14.6 ms | 6.86 ms/img | 1.37 ms | 1.28 ms | 0.245 ms/img | 0.135 ms/img |
| `flash-yolo-e100-640` | 13.2 ms | 5.84 ms/img | 1.31 ms | 1.27 ms | 0.210 ms/img | 0.123 ms/img |

读数要点：

- **v3 的 fp16 在 b1 反而慢 ~70%**（0.68 → 1.19 ms；小图 b1 由调度开销主导）→ **b1 用 fp32、
  b8 用 fp16**；yolo26n / flash 两精度 b1 持平（≈1.3 ms）、b8 fp16 快 1.7–1.8×。
- **fp16 引擎数值一致性**（与 fp32 引擎逐检对照，按同类 IoU 贪心配对；traffic.jpg）：
  v3-silu 300 vs 300 检出、配对 299（1 个为 max_det 截断边缘），IoU min 0.972 / 均值 0.993、
  最大框差 0.79 px；yolo26n 21 vs 21 全部配对，IoU min 0.990、最大框差 0.23 px。
  **不要按分数序逐位对照**——近分框在两精度间会互换（同导出链路的既有教训）。
- ORT-CPU 绝对数受宿主邻居负载影响（早期会话在 load ≈40 时 b8 读到 8.27 / 6.09）——同机同
  协议可比，跨会话看趋势。
- 2026-10-26 复跑的两份 v3 权重与上表 v3 两行**架构与导出链路完全相同**，延迟不随权重数值
  变化，故不新增表行。本次抽检（416，单图 model-only）：pt ≈2.4–2.7 ms、ORT-CPU
  ≈4.4–5.1 ms、TRT fp32 ≈0.8 ms、TRT fp16 ≈1.0–1.2 ms，与上表同量级。
- 2026-10-28 的两份检测 300 轮权重（§1.2）与 100 轮检测权重架构/导出链路相同，延迟不随权重
  数值变化，不新增表行。

## 4. 部署产物（`runs/release-weights/`，与本页同批生成）

每份权重 9 个文件：

| 文件 | 口径 |
|---|---|
| `<stem>.onnx` | fp32 · batch 1（yolo26 系 = E2E 图内 top-k；v3 = 解码输出 `(1, NA, 5+nc)`） |
| `<stem>.fp16.onnx` / `<stem>.fp16.engine` | 半精度图 + strongly-typed 构建（TRT 11 无 FP16 flag） |
| `<stem>.engine` | fp32 · b1 |
| `<stem>.dyn.onnx` | 动态 batch 图（ORT b8 用） |
| `<stem>.b8.fp32.onnx` / `.b8.fp32.engine` | raw batch-8 图 + 引擎（TRT b8 fp32） |
| `<stem>.b8.fp16.onnx` / `.b8.fp16.engine` | 同上 · fp16 |

- **engine 绑定本机**（RTX 5090 + TensorRT 11.4）——换机需重建：
  `scripts/export.py --weights <w> --trt [--fp16]`；b8 的 raw 构造见
  `runs/artifacts/bench/release_build_bench_b8.py`（仓库导出路径固定 b1）。
- 复现命令：
  ```bash
  python scripts/eval.py --weights <release 权重>.safetensors --data coco          # v3 加 --model 可省（metadata）
  python scripts/export.py --weights <release 权重>.safetensors --out <name>.onnx --trt [--fp16]
  ```

- 2026-10-26 复跑权重的本地部署产物（未进 release 档案）：`runs/export/` 下每份 4 个文件——
  `<stem>.onnx` / `<stem>.engine`（fp32）与 `<stem>.fp16.onnx` / `<stem>.fp16.engine`
  （stem = `v1-yolov3-tiny-softobj-e300-416` / `v1-yolov3-tiny-silu-softobj-e300-416`）；
  b8 引擎未重建（同架构，历史 b8 数字仍适用）。

## 5. 溯源（随权重 metadata）

- 两份 v3 权重：header 自带 `arch/scale/nc/imgsz/names` + provenance
  （`version 0.5.0` · `commit 0baa0bf` · `describe v0.5.0-20-g0baa0bf`）
- 两份 v0.7.0 复跑权重（本地）：header 自带 `arch/scale/nc/imgsz/names` + provenance
  （`version 0.7.0` · `commit e7cac56` · `describe v0.7.0-3-ge7cac56-dirty`）
- 两份检测 300 轮权重（本地）：同 provenance（`version 0.7.0` · `commit e7cac56` ·
  `describe v0.7.0-3-ge7cac56-dirty`）
- `flash-yolo-e100-640`：自带 arch/scale/nc/imgsz（保存于 2026-10-11，无 provenance 字段）
- `yolo26n-from-scratch-e100-640`：无 metadata（保存于 2026-10-08）——消费时需显式
  `--model yolo26 --scale n --imgsz 640`

## 6. 相关文档

- [与官方实现的对齐与验证](alignment.md) — 官方权重逐位对齐、训练内部 parity、测试套件
- [工程结构与模块职责](architecture.md) — 目录结构、模块边界、扩展点
- [评估口径](val.md) — pycocotools / FastMetrics / NMS 与评估坑

# 发布权重档案（Released checkpoints）

> 4 份**从零自训** checkpoint 的精度 / 速度 / 部署产物记录（2026-10-15 复测）。
> 权重本体随 GitHub [Releases](https://github.com/zjykzj/flash-yolo/releases) 分发（自训产物，
> 官方来源的转换权重不随附——用 `scripts/download_weights.py` + `convert_weights.py` 自行获取）。
> 训练过程与配方见 `yolov3-tiny-runs.md` / `yolo26-runs.md` / `flash-yolo-spec.md`。

## 1. 清单与精度（COCO val2017，pycocotools，`scripts/eval.py`）

| 权重（release 资产名） | 架构 / 配置 | 训练 | 输入 | mAP50-95 | mAP50 | P / R |
|---|---|---|---|---|---|---|
| `yolov3-tiny-silu-softobj-e300-416` | YOLOv3-tiny-SiLU + IoU-aware objectness（推荐配置） | 300 ep 从零 · default 配方 | 416 | **16.52** | **30.44** | 0.4612 / 0.3216 |
| `yolov3-tiny-softobj-e300-416` | YOLOv3-tiny（原版算子）+ IoU-aware objectness | 300 ep 从零 | 416 | 16.07 | 29.56 | 0.4436 / 0.3150 |
| `yolo26n-from-scratch-e100-640` | YOLO26n · E2E（NMS-free） | 100 ep 从零 | 640 | 36.22 | 51.32 | 0.6342 / 0.4895 |
| 同上 · NMS（o2m） | | 同 run | 640 | 37.15 | 52.67 | 0.6554 / 0.4965 |
| `flash-yolo-e100-640` | Flash-YOLO（S2）· E2E（NMS-free） | 100 ep 从零 | 640 | 33.77 | 48.63 | 0.6220 / 0.4594 |
| 同上 · NMS（o2m） | | 同 run | 640 | 34.72 | 50.16 | 0.6254 / 0.4801 |

参考系：官方 darknet yolov3-tiny 权重同管线 @416 = **16.94 / 34.37**；官方 YOLO26n（O365 预训练
+ COCO 微调）= **40.27 / 55.80**（E2E）· **40.89 / 56.87**（NMS）。v3 两行与官方 v3 的 mAP50-95
差 0.42 / 0.87；mAP50 差 3.93 / 4.81。

## 2. 速度（2026-10-15 复测；model-only 推理段，不含前后处理）

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
  **不要按分数序逐位对照**——近分框在两精度间会互换（同 README 导出脚注的既有教训）。
- ORT-CPU 绝对数受宿主邻居负载影响（README 表在 load ≈40 时 b8 读到 8.27 / 6.09）——同机同
  协议可比，跨会话看趋势。

## 3. 部署产物（`runs/release-weights/`，与本页同批生成）

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

## 4. 溯源（随权重 metadata）

- 两份 v3 权重：header 自带 `arch/scale/nc/imgsz/names` + provenance
  （`version 0.5.0` · `commit 0baa0bf` · `describe v0.5.0-20-g0baa0bf`）
- `flash-yolo-e100-640`：自带 arch/scale/nc/imgsz（保存于 2026-10-11，无 provenance 字段）
- `yolo26n-from-scratch-e100-640`：无 metadata（保存于 2026-10-08）——消费时需显式
  `--model yolo26 --scale n --imgsz 640`

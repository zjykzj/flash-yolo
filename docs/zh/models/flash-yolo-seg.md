# Flash-YOLO-seg（实例分割）

> `flash-yolo` 的实例分割变体（`arch_task = segment`）。backbone/neck 与检测版**逐行相同**，
> 只把最后一行 `Detect` 换成 `Segment26`——与 `yolo26 → yolo26-seg` 是同一种关系。
> 结构定义：`config/models/flash-yolo-seg.yaml`；共享结构见
> [flash-yolo.md](flash-yolo.md)，姿态版见 [flash-yolo-pose.md](flash-yolo-pose.md)。

## 1. 命名与档位

- 家族名在前、任务后缀在后：`flash-yolo-seg`（不是 `flash-yolo-seg-n`）。
- 单档 `flash`：通道已材料化，没有 n/s/m/l/x 维度；`--scale` 不适用。
- 权重文件名可反推架构（`flash-yolo-seg.safetensors`），长名正则优先，不会被
  `flash-yolo` 前缀误吸。

## 2. 结构

25 行 yaml，与 `flash-yolo.yaml` 的差异**只有最后一行**：

```yaml
- [[17, 20, 23], 1, Segment26, [nc, 32, 64]]
```

`Segment26` = `Detect` + `cv4` 掩码系数分支 + `Proto26` 原型分支 + 训练期 semseg 辅助头
（与 yolo26-seg 共用一个实现）。三个检测层仍是 P3/P4/P5（idx 17/20/23）。

- `npr = 64`：`yolo26-seg.yaml` 写 256，n 档 `width=0.25` 经 `make_divisible(256*0.25,8)`
  得 64；flash 档 `width=1.0`，所以直接写生效值 64（写 256 会得到 4.68M / 47.8 GFLOPs 的错误结构）。
- `nm = 32` 是任务常数，不缩放。

| 指标 | flash-yolo | flash-yolo-seg | yolo26n-seg |
|---|---:|---:|---:|
| 参数 | 1,974,248 | **2,528,248** | 3,126,280 |
| GFLOPs @640 | 3.80 | **7.6** | 9.3 |
| 头参数 | 309,656 | 863,656 | 863,656 |

- backbone/neck 逐键同构（测试 `test_backbone_neck_identical_to_detect`）。
- 与 yolo26n-seg 的差异只省在 backbone/neck（−598,032 参数、−1.7 GFLOPs）；分割头逐键相同。
- GFLOPs 相对检测版约翻倍来自 Proto26 原型分支（官方 seg 同样如此）。

## 3. 复用的链路

| 环节 | 复用方式 |
|---|---|
| 前向契约 | E2E `(B,300,6+nm)` + proto `(B,32,160,160)`；raw `(B,4+nc+nm,N)` + proto |
| 训练模式 | 双分支 dict（`one2many`/`one2one`，o2o 的 proto/semantic 均 detach） |
| 损失 | `build_loss` 分派到 `ComputeLossSeg`（检测三项/归一化/ProgLoss 复用 + 掩码 BCE） |
| 验证 | 训练中 Box+Mask，best fitness = box + mask mAP50-95；正式口径走 `scripts/eval.py` |
| 权重 | metadata 写 `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz` |
| 引擎 | pt / onnx / trt 双输出（output0 检测+系数、output1 原型） |

除 `ARCHS`、损失分派表与权重名正则外，没有新增分支。

## 4. 权重下载

下载链接指向 **v1.0.0** 发布资产。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `flash-yolo-seg-e300-640` | 300 ep 从零 | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/flash-yolo-seg-e300-640.safetensors) |

- 权重 metadata 自带 `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz`，
  eval/infer/export 可自动解析。

## 5. 正式指标（COCO val2017 @640）

300 轮从零（default 配方、batch 32、nbs 64、fp32、mask_ratio 4、seed 0）：

| 训练 | Params | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| 从零 300 ep | 2,528,248 | 7.6 | 50.74 | **35.05** | 48.14 | **29.98** |

## 6. 推理延迟

> 口径：**model-only**（不含掩码解码/前后处理），输入 640；b1 = 单图，b8 = batch 8 每图平均。
> 引擎 init 已做 warmup。b1 为单发（含 H2D 与输出读回），b8 为流水线（每图）。

### 云服务器 / 工作站

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.57 ms | 0.271 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 2.47 ms | 0.136 ms |
| Xeon 8470Q（25 核配额） | ONNX Runtime 1.30 · fp32 | 14.9 ms | 10.8 ms |

- ORT 条件：intra_op=16 / inter_op=1；b1 固定图、b8 动态图；300 样本取 best-case min。
- TRT 条件：b1 固定引擎、b8 batch-8 引擎；30 次单发 + 100 次流水线。
- **b1 的 fp16 反而慢于 fp32**：proto 输出（32×160×160）的读回与 half→float 转换主导；
  纯 GPU 执行两者接近（0.73 vs 0.75 ms）。b8 才是 fp16 的优势区间（0.136 vs 0.271 ms）。
- ORT 绝对值受宿主邻居负载影响；b1 贴近启动开销地板，b8 才能反映算力差。

### 本地笔记本

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 4.07 ms | 1.85 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 3.77 ms | 0.702 ms |

- b1 为三轮 best-case min（笔记本 GPU 动态频率下单发漂移明显）；b8 稳定。

> 掩码解码与 IoU 目前主要在 CPU；部署时的端到端延迟显著高于 model-only，需单独测量。

## 7. 用法

```bash
python model/summary.py --model flash-yolo-seg
# -> 311 layers, 2,528,248 parameters, 7.6 GFLOPs

python scripts/train.py --data coco --model flash-yolo-seg --name flash-seg \
    --batch 32 --workers 8 --mask-ratio 4
python scripts/eval.py --weights flash-yolo-seg-e300-640.safetensors --data coco
python scripts/export.py --weights flash-yolo-seg-e300-640.safetensors --trt
python scripts/infer.py --weights flash-yolo-seg-e300-640.safetensors \
    --image assets/traffic.jpg
```

## 8. 注意

- 单档 `flash`，不要传 `--scale`。
- 分割训练由 loader 与验证主导，架构差异在吞吐上不可辨。
- 跨设备（CUDA vs CPU oneDNN）对齐会有 1e-3~4e-3 的原始输出差，掩码边界可能翻转；
  同运行时比较或走 TensorRT。

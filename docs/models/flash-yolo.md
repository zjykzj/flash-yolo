# Flash-YOLO（检测）

> 本仓库自有轻量架构（模型名 = 项目名）。目标：在 640 输入下 **GFLOPs 严格低于 YOLO26n
> （5.48）**，精度高于 YOLOv3-tiny 的收敛值，并且改动来自新的结构思路而非调参。
> 结构定义：`config/models/flash-yolo.yaml`；分割版见 [flash-yolo-seg.md](flash-yolo-seg.md)。

## 1. 设计命题

- **YOLOv3-tiny 的廉价空间下采样 × YOLO26 的 CSP 容量分配**；头（Detect / E2E）、损失、
  训练管线完全复用 YOLO26——改动只落在 backbone / neck。
- **六处 stride-2 3×3 卷积 → PoolConv**（`MaxPool2d(2,2) → 1×1 Conv`）：通道变换在
  半分辨率之后做，等通道下比 stride-2 3×3 卷积省 ~9× FLOPs。
- **stem 的 C3k2 → DW 3×3 + 1×1 投影**：P1/P2 段没有检测头，整块 CSP 容量是浪费；
  一个 depthwise 3×3 补回空间混合，几乎不花算力。
- 与 YOLO26n 的差异清单就两条：六处下采样 → PoolConv、stem → DW 3×3 + 1×1；
  C3k2 / SPPF / C2PSA / Detect 与三个检测层原样不动。
- 结构取舍的结论：DW stem 在同算力下有效；neck 缩容与 LiteBlock 均劣于定稿结构。

## 2. 定稿结构

25 行 yaml（比 YOLO26n 多一行：stem 拆成 DW 3×3 + 1×1），262 层模块树：

- 规模：**1,974,248 参数（1.97M，−23% vs YOLO26n）· 3.80 GFLOPs @640（−31%）**。
- 检测层 P3/P4/P5 与 Detect 头原样保留；权重键与 YOLO26 前缀一一对应，构建/引擎/训练
  链路零特判。
- 速度：b1 与 YOLO26n 接近（per-run 开销主导），b8 优势明显；完整延迟见 §5。

## 3. 权重下载

下载链接固定到 **v1.0.0**（下一个发布版本；发布前会 404，可先用 Releases 页面的当前版本）。
资产名固定，发新版时统一替换 tag 即可。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `flash-yolo-e300-640` | 300 ep 从零 | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/flash-yolo-e300-640.safetensors) |

- 权重 metadata 自带 `arch=flash-yolo` / `scale=flash` / `nc` / `imgsz` / `names`，
  eval/infer/export 可自动解析。

## 4. 正式指标（COCO val2017 @640）

| 训练 | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| 从零 300 ep | **35.14** | **50.15** | **36.09** | **51.56** | 1.97M | 3.80 |

## 5. 推理延迟

> 口径：**model-only**（不含前后处理），输入 640；b1 = 单图，b8 = batch 8 每图平均。
> 引擎 init 已做 warmup，数字为重复测量的稳态值；同一架构不同权重的延迟相同。

### 云服务器 / 工作站

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.31 ms | 0.210 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 1.27 ms | 0.123 ms |
| Xeon 8470Q（25 核配额） | ONNX Runtime 1.30 · fp32 | 13.2 ms | 5.84 ms |

- ORT 条件：intra_op=16 / inter_op=1；b1 固定图、b8 动态图；300 样本取 best-case min。
- TRT 条件：b1 固定引擎、b8 batch-8 引擎；30 次单发 + 100 次流水线。
- b1 贴近启动开销地板，b8 才能反映算力差；ORT 绝对值受宿主邻居负载影响。

### 本地笔记本

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| （CPU 型号待填） | ONNX Runtime · fp32 | 待测 | 待测 |

## 6. 用法

```bash
# 结构自检（无需数据）
python model/summary.py --model flash-yolo

# 训练
python scripts/train.py --data coco --model flash-yolo --name flash-coco-e300 --batch 32 --workers 8

# 评估（E2E / NMS）
python scripts/eval.py --weights flash-yolo-e300-640.safetensors --data coco
python scripts/eval.py --weights flash-yolo-e300-640.safetensors --data coco --nms

# 导出 / 推理
python scripts/export.py --weights flash-yolo-e300-640.safetensors --trt
python scripts/infer.py --weights flash-yolo-e300-640.safetensors \
    --image assets/traffic.jpg
```

## 7. 注意

- `flash-yolo` 只有 `flash` 单档，不要传 `--scale`。
- 结构改动只在 backbone/neck；新增算子（PoolConv/DW stem）在 `model/basic/` 与
  `model/build.py` 注册。
- 这是架构取舍的产物：精度低于 YOLO26n 是预期，不是训练 bug。

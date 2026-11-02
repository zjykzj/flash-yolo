# YOLOv3-tiny / YOLOv3-tiny-SiLU（检测）

> 本仓库的第二架构，也是“廉价空间下采样”思路的来源。结构基准 = **darknet 官方 cfg**
> （`cfg/yolov3-tiny.cfg`，论文 arXiv:1804.02767 谱系，原生 416 输入）。
> 结构定义：`config/models/yolov3-tiny.yaml` / `config/models/yolov3-tiny-silu.yaml`。

## 1. 口径声明

- **结构基准是 darknet，不是 ultralytics**。ultralytics 8.4.x 的 "v3"（资产
  `yolov3-tinyu.pt`）是无锚 DFL 重写（SiLU + reg_max=16，`Detect` 连 `anchors` 形参都没有），
  不能作为结构来源或数值对齐对象。
- **SiLU 变体**：`yolov3-tiny-silu` 与原版**逐行同构**，唯一差异 = LeakyReLU(0.1) → SiLU
  （激活无参数：state_dict 键集、参数量 8,852,366、@416 GFLOPs 均逐位一致）。
  官方 `.weights` 只配原版；两版由权重 metadata 的 `arch` 字段与文件名区分。
- 与 YOLO26 共享的只有训练管线（trainer/EMA/优化器/增强/数据加载/引擎）；模型、头、解码、
  损失、assigner 都是独立实现。

## 2. 结构（416 输入，21 个模块）

| 段 | idx | 内容 |
|---|---|---|
| Backbone | 0–13 | LeakyConv k3 s1 + MaxPool2d 下采样；13 层 1024 通道 |
| P5 支路 | 14–15 | 1×1 → 3×3，512 通道（P5/32 头输入） |
| Neck | 16–19 | route(-4) → 1×1 → Upsample ×2 → Concat(P4) → 3×3 |
| Head | 20 | V3Detect：每级 1×1 → 3·(5+nc)，双尺度 P4/16 + P5/32 |

- 规模：**21 层 / 8,852,366 参数（8.85M）/ 5.56 GFLOPs @416**。
- **保尺寸池化**：darknet `[maxpool] size=2 stride=1` 隐含右下 padding，PyTorch 写作
  `ZeroPad2d([0,1,0,1]) + MaxPool2d(2,1)`（idx 11+12）。不要用居中 padding。
- **两颗头卷积折进 V3Detect**（darknet cfg 末尾的独立卷积层），无 cv2/cv3 属性、
  无 `end2end` 开关、无 bias_init。
- **route 相对索引在 parse 期归一化**：第 16 层 yaml 写 `-2`（darknet `route -4`），
  组装器折算为绝对下标 14。

## 3. anchors（最大的坑）

每级 3 槽 × [w, h]，级序 = V3Detect 的 `from` 顺序（[P4/16, P5/32]）：

| 级 | stride | 槽位（像素） |
|---|---|---|
| P4/16 | 16 | [23,27] · [37,58] · [81,82] |
| P5/32 | 32 | [81,82] · [135,169] · [344,319] |

- 官方 mask 照原样保留（P5 档 3,4,5 / P4 档 1,2,3；81,82 两尺度共用、10,14 未启用）。
- **固定像素单位，绝不随 imgsz 缩放**。官方权重在多尺度训练下由 tw 自补偿输入尺度。
  实测（500 图 FastMetrics，官方权重）：416 原生 mAP50 0.384 / 640 原值 0.412 /
  640 线性缩放 **0.154**。
- `anchors` / `stride` 是 `persistent=False` buffer：不进 state_dict，随 `.to(device)` 搬运。
- 从零训练换输入尺寸才需要重聚类：`python scripts/compute_anchors.py --data <desc>`。
  重聚类后的锚点写进权重 metadata，`PtEngine` 构建后 `apply_meta` 覆盖 `head.anchors`。

## 4. 解码契约

唯一实现：`model/head/v3_detect.py::decode_level`（head 与 loss 共用，darknet `get_yolo_box`）：

```text
raw (B, 3·(5+nc), H, W)  通道序 = 逐锚 [x, y, w, h, obj, cls...]
xy = (σ(t) + grid) · stride        # 网格偏移 0（darknet 口径）
wh = anchor · exp(t)               # tw/th 取原始 logits，clamp(±8) 防 exp 溢出
```

- eval 前向输出：`(B, NA, 5+nc) = [x1,y1,x2,y2, σ(obj), σ(cls)...]`，
  NA = 3·ΣH·W（416 → 2535；640 → 6000）。
- **不要换成 v5 式 `(2σ−0.5)/(2σ)²`**：与官方权重、cv2.dnn 输出、33.1 AP 验收全部对不上。
- 最终置信 = **obj × cls**，在 `utils/postprocess.v3_detections` 里做；类感知 NMS
  （默认 conf 0.001 / iou 0.7 / max_det 300）。
- 无 E2E 路径：`--nms` / `--raw` 是 YOLO26 专属开关，对 v3 直接报错。

## 5. 训练（损失与匹配）

- **匹配（darknet 语义）**：逐 GT、逐级，在 3 个槽位里取形状 IoU 最大者为唯一正样本；
  责任格子 = GT 中心所在 cell。每 GT 每级至多一个正样本——没有 TAL/STAL、没有 o2m/o2o
  双头、没有 ProgLoss。
- **忽略区**：非正样本中预测解码框与任一 GT 的 IoU ≥ 0.7（`ignore_thresh`）的槽位 obj 不罚。
- **损失三项**：`BCE(obj)`（按非忽略槽位逐图归一）+ `BCE(cls)` 独热（仅正样本）+
  `CIoU`（仅正样本；box_gain 7.5 / cls_gain 0.5 / obj_gain 1.0）。
- 推荐训练配置：`--model yolov3-tiny-silu --obj-iou-target`（+ default 配方）；
  官方 `.weights` 只配原版。
- `--obj-iou-target`：正样本 obj 目标 = IoU(预测框, 责任 GT)、权重 ×5；默认 darknet 硬目标 0/1。

## 6. 权重下载与转换

下载链接指向 **v1.0.0** 发布资产。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `yolov3-tiny-e300-416` | 300 ep 从零 · 原版 + IoU-aware objectness | 416 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolov3-tiny-e300-416.safetensors) |
| `yolov3-tiny-silu-e300-416` | 300 ep 从零 · SiLU + IoU-aware objectness | 416 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolov3-tiny-silu-e300-416.safetensors) |

官方 darknet `.weights` 不随 Release 分发，需自行下载并转换：

```bash
python scripts/download_weights.py --model yolov3-tiny
python scripts/convert_weights.py --src weights/yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors
```

### darknet `.weights` 转换细节

- 文件布局：4 或 5 个 int32 头 + float32 小端张量流；头部宽度按文件长度自动判别，
  并要求文件恰好用尽。
- 张量序 = darknet cfg 层序：**P5 头卷积（512→255）在 P4 支路卷积之前**。
- 块内顺序：BN = `[bias(β), scale(γ), mean, var, conv 权重]`；输出层 = `[bias, 权重]`；
  conv 权重布局 `(out, in, kh, kw)`，无需转置。
- 映射以输入通道数对号：512ch → P5 头 / 256ch → P4 头（`_darknet_specs`，改 yaml 需同步）。
- 实测：官方文件 35,434,956 B = 8,858,734 floats + 20 B 头，严格加载通过。

## 7. 正式指标（COCO val2017）

| 权重 | 输入 | mAP50-95 | mAP50 |
|---|---|---:|---:|
| 官方 darknet 权重 | 640 | **17.16** | **35.90** |
| 官方 darknet 权重 | 416 | 16.94 | 34.37 |
| 从零 300 ep（SiLU + obj-iou） | 416 | 16.15 | 29.93 |
| 从零 300 ep（原版 + obj-iou） | 416 | 16.03 | 29.59 |

- 官方发布的 33.1 AP50 是 **COCO test-dev @416** 口径；同权重本管线 640 比 416 高，
  两者不可直接相减。
- 官方权重在 640 上更高来自锚值不缩放 + 有效分辨率更高。

## 8. 推理延迟

> 口径：**model-only**（不含前后处理），输入 **416**；b1 = 单图，b8 = batch 8 每图平均。
> 引擎 init 已做 warmup，数字为重复测量的稳态值；同一架构不同权重的延迟相同。

### 云服务器 / 工作站

| 权重 | 平台 / 运行时 | 精度 | b1 | b8（每图） |
|---|---|---|---:|---:|
| v3-SiLU + objectness | RTX 5090 · TensorRT 11.4 | fp32 | 0.68 ms | 0.133 ms |
| v3-SiLU + objectness | RTX 5090 · TensorRT 11.4 | fp16 | 1.19 ms | 0.048 ms |
| v3 原版 + objectness | RTX 5090 · TensorRT 11.4 | fp32 | 0.68 ms | 0.133 ms |
| v3 原版 + objectness | RTX 5090 · TensorRT 11.4 | fp16 | 1.15 ms | 0.049 ms |
| v3-SiLU + objectness | Xeon 8470Q（25 核配额） · ONNX Runtime 1.30 | fp32 | 4.82 ms | 3.74 ms |
| v3 原版 + objectness | Xeon 8470Q（25 核配额） · ONNX Runtime 1.30 | fp32 | 4.52 ms | 3.61 ms |

- ORT 条件：intra_op=16 / inter_op=1；b1 固定图、b8 动态图；300 样本取 best-case min。
- TRT 条件：b1 固定引擎、b8 batch-8 引擎；30 次单发 + 100 次流水线。
- **v3 的 fp16 在 b1 反而慢 ~70%**（小图由调度开销主导）→ b1 用 fp32、b8 用 fp16。
- b1 贴近启动开销地板，b8 才能反映算力差；ORT 绝对值受宿主邻居负载影响。

### 本地笔记本

| 权重 | 平台 / 运行时 | 精度 | b1 | b8（每图） |
|---|---|---|---:|---:|
| v3-SiLU + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp32 | 1.53 ms | 0.620 ms |
| v3-SiLU + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp16 | 1.97 ms | 0.197 ms |
| v3 原版 + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp32 | 1.47 ms | 0.615 ms |
| v3 原版 + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp16 | 1.99 ms | 0.193 ms |

- b1 为三轮 best-case min（笔记本 GPU 动态频率下单发漂移明显）；b8 稳定。
- fp16 在 b1 反而慢（1.99 vs 1.47 ms，与 5090 同现象）→ b1 用 fp32、b8 用 fp16。

## 9. 用法

```bash
# 官方权重：下载 + 转换 + 评估
python scripts/download_weights.py --model yolov3-tiny
python scripts/convert_weights.py --src weights/yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors
python scripts/eval.py --weights weights/yolov3-tiny.safetensors --data coco

# 从零训练（推荐 SiLU + obj-iou-target；416 原生输入）
python scripts/train.py --data coco --model yolov3-tiny-silu --imgsz 416 \
    --obj-iou-target --name v3tiny-e300 --batch 32 --workers 8

# 已发布权重：评估 / 推理
python scripts/eval.py --weights yolov3-tiny-silu-e300-416.safetensors --data coco
python scripts/infer.py --weights yolov3-tiny-silu-e300-416.safetensors --image assets/traffic.jpg

# 锚点覆盖判定 / 重聚类
python scripts/compute_anchors.py --data coco
```

## 10. 注意

- **416 是规范输入**；`config/train.yaml` 默认 640，复现 v3 协议需显式 `--imgsz 416`。
- 结构固定，不要传 `--scale`（告警且不生效）。
- 官方 `.weights` 只配原版 `yolov3-tiny`；SiLU 变体需要自己训练或用对应产物。
- 锚点绝不随 `imgsz` 线性缩放；从零训练换尺寸才重聚类。

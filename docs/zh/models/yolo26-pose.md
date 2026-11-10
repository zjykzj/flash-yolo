# YOLO26-pose（姿态估计）

> 复现对象：官方 `yolo26n-pose` 权重经本仓库管线 **0.5687** pose AP / **0.8250** AP50
> （COCO-pose val2017，2,346 图，pycocotools OKS）；同数据 ultralytics 默认 val（NMS）
> 为 0.570 / 0.831，E2E 为 0.572 / 0.833。
> 结构定义：`config/models/yolo26-pose.yaml`；检测/分割共享结构见 [yolo26.md](yolo26.md) /
> [yolo26-seg.md](yolo26-seg.md)。

## 1. 口径声明

- 官方 pose checkpoint 是 **Objects365 检测预训练 + COCO-pose 400 轮微调**
  （官方 `train_args.pretrained = yolo26n-objv1-150.pt`、`epochs = 400`、`batch = 128`）。
  从零训练用本仓库配方可以跑通，但**与官方 pose 数字不是 like-for-like**。
- COCO-pose 的评估协议是 **2,346 张有有效关键点的图**，不是检测的 5,000 张
  `val2017`；`scripts/eval.py` 会自动按 `num_keypoints > 0` 过滤。
- 指标 = pycocotools `iou_type="keypoints"`（OKS，maxDets=20）；表头是
  AP / AP50 / AP75 / AP(M) / AP(L) / AR，没有 bbox 表的 P/R 与 small bucket。
- 「官方」指 Ultralytics 发布的 pose 权重/配方/指标，本仓库只提供下载与转换脚本，不分发。

## 2. 结构（n 档）

backbone/neck 与 [yolo26.md](yolo26.md) 的检测版**逐行相同**（官方 yaml 只换最后一行）；
头换成 `Pose26` = `Detect` + 关键点分支：

| 件 | 内容 |
|---|---|
| `cv4` / `one2one_cv4` | 每级两个 3×3 Conv；`c4 = max(ch[0]//4, K·(D+2))`，n 档 85 |
| `cv4_kpts` | 每级 1×1 → `nk = K·D`（COCO 17×3 = 51），x/y/v 交错 |
| `cv4_sigma` | 每级 1×1 → `2K = 34`；训练期 RLE 的不确定度，推理/导出不用 |
| `flow_model`（RealNVP） | 6 层 coupling 的归一化流；只服务训练期 RLE，权重随 ckpt 走 |
| `one2one_cv4*` | `end2end=True` 时对以上三支的深拷贝（o2o 分支） |

- n 档合计 **3,679,464 参数 / 363 层 / 7.6 GFLOPs @640**（模块树口径，与官方
  `sum(p.numel())` 一致；官方 yaml 注释里的 3,747,554 是旧值）。
- `kpt_shape = (17, 3)` 是结构参数：n 档 17 个关键点 × (x, y, visibility)；
  随权重 metadata 的 `kpt_shape` 走，`PtEngine`/导出/训练建图时按它重建头。
- bias_init 只覆盖 `cv2/cv3`；`cv4` 不设先验（官方同口径）。

## 3. 推理与导出契约

- **E2E（默认）**：`(B, 300, 6 + nk)` = `[x1,y1,x2,y2,score,cls, kpts...]`；关键点与框
  共享**同一次**两阶段 top-k（`e2e_topk` 的 extra 机制）。
- **raw / NMS**：`(B, 4 + nc + nk, N)`，内容是**已解码**像素 xywh + sigmoid 分数 + 解码
  关键点（官方 pose `_inference` 口径）。注意这与检测头 raw 的“原始 ltrb+logits”不同：
  关键点解码需要 anchors/strides，留在图内更简单，也让 `--raw` 输出与官方 pose 逐位可比。
- 关键点解码：`(t + anchor) * stride`，v 通道 sigmoid；没有 v8 Pose 的
  `2t - 0.5` 口径。
- 导出图只保留推理分支：`cv4_sigma` / `flow_model` 是训练专用，不进 ONNX/TRT。
- ONNX metadata / TRT 旁车 JSON 带 `arch/scale/nc/imgsz/names/kpt_shape/end2end`；
  `infer.py` / `eval.py` 不再需要原 `.safetensors`。

## 4. 权重下载与初始化

官方 pose 权重不随 Release 分发，自行下载并转换：

```bash
python scripts/download_weights.py --model yolo26n-pose
python scripts/convert_weights.py --src weights/yolo26n-pose.pt \
    --dst weights/yolo26n-pose.safetensors
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose
```

- 转换器写入 `arch=yolo26-pose` / `scale` / `nc=1` / `names=["person"]` /
  `kpt_shape=[17,3]`；strict load 全 key 匹配（含 `cv4*` 与 `flow_model`）。
- 微调从官方 pose 权重开始时，直接 `--weights`；从 O365 **检测**权重初始化时，
  pose 头里的 `cv4/cv4_kpts/cv4_sigma/flow_model` 不存在，`load_weights_finetune` 的
  `allow_missing_head` 会把它们随机初始化，其余 key 逐位加载。

## 5. 正式指标（COCO-pose val2017，2,346 图）

| 训练 / 实现 | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|
| 官方权重 · 本仓库 E2E | **0.5687** | 0.8250 | 0.6176 | 0.4814 | 0.6924 | 0.6350 |
| 官方权重 · 本仓库 NMS | **0.5670** | 0.8280 | 0.6116 | 0.4774 | 0.6923 | 0.6303 |
| 官方权重 · ultralytics E2E（同 2,346 图） | 0.572 | 0.833 | 0.618 | 0.499 | 0.689 | 0.635 |
| 官方权重 · ultralytics NMS（默认 val） | 0.570 | 0.831 | 0.614 | 0.496 | 0.686 | 0.631 |

- 本仓库与 ultralytics 的差异来自 square letterbox（官方 val 是 auto/rect）与指标实现
  噪声；同输入 forward 本身逐位一致（见
  [../development/alignment.md](../development/alignment.md)）。
- 官方训练期 validator 记录 pose 0.5717 / AP50 0.8332；与本管线同量级。
- **尚无完整 400 轮 / 从零全长 pose 结果**；本页不把短跑数字当正式指标。

## 6. 推理延迟（model-only，b1，640）

> 口径：`engine.predict_timed` 的 inference 段，连续 30 次取中位数；引擎 init 已 warmup。
> GPU / TensorRT 为 RTX 3090 + TensorRT 11.4；CPU 行为 Xeon 8470Q（25 核配额）。
> b1 是 per-run 开销地板，不能直接外推到 b8。

| 运行时 / 精度 | b1 |
|---|---:|
| TensorRT 11.4 · fp16 | 1.65 ms |
| TensorRT 11.4 · fp32 | 2.22 ms |
| ONNX Runtime 1.30 · CPU fp32 | 12.4 ms |
| PyTorch · CUDA fp32 | 10.5 ms |
| PyTorch · CPU fp32 | 47.3 ms |

- TensorRT 引擎与构建机 GPU/TRT 版本绑定；换机需重跑 `scripts/export.py --trt`。
- b8 表尚未发布；同一架构的 batch 缩放规律与 [yolo26.md](yolo26.md) 的检测版一致
  （额外成本来自关键点分支）。

## 7. 训练

- 数据：`config/datasets/coco-pose.yaml`（COCO keypoints json；17 点，`nc=1`）；
  描述符字段 `kpt_shape` / `flip_idx` / `kpt_names` / `skeleton`，见
  [../datasets/README.md](../datasets/README.md)。
- 损失：检测三项（box/cls/l1 + ProgLoss）复用 `ComputeLoss`，加
  `pose`（KeypointLoss，COCO OKS sigmas）、`kobj`（可见性 BCE）、`rle`（RealNVP +
  RLELoss）；默认增益 `pose_gain=24` / `kobj_gain=4` / `rle_gain=1`。
- 训练中验证同时算 Box + Pose；best fitness = box mAP50-95 + pose mAP50-95
  （官方 `PoseMetrics.fitness` 口径）；正式数字仍以 `scripts/eval.py` 为准。
- 官方配方：`config/recipes/yolo26-pose-coco.yaml`——400 轮、batch 128、nbs 64、
  lr0 0.005、lrf 0.1、cls_w 2、o2m 1.0、muon_w/sgd_w 0.5、mosaic 1.0、fliplr 0.5。
- 从零训练建议 `--no-amp`（官方 `amp=true` 依赖 O365 预训练初始化；检测从零 AMP 的
  NaN 机制同样适用）。
- pose 下 `copy_paste > 0` 会直接报错（官方 pose 配方为 0，且没有分割多边形）；
  `fliplr` 需要描述符里的 `flip_idx`，否则翻转后的关键点语义会错。

```bash
# 微调官方 pose 权重（建议先 10 轮验证链路）
python scripts/train.py --data coco-pose --model yolo26-pose \
    --weights weights/yolo26n-pose.safetensors \
    --recipe yolo26-pose-coco --epochs 10 --batch 32 --device 0

# 从零（不与官方 like-for-like；先关 AMP）
python scripts/train.py --data coco-pose --model yolo26-pose \
    --recipe yolo26-pose-coco --no-amp --epochs 100 --batch 32 --device 0,1
```

## 8. 用法

```bash
# 评估（E2E / NMS）
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose --nms

# 推理：画骨架 + 写 YOLO-pose 标签
python scripts/infer.py --weights weights/yolo26n-pose.safetensors \
    --image /path/to/person.jpg --data coco-pose

# 导出：默认 E2E 单输出 (B,300,57)；--raw 出 (B,56,8400) 解码输出
python scripts/export.py --weights weights/yolo26n-pose.safetensors
python scripts/export.py --weights weights/yolo26n-pose.safetensors --raw
python scripts/export.py --weights weights/yolo26n-pose.safetensors --trt --fp16
```

训练参数、DDP、resume 见 [../pipeline/train.md](../pipeline/train.md)；评估口径见
[../pipeline/eval.md](../pipeline/eval.md)；导出契约见
[../pipeline/export.md](../pipeline/export.md)。

## 9. 注意

- `kpt_shape` 是结构参数：自定义关键点数据集必须先改描述符/权重 metadata，不能只改
  `nc`。
- `cv4_sigma`/`flow_model` 只在训练期使用；验证的 `val_rle` 恒 0（验证前向不返回
  sigma），不代表 RLE 未训练。
- `--single-cls` 对 pose 不适用（COCO-pose 已经是 `nc=1`），CLI 会直接报错。
- COCO-pose val 是 2,346 图；报告数字时不要与检测的 5,000 图混用。
- 官方 pose 权重由 O365 检测预训练 + 400 轮 COCO-pose 微调得到；从零数字只作本仓库基线。

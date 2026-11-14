# Flash-YOLO-pose（姿态估计）

> `flash-yolo` 的轻量 backbone/neck + `Pose26` 头；单档 `flash`，@640 **3.08M 参数 /
> 5.9 GFLOPs**。姿态头与 [yolo26-pose.md](yolo26-pose.md) 完全同构（关键点 + 可见性 +
> 训练期 RLE），检测/分割结构见 [flash-yolo.md](flash-yolo.md) /
> [flash-yolo-seg.md](flash-yolo-seg.md)。
> 当前状态：结构 / 训练 / 评估 / 导出 / 推理链路已接入；**从零 300 轮正式结果 E2E
> 0.5004 / NMS 0.5055**（COCO-pose val2017，2,346 图）。

## 1. 结构

- backbone/neck 与 `flash-yolo` 逐行相同（25 行 yaml 的前 24 行），只把最后的 `Detect`
  换成 `Pose26`：`config/models/flash-yolo-pose.yaml`。
- `Pose26` = `Detect` + `cv4`（每级两个 3×3 Conv，`c4 = max(ch[0]//4, K·(D+2)) = 85`）
  + `cv4_kpts`（1×1 → 51，x/y/v 交错）+ `cv4_sigma`（训练期 RLE，2K=34）
  + `flow_model`（RealNVP，训练期 RLE）；`end2end=True` 时另有 `one2one_*` 深拷贝。
- 头输入通道 = `[64, 128, 256]`（P3/P4/P5）；模块树 **365 层**，其中头 1,416,840 参数。
- `cv4_sigma` / `flow_model` 只服务训练，不进 ONNX/TRT 的推理图。
- `kpt_shape = (17, 3)` 是结构参数，随权重 metadata 的 `kpt_shape` 走；自定义关键点数据集
  必须同时改描述符/权重 metadata，不能只改 `nc`。

## 2. 契约

- **E2E（默认）**：`(B, 300, 6 + nk)` = `[x1,y1,x2,y2,score,cls, kpts...]`；关键点与框共享
  同一次两阶段 top-k。COCO 17 点下 `nk=51`，即 `(B, 300, 57)`。
- **raw / NMS**：`(B, 4 + nc + nk, N)`，内容是**已解码**像素 xywh + sigmoid 分数 + 解码
  关键点；COCO 下 `(B, 56, 8400)`，NMS 在外。与检测头 raw 的 ltrb+logits 不同。
- 关键点解码 `(t + anchor) * stride`，v 通道 sigmoid；没有 v8-Pose 的 `2t - 0.5` 口径。
- 训练中验证同时算 Box + Pose；best fitness = box mAP50-95 + pose mAP50-95；正式数字仍以
  `scripts/eval.py` 为准。

## 3. 正式指标（从零 300 轮）

口径：COCO-pose val2017 2,346 图，pycocotools OKS，maxDets=20；default 配方、batch 32、
nbs 64、fp32、单卡；best = ep300。

| 训练 / 路径 | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|
| 从零 300 轮 · E2E | **0.5004** | 0.7832 | 0.5285 | 0.4024 | 0.6390 | 0.5681 |
| 从零 300 轮 · NMS | **0.5055** | 0.7884 | 0.5380 | 0.4051 | 0.6450 | 0.5669 |
| 参考：yolo26n-pose 从零 300 轮 E2E | 0.5529 | 0.8200 | 0.5947 | 0.4686 | 0.6748 | 0.6193 |
| 参考：yolo26n-pose 从零 300 轮 NMS | 0.5541 | 0.8249 | 0.5984 | 0.4704 | 0.6747 | 0.6166 |

- 同预算下 flash 比 yolo26n-pose 低 **5.25 AP（E2E）/ 4.86 AP（NMS）**，相对约 −9%；代价是
  参数少 16.3%（3.08M vs 3.68M）、GFLOPs 少 22.4%（5.9 vs 7.6）。
- 10 轮从零时差距为 3.54 AP（E2E：0.2719 vs 0.3073），300 轮扩大到 5.25 AP；差距主要来自
  架构容量，训练预算拉长不会自动收窄。
- 训练中 FastPoseMetrics 已按 COCO keypoint 口径修正（忽略 `num_keypoints==0`、OKS 用标注
  `area`）；同一 best 复算为 0.4791，与正式 0.5004 差约 0.02，AR@20 与正式一致。

## 4. 训练

从零训练用内置 `default` 配方（默认 100 轮；本次正式 run 用 300 轮）——
`yolo26-pose-coco` 是官方 pose 权重的 400 轮微调配方，依赖 O365 预训练起点，不适合直接
拿来从零训 flash 结构：

```bash
# 结构自检（无需数据）
python model/summary.py --model flash-yolo-pose

# 从零训练（COCO-pose；从零默认 fp32）
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --epochs 300 --batch 32 --workers 12 --no-amp \
    --name flash-pose-scratch-300ep

# 可选：从 flash-yolo 检测权重初始化 backbone/neck；pose 头随机初始化
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --weights flash-yolo-e300-640.safetensors \
    --name flash-pose-ft --batch 32 --workers 8 --amp
```

- 从检测权重初始化时，`cv4/cv4_kpts/cv4_sigma/flow_model` 在检测 ckpt 中不存在，
  `load_weights_finetune(..., allow_missing_head=True)` 会随机初始化它们，backbone/neck
  逐位加载（见 `tests/test_finetune_load.py`）。
- pose 下 `copy_paste > 0` 会直接报错；`fliplr` 使用描述符里的 `flip_idx`。
- FastPoseMetrics 的口径修正与复算见 §3；训练日志只用于趋势与 best 选取，正式数字用
  `scripts/eval.py`（与 [yolo26-pose.md](yolo26-pose.md) 同口径）。

## 5. 推理延迟（model-only，640）

> 口径：`engine.predict_timed` 的 inference 段，每轮连续 30 次取中位数、共 5 轮取 best-case
> min；引擎 init 已 warmup。b1 = 单图，b8 = batch 8 每图平均（batch-8 引擎）；b1 是 per-run
> 开销地板，不能直接外推到 b8。权重 = 从零 300 轮产物 `best.safetensors`（同一架构不同权重
> 延迟相同）。

### 云服务器 / 工作站

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | — | — |
| RTX 5090 | TensorRT 11.4 · fp16 | — | — |
| Xeon 8470Q（25 核配额） | ONNX Runtime 1.30 · fp32 | — | — |

- 5090 / 8470Q 行待服务器测量后回填（ORT 条件同 [yolo26.md](yolo26.md) §5）。

### 本地笔记本

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 3.61 ms | 2.08 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 4.37 ms | 1.22 ms |
| RTX 4060 Laptop | PyTorch · CUDA fp32 | 13.8 ms | — |
| Ultra 9 185H（22 线程） | PyTorch · CPU fp32 | 54.9 ms | — |

- b1 为多轮 best-case min（笔记本 GPU 动态频率下单发漂移明显）；b8 稳定。
- fp16 的 b1 慢于 fp32（笔记本小图调度开销主导，与 [yolov3-tiny.md](yolov3-tiny.md) 同现象）；
  b8 才反映算力差。
- TensorRT 引擎与构建机 GPU/TRT 版本绑定；换机需重跑 `scripts/export.py --trt`。
- 构建命令：`scripts/export.py --weights <best.safetensors> --trt [--fp16]`（E2E 单输出
  (B,300,57)）；batch-8 引擎按 b8 口径另行构建。

## 6. 评估 / 导出 / 推理

```bash
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose --nms

python scripts/export.py --weights runs/train/<run>/weights/best.safetensors
python scripts/export.py --weights runs/train/<run>/weights/best.safetensors --raw
python scripts/infer.py --weights runs/train/<run>/weights/best.safetensors \
    --image assets/traffic.jpg --data coco-pose
```

- COCO-pose val 只评估 **2,346 张有有效关键点的图**（不是 5,000 张检测 val）；pycocotools
  OKS，maxDets=20。
- 导出 metadata 带 `arch=flash-yolo-pose / scale / nc / imgsz / names / kpt_shape / end2end`；
  infer/eval 不再需要原 `.safetensors`。
- `--nms` 支持（pose 的 o2m 分支）；`--single-cls` 对 pose 不适用。

## 7. 注意

- 只有 `flash` 单档，不要传 `--scale`。
- 官方 `yolo26n-pose` 的参考指标（0.5687 / 0.8250 E2E）**不适用于**本结构；本页的 300 轮
  从零数字见 §3，与 yolo26n-pose 同预算对照也已列出。
- 结构参数：3,081,432 参数、5.9 GFLOPs @640；与 `flash-yolo` 检测版共用前 24 行结构。
- 300 轮从零权重为本地训练产物，未随 Release 分发。
- 训练中 `val_rle` 恒 0（验证前向不返回 sigma），不代表 RLE 未训练。

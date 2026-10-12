# flash-yolo 设计说明（第一轮筛选 S0–S4 与定稿结构）

> 定位：本项目自有轻量架构（**模型名 = 项目名**）。本文记录**创建过程**：设计目标、设计命题、
> 第一轮筛选（S0–S4）的全部实验与结论、定稿结构事实与速度对拍。
> 正式训练/评估数字（100 轮从零 run）与对照表见 README 的 Results 一节。

## 1. 设计目标（立项口径，2026-10-09）

- 输入 640 与 yolo26n 对齐；**GFLOPs 必须严格低于 yolo26n（5.48）**；
- 精度允许低于 yolo26n，但**必须高于 yolov3-tiny 的收敛值**（v3 是当时的精度下界）；
- 要有**新的架构思路**（新算子/block/借鉴既有谱系），不是改配置刷点；
- 方法 = **单变量筛选**：先在同一预算下横向筛结构（每处改动只动一个变量），胜者才投入全长训练。

## 2. 设计命题（v1 主设计 = S1）

- **v3-tiny 的廉价空间下采样哲学 × yolo26 的 CSP 容量分配**；头（Detect / E2E）、损失、
  训练管线完全复用 yolo26 —— 改动只落在 backbone / neck。
- **六处 stride-2 3×3 卷积 → PoolConv**（`MaxPool2d(2,2) → 1×1 Conv`）：通道变换在半分辨率
  之后做，等通道下比 stride-2 3×3 卷积省 **~9× FLOPs**；RGB 入口的那处 3×3 s2 保留。
  位置 = yolo26n 的 L1 / L3 / L5 / L7 / L17 / L20 —— 即 P2→P5 的下采样四处 + neck 的两处
  （定稿结构里因 stem 拆成两行，对应 idx 1 / 4 / 6 / 8 / 18 / 21）。
- **stem 的 C3k2 → 1×1 投影**：P1/P2 段那个块没有下游消费者（没有 P1/P2 检测头），
  整块 CSP 容量是浪费。
- 预期代价：约 −1.5 GFLOPs（设计期对六处 PoolConv 的估算 −1.46；加上 stem 简化后定稿实测
  3.80，见第 4 节）。

## 3. 第一轮筛选（2026-10-09 ~ 10-10）

设置：5 个 20 轮 run，640 输入、batch 32、同 default 配方、同 seed；指标 = 训练中 FastMetrics
（快速近似口径，**只用于相对排序**，每处改动单变量）；5 个 run 的 best 都出现在最后一轮
（20 轮未收敛，绝对数字不作数，看的是相对差）。

| Run | 设计 | GFLOPs | mAP@[.5:.95] | mAP@50 |
|---|---|---|---|---|
| S0 | YOLO26n（对照） | 5.48 | 29.22 | 42.45 |
| S1 | flash-yolo 主设计（六处 PoolConv + 1×1 stem） | 3.79 | 26.25 | 38.92 |
| **S2** | S1 + DW stem | 3.80 | **26.92** | 39.67 |
| S3 | S1 + neck 通道 ×0.75 | 3.09 | 25.00 | 37.66 |
| S4 | S1 + LiteBlock（neck） | 3.00 | 21.83 | 34.55 |

```bash
# 筛选 run（每个 ~4 h，单张 RTX 5090）；s1–s4 即 --cfg 语料（flash-yolo.yaml = 定稿 S2）
python scripts/train.py --data coco --model yolo26 --scale n --name fy-s0-yolo26n --batch 32 --workers 8 --imgsz 640 --epochs 20
python scripts/train.py --data coco --model flash-yolo --cfg config/models/flash-yolo-s1.yaml --name fy-s1-flash-yolo --batch 32 --workers 8 --imgsz 640 --epochs 20
python scripts/train.py --data coco --model flash-yolo --cfg config/models/flash-yolo-s2.yaml --name fy-s2-dwstem --batch 32 --workers 8 --imgsz 640 --epochs 20
python scripts/train.py --data coco --model flash-yolo --cfg config/models/flash-yolo-s3.yaml --name fy-s3-nk75 --batch 32 --workers 8 --imgsz 640 --epochs 20
python scripts/train.py --data coco --model flash-yolo --cfg config/models/flash-yolo-s4.yaml --name fy-s4-lite --batch 32 --workers 8 --imgsz 640 --epochs 20
```

**结论**：

1. **DW stem 有效**：同算力下 S2 比 S1 高 **+0.67 mAP**（26.92 vs 26.25，3.80 vs 3.79 GFLOPs）
   —— P1/P2 段的池化下采样丢掉了空间混合能力，一个 depthwise 3×3 就补回来，且几乎不花算力。
2. **neck 通道 ×0.75 是有效的"更省"路线**：S3 用 −18% GFLOPs 换 −1.9 点（相对 S2）——
   想要更小模型时它是备选，本轮不作为主设计。
3. **LiteBlock 在 neck 失败**：同算力下 S4 比 S3 低 3.2 点（21.83 vs 25.00）——
   MobileNeXt 式"低维瓶颈 + DW"在 neck 的通道数下不成立，淘汰。
4. 定稿选 **S2**（DW stem 版）；S1 存为旧主设计，s1–s4 全部保留为筛选语料（`--cfg` 可精确复现）。

## 4. 定稿结构（`config/models/flash-yolo.yaml` = S2）

25 行 yaml（比 yolo26n 多一行：stem 的 C3k2 拆成 DW 3×3 + 1×1 两步）：

| idx | from | 层 | 参数 | 说明 |
|---|---|---|---|---|
| 0 | -1 | Conv [16, 3, 2] | 464 | P1/2（RGB 入口，保留 3×3 s2） |
| 1 | -1 | PoolConv [32] | 576 | P2/4 |
| 2 | -1 | DWConv [32, 3, 1] | 352 | **stem：depthwise 空间混合（筛选胜出的一刀）** |
| 3 | -1 | Conv [64, 1, 1] | 2,176 | stem 通道投影 |
| 4 | -1 | PoolConv [64] | 4,224 | P3/8 |
| 5 | -1 | C3k2 [128, False, 0.25] | 26,080 | |
| 6 | -1 | PoolConv [128] | 16,640 | P4/16 |
| 7 | -1 | C3k2 [128, True] | 87,040 | |
| 8 | -1 | PoolConv [256] | 33,280 | P5/32 |
| 9 | -1 | C3k2 [256, True] | 346,112 | |
| 10 | -1 | SPPF [256, 5, 3, True] | 164,608 | |
| 11 | -1 | C2PSA [256] | 249,728 | |
| 12 | -1 | Upsample | 0 | |
| 13 | [-1, 7] | Concat | 0 | ← backbone P4 |
| 14 | -1 | C3k2 [128, True] | 119,808 | |
| 15 | -1 | Upsample | 0 | |
| 16 | [-1, 5] | Concat | 0 | ← backbone P3 |
| 17 | -1 | C3k2 [64, True] | 34,304 | (P3/8) |
| 18 | -1 | PoolConv [64] | 4,224 | |
| 19 | [-1, 14] | Concat | 0 | |
| 20 | -1 | C3k2 [128, True] | 95,232 | (P4/16) |
| 21 | -1 | PoolConv [128] | 16,640 | |
| 22 | [-1, 11] | Concat | 0 | ← backbone P5 / C2PSA |
| 23 | -1 | C3k2 [256, True, 0.5, True] | 463,104 | (P5/32, attn) |
| 24 | [17, 20, 23] | Detect [nc] | 309,656 | 三检测层 P3/P4/P5 全保留 |

- 规模：**1,974,248 参数（1.97M，−23% vs yolo26n）· 3.80 GFLOPs @640（−31%）**；
  模块树 262 层（yolo26n 为 260）。
- 与 yolo26n 的差异清单就两条：**六处下采样 → PoolConv**、**stem → DW 3×3 + 1×1**；
  C3k2 / SPPF / C2PSA / Detect 与三个检测层原样不动（所以权重键与 yolo26 的前缀一一对应，
  构建/引擎/训练链路零特判）。

**速度对拍（S1 vs S2，2026-10-10；b1 @640，随机权重，E2E 融图，`predict_timed` 推理段）**：

| 口径 | S1 | S2（定稿） |
|---|---|---|
| TensorRT fp32 | 1.32 ms | 1.33 ms |
| ORT-CPU（intra_op=16） | 10.1 ms | 9.8 ms |

→ **打平**（DW stem 零速度代价）→ 定稿选精度更高的 S2。

**b1 速度验收（2026-10-09，对照 yolo26n 同条件）**：flash-yolo 全面更快 ——
torch-CPU +10.9% · ORT-CPU +45% · CUDA-graph +11% · TensorRT +12.6% · eager ≈ 平
（eager 下 260 层 Python 发射开销掩盖差异，不是模型劣势）。

**ORT-CPU 复测（2026-10-11，定稿权重 + E2E 融图，b1/b8，README 表口径）**：交替 A/B、best-of-N
两轮重复 —— b1 **14.5 vs 17.5 ms**、b8 **5.6 vs 13.0 ms/img**（flash-yolo vs yolo26n）。
两点注意：① 测量时 v3-darknet 训练占着 16 个 loader worker（25 核配额），**绝对值偏悲观、
b1 差距最不稳**（跨轮次 11–18 ms），表中已按"临时值"标注、待空载复测；② 10-09 那条 "+45%" 是
空载随机权重口径，今天的受限环境下不可复现 —— 两个口径不要混用。b8 的 2.3× 优势在两轮独立测量里
稳定（44.3/45.2 vs 103.5/105.7 ms），且方向与 TRT b8 一致。

**ORT-CPU 空载复测（2026-10-12，容器空载、宿主仍共享）**：四轮 × 300 样本交替 A/B
（脚本存档 `runs/artifacts/bench/ort_bench3.py`），取每轮 best-case min 的中位数 ——
b1 **13.2 vs 14.8 ms**、b8 **6.09 vs 8.27 ms/img**（flash-yolo vs yolo26n）。对 10-11 的修正：
① 那次的 "5.6 vs 13.0 / 2.3×" 是训练占 CPU 时测的，**两个模型对负载的敏感度不同**（yolo26n 算子
更多 → 线程同步点更多 → 被噪声放大更多），空载下 b8 差距回到 **−26%**，与 GFLOP 比
5.48/3.80 = 1.44（−31%）同量级、方向与 TRT b8（−14% fp32 / −9% fp16）一致；② b1 在两种口径下
都近似打平（差 1–3 ms，跨轮次符号会翻）——b1 是 per-run 开销主导，不反映算力差。
残留噪声：宿主共享，绝对数随邻居负载 ±20%（本次 load ≈ 40，10-09 空载同架构实测 b1 ≈ 10 ms，
今天最好也只到 11.3 ms）。

## 5. 全长跑与验收（2026-10-11）

100 轮从零训练（batch 64，与 yolo26n 控制同配方/同预算，~15.5 h）：

| 路径 | mAP@[.5:.95] | mAP@50 |
|---|---|---|
| YOLO26n 控制（同预算） | 36.22 | 51.32 |
| **flash-yolo · E2E（NMS-free）** | **33.77** | **48.63** |
| **flash-yolo · NMS（o2m）** | **34.72** | **50.16** |

- −2.45 点（E2E）/ −2.43 点（NMS），与筛选期预测的 −2.30 吻合；代价换来 −31% GFLOPs、
  −23% 参数、−15% batch-8 fp32 延迟。
- 两侧末 3 轮 in-training mAP 都平（≤0.0004 变化）→ 差距是**架构取舍**，不是预算不足。

## 6. 复现

```bash
# 全长训练与正式评估（100 轮，同 yolo26n 控制配方/预算）
python scripts/train.py --data coco --model flash-yolo --name flash-coco-e100 --batch 64 --workers 16
python scripts/eval.py --weights runs/train/flash-coco-e100/weights/best.safetensors --data coco
python scripts/eval.py --weights runs/train/flash-coco-e100/weights/best.safetensors --data coco --nms
```

筛选 run 的命令见第 3 节；`config/models/flash-yolo-s1.yaml` … `-s4.yaml` 是筛选语料，
`flash-yolo.yaml` 是定稿结构（`--cfg` 可指任意一个精确复现对应的 run）。

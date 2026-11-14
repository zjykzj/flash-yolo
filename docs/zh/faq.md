# 常见问题（FAQ）

> 跨页面的“为什么 / 怎么办”在这里回答；分页细节见
> [datasets](datasets/README.md) · [models](models/README.md) ·
> [train](pipeline/train.md) · [eval](pipeline/eval.md) · [export](pipeline/export.md) ·
> [infer](pipeline/infer.md) · [docker](docker.md)。

## 1. 数字与口径

### 官方 YOLO26n 的 40.1 是怎么来的？为什么从零训练到不了？

官方**不是从零训练**：所有官方 checkpoint 都是 Objects365v1 预训练 150 轮 + COCO 微调
（n 档 245 轮、batch 128）。官方文档原话 "No YOLO26 checkpoint was trained on COCO from
random weights"。本仓库同预算从零对照：YOLO26n 300 轮 37.82 / Flash-YOLO 35.14（E2E，
val2017）——**从零数字和官方数字放在一起比较不是 like for like**。

### 那“复现”到底复现了什么？

复现的是**可验证性**，不是刷官方数字：官方权重加载进本仓库管线 = 40.27 / 40.89
（E2E / NMS），与官方 40.1 / 40.9 的 ~0.1 差是度量实现噪声；同一权重换 ultralytics 跑本仓库
评估管线，数字逐位一致。训练内部同样对齐：同权重、同真实 batch 的双实现对照，总损失差 <1%、
梯度范数差 <2%；增强管线与官方逐像素 0 差异。

### Flash-YOLO 比 YOLO26n 低 2.68 点，为什么还值得用？

买的是算力：−31% GFLOPs（3.80 vs 5.48）、−23% 参数、b8 延迟 −11~−12%（TensorRT fp32，
RTX 4090 / 4060 Laptop）/ −26%（ORT-CPU b8）。精度-算力比是本仓库轻量化的核心指标；结构
取舍的结论见 [models/flash-yolo.md](models/flash-yolo.md)。

### 延迟表里 b1 在服务器上几乎打平、b8 优势明显，为什么？

batch 1 是 per-run 开销（启动/绑定/发射）主导的**地板**：4090 TRT fp32 1.71 vs 1.71 ms（打平），
4060 3.07 vs 3.63 ms（Flash 反而快 ~15%，笔记本调度/频率主导），两个架构在服务器上几乎分不
出来；batch 8 才稳定反映算力差：4090 0.281 vs 0.316 ms/img（−11%）、4060 1.25 vs 1.40
ms/img（−11%），与 GFLOP 比 1.44 同量级。**GFLOPs ≠ 延迟**——选型时看目标 batch 的口径。

### YOLOv3-tiny 官方 33.1 怎么对不上？

口径不同：官方 33.1 是 **test-dev @416 的 AP50**；本仓库同权重 @640 在 val2017 上用
pycocotools 得 35.90 AP50 / 17.16 mAP50-95。v3 两变体：`yolov3-tiny`（darknet 逐行同构）
只配官方 `.weights`；`yolov3-tiny-silu`（唯一差异 LeakyReLU→SiLU）是推荐训练配置的一部分
（配合 `--obj-iou-target`），两者键集/参数量/GFLOPs 完全一致。

### pose 的 val 为什么只有 2,346 张图？

COCO-pose 官方协议只评估**有有效关键点**的图（`num_keypoints > 0`）；
`person_keypoints_val2017.json` 里有 5,000 张图，但其中 2,654 张没有 pose GT。
`scripts/eval.py` 会自动过滤，报告数字时不要与检测的 5,000 张混用。

### pose 官方数字为什么不能和从零训练比？

官方 `yolo26n-pose` 是 **Objects365 检测预训练 + COCO-pose 400 轮微调**；从零训练
可以跑通，但不是 like-for-like。另：官方配方 `amp=true` 依赖 O365 初始化，从零建议
`--no-amp`。

### 训练中 pose mAP 和 `scripts/eval.py` 差很多，正常吗？

FastPoseMetrics 是快速近似（贪婪匹配 / 101 点插值 / 忽略 crowd），但已按 COCO keypoint 口径
修正：忽略 `num_keypoints==0` 标注、OKS 使用标注 `area`。同一 checkpoint 与正式 AP 差约
0.02，AR@20 基本一致；剩余差异来自 crowd/ignore 区域语义。训练日志只用于趋势和 best 选取，
跨 run / 跨实现比较只用 `scripts/eval.py`。

## 2. 训练

### 从零训练能开 AMP 吗？

不能：实测 fp16/bf16 都会在训练早期 NaN/爆炸。**从零训练默认 fp32**；**微调**可试 `--amp`。

### 第一个 epoch 的 val mAP 是 0，正常吗？

正常。cls 初始偏置让 sigmoid ≈ 1e-5，低于 conf 阈值，全部预测被过滤；训练一两步后恢复。
不要误判为管线故障。

### pose 训练日志里的 `val_rle` 为什么恒为 0？

`val_rle` 恒为 0 是设计行为：验证前向不计 RLE 项；训练侧的 `rle_loss` 仍正常更新。

### 训练要多久 / 显存怎么定 / batch 怎么调？

训练耗时取决于 GPU、数据管线与轮数，按目标环境用 `scripts/bench_io.py` 实测
（见 [pipeline/train.md](pipeline/train.md) §6）。物理 batch 由显存定即可——**梯度语义与
lr 由 nbs 保证（default 64），调 batch 不用动 lr**。

### 支持多卡 / 分布式训练吗？

支持（**仅训练**）。`--device 0,1` 自动为每卡起一个 worker，也支持 `torchrun`：

```bash
python scripts/train.py --data coco --device 0,1 --batch 32 --workers 8
```

`--batch` 是**全局** batch（必须整除），`--workers` 是每卡进程数。eval / export / infer
仍是单进程。多卡与单卡不逐位一致（无 SyncBatchNorm），同卡数重复运行可逐位复现。

### resume 能从单卡切到多卡、或换设备（CPU↔GPU）吗？

可以。`resume.pt` 记录 `world_size` / `device`，恢复时先落到 CPU 再复制到目标设备；
换卡数/设备会提示轨迹不再逐位一致。同卡数同设备续训仍逐位等价。换
`--batch`/`--nbs`/`--imgsz`/`--amp` 同样可续训但会告警。

### 换自己的数据集要改什么？

写一个描述符 yaml（[datasets/README.md](datasets/README.md) §1）+ `--data <描述符>`：
nc/names 随描述符走（模型头按 nc 建），索引 = 类别 id。自定义数据集检查清单见
[datasets/README.md](datasets/README.md) §5。

### yolov3-tiny 换输入尺寸，锚点要重算吗？

**官方权重：不用**——锚点是 darknet 固定像素值，多尺度训练下由网络自补偿；按 imgsz 线性
缩放是错的（实测 500 图 mAP50：640 原值 0.412 → 线性缩放 0.154）。**从零训练**换输入尺寸
才需要重聚类：`python scripts/compute_anchors.py --data <描述符>`。

## 3. 数据

### 扫描时告警 train2017 缺 1 张图？

官方 train2017 确实少 `000000391895.jpg`。加载器初始化时按文件存在性过滤并告警——预期行为。

### YOLO 标签文件名必须和图片同名吗？

支持两级回退：`<stem>.txt` 优先，缺失时回退 `str(int(stem)).txt`——兼容“按 COCO image id
命名”（`000000397133.jpg` → `397133.txt`）与 `1.txt ↔ 000001.jpg`。整目录零命中会**直接
报错**（全背景守卫）。见 [datasets/yolo.md](datasets/yolo.md)。

### crowd 标注怎么处理？为什么两种格式的 val 数字有差？

训练侧两边都剔除 crowd（统计逐项相等）；val 侧 COCO 保留（官方口径）、YOLO 无 crowd 概念，
val 差异只来自这一项（coco-tiny 全量 0.4080 vs 0.4024）。

### 图坏了会怎样？

扫描期不逐张解码（惰性）；`load_image` 读失败时计数并告警，训练不中断：train 侧返回空白图 +
**空标签**（不教假框），val 侧返回空白图 + **保留 GT**（诚实记为漏检）。

### `--limit` 抽样能跨格式比较吗？

不能直接比：COCO 按 json 顺序取前 N，YOLO 按文件名排序取前 N——是两批不同的图。跨格式比较
用全量或对齐集合。

## 4. 评估

### eval 和 infer 的 conf 默认值为什么不一样？

`eval.py` 用 0.001（对齐官方评估口径），`infer.py` 用 0.25（出图省事）。别拿推理图上的
数量去心算 mAP。

### E2E 和 NMS 差多少？该看哪个？

同权重一般差 0.6~1 点（官方 40.89 vs 40.27），两个都报。E2E 是默认路径（部署省一步、
NMS-free 是 YOLO26 的卖点）；NMS 走 o2m 头。见 [pipeline/eval.md](pipeline/eval.md) §4。

## 5. 导出与部署

### 导出的 onnx 旁边怎么多了一个 `.onnx.data`？

torch 2.9 起 `torch.onnx.export` 默认走 dynamo 导出器，其 `external_data=True` 默认会把
~10MB 模型也拆成两片——`.onnx` 单独不可用。`scripts/export.py` 固定 `dynamo=False`
（权重内嵌、单文件）；若未来 torch 移除旧路径，退路是 dynamo + `external_data=False`。

### TensorRT 引擎能换到别的机器用吗？

不能：引擎绑定**构建机的 GPU 架构 + TensorRT 版本**，换机器/换版本必须在目标机上重跑
`scripts/export.py --trt [--fp16]`。

### TRT 11 怎么开 fp16？为什么没有 FP16 flag？

TRT 11 移除了 `BuilderFlag.FP16`，也没有 precision-constraint 类标志——精度完全由网络
张量类型决定。`--fp16` 先 `model.half()` 另导半精度 onnx，再用 strongly-typed 网络构建
`<stem>.fp16.engine`（与 fp32 引擎共存）。收益在 batch：检测 b8 每图 −49~57%（4090 / 4060）；
精度无可测损失。

### onnxruntime 能跑半精度 onnx 吗？

CPU 的 ORT 没有 fp16 kernel，跑不了半精度图；半精度验收锚是 **torch half 模型**（实测
TRT fp16 引擎与它 raw 输出逐位一致）。ORT-CPU 的定位是 fp32 参考口径。

### pip 安装 TensorRT 卡住 / 装不上？

PyPI 上的 `tensorrt` 同名包是壳（会挂起）；按 Dockerfile 的路线直连 NVIDIA 索引装，或直接
用本仓库 Docker 镜像（已含 TRT 11.4）。见 [docker.md](docker.md)。

## 6. 复现与环境

### 同一个 seed 两跑，结果不一样？

按四件事排查：① `deterministic` 是否开（默认开）；② `workers`/`batch` 是否一致（它们是
复现条件的一部分）；③ 是否有人新增了裸的全局 `np.random`/`random` 调用；④ 是否 resume 场景。
**跨机器到不了逐位**（kernel 不同），只承诺统计一致。见
[pipeline/train.md](pipeline/train.md) §5。

### 拿到一个权重文件，怎么知道是哪版代码、什么设置训的？

权重 metadata 自带 `arch/scale/nc/imgsz/names` + 出处 `version/repo/commit/describe`。
读取三级回退：**CLI 显式 > metadata > 文件名/默认**；缺键 = 未知（不猜）。`version` 是发布线、
`commit` 是精确坐标，两者不要求一致——`describe` 把落差写明。

### 容器里 `nproc` 显示的核数偏大，该开多少 workers？

容器常见的 CPU 谎报：affinity mask 报宿主核数，实际受 cgroup 配额限制。workers 按
**实测**选：`scripts/bench_io.py`（两阶段：loader-only 网格 → 端到端）——
**加载器快 ≠ 训练快**，且 workers 不是越多越好。

### WSL2 下耗时/速度偶尔显示异常（负值）？

WSL2 时钟同步会让 `time.time()` 跳变。本项目统一用 `time.monotonic()`；改计时相关代码时
别用 `time.time()`。

### Docker 构建失败怎么办？

按 [docker.md](docker.md) §5 的检查点顺序排查：base tag、Python 版本、TensorRT 索引安装、
apt 包名。构建成功后建议顺序：`pytest tests/` → 单图推理 → `export.py --trt` → 短跑训练。

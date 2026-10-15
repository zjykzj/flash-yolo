# 常见问题（FAQ）

> 用途：把开发中踩过的"只有真做过才知道"的问题集中回答——数字口径、训练稳定性、数据口径、
> 复现与环境。跨页面的"为什么/怎么办"在这里给答案，分页细节仍留在各手册页：
> `dataset.md`（数据）· `train.md`（训练）· `val.md`（评估）· `export.md`（导出部署）·
> `infer.md`（推理）· `docker.md`（容器）。各模型的正式数字见 `yolo26-runs.md` /
> `flash-yolo-spec.md` / `yolov3-tiny-runs.md`。

## 1. 数字与口径

### 官方 YOLO26n 的 40.1 是怎么来的？为什么从零训练到不了？

官方 **不是从零训练**：所有官方 checkpoint 都是 Objects365v1 预训练 150 轮 + COCO 微调
（n 档 245 轮、batch 128），官方文档原话 "No YOLO26 checkpoint was trained on COCO from random
weights"。本仓库从零 100 轮、同预算对照：YOLO26n 36.22 / flash-yolo 33.77（E2E，val2017）——
**把从零数字和官方数字放在一起比较不是 like for like**。

### 那"复现"到底复现了什么？

复现的是**可验证性**，不是刷官方数字：官方权重加载进本仓库管线 = 40.27 / 40.89（E2E / NMS），
与官方 40.1 / 40.9 的 ~0.1 差是度量实现噪声——同一权重换 ultralytics 跑本仓库同一条评估管线，
数字逐位一致。训练内部同样对齐：同权重、同真实 batch 的双实现对照，总损失差 <1%、梯度范数差
<2%；增强管线与官方逐像素 0 差异。

### 从零训练 100 轮就停，300 轮会不会更高？

100 轮是本仓库从零 run 的固定预算（与 YOLO26n 对照同配方、同预算、可逐轮对照）。两侧 run 末
3 轮 in-training mAP 都已平（变化 ≤0.0004）→ **flash-yolo 与 YOLO26n 的 2.45 点差距是架构
取舍，不是预算不足**。再往下加的收益没有证据支持，先把这条预算钉死。

### flash-yolo 比 YOLO26n 低 2.45 点，为什么还值得用？

买的是算力：−31% GFLOPs（3.80 vs 5.48）、−23% 参数、batch-8 延迟 −14%（TensorRT fp32）/
−26%（ORT-CPU b8）。精度-算力比是本仓库轻量化的核心指标，flash-yolo 的取舍是单变量筛选出来的
（见 `flash-yolo-spec.md`）。

### 延迟表里 b1 几乎打平、b8 优势明显，为什么？

batch 1 是 per-run 开销（启动/绑定/发射）主导的**地板**：TRT fp32 1.32 vs 1.38 ms（−4%），
两个架构几乎分不出来；batch 8 才反映算力差：0.213 vs 0.248 ms/img（−14%，与 GFLOP 比 1.44
同量级）。**GFLOPs ≠ 延迟**——选型时看目标 batch 的口径，别拿 b1 拍板。

### 数值对齐为什么要抠到 0.1 / 像素级？

对齐不是炫技：差 0.1 看着小，但它可能来自两种完全不同的东西——度量实现噪声（可接受，见
上面 40.27 一条）或实现错误导致的系统性偏离（不可接受）。两个真实教训：BN eps 用 PyTorch
默认 1e-5（官方是 0.001）会让第 0 层起的激活全部偏离；o2m 路径把 xywh 解码成 xyxy 会得到
~632 量级的输出误差。对齐测试（`tests/test_weight_alignment.py` 等）就是给这类"低 0.1 假象"
兜底的闸门。

### yolov3-tiny 官方 33.1 怎么对不上？

口径不同：官方 33.1 是 **test-dev @416 的 AP50**；本仓库同权重 @640 在 val2017 上用
pycocotools 得 35.90 AP50 / 17.16 mAP50-95——同一权重 640 比 416 高（500 图 FastMetrics 对照
0.412 vs 0.384）。另外两个 v3 变体：`yolov3-tiny`（darknet 逐行同构）只配官方 `.weights`；
`yolov3-tiny-silu`（唯一差异 LeakyReLU→SiLU）是推荐训练配置的一部分（配合
`--obj-iou-target`），两者键集/参数量/GFLOPs 完全一致。

## 2. 训练

### 从零训练能开 AMP 吗？

不能，实测必炸：fp16 在第 1 轮内 NaN、bf16 首个优化步就炸（机制 = BN eps=0.001 对 mosaic
灰边等近常数区域的放大级联 + fp16 上限 65504）。**从零训练默认 fp32**；**微调**（`--weights`
初始化，激活远离危险区间）可以试 `--amp`。`train.md` §7。

### 训练 loss 正常，但权重爆掉 / 第几轮就 NaN？

历史坑：梯度裁剪 `max_norm=10` 是官方训练循环的一部分，必须每步无条件执行。缺了它，o2o 分类
头会进入"BN 掩蔽式复利增长"——loss 全程正常（BN 把权重尺度藏起来），非 bias 权重却逐 step
复利到 1e10+，然后 NaN。排查这类问题**盯非 bias 权重的 |W|max，别只看 loss**。

### 第一个 epoch 的 val mAP 是 0，正常吗？

正常。cls 初始偏置让 sigmoid ≈ 1e-5，低于 conf 阈值，全部预测被过滤；训练一两步后恢复。
不要误判为管线故障。`train.md` §7。

### 训练卡死：进程活着、GPU 0%、没有任何报错？

已知场景，框架内已修复：OpenCV 线程池非 fork 安全——父进程跑过 cv2 之后，worker 里再调
`cv2.setNumThreads()` 会永久阻塞（典型表现：epoch 1 正常，epoch 2 起整体卡死）。本仓库的契约
是**线程数在 `data/loader.py` 导入时由父进程设定、worker 只继承**；改数据管线时别破坏它。

### 训练要多久 / 显存怎么定 / batch 怎么调？

参考量级：flash-yolo 100 轮 batch 64 @640，单张 RTX 5090 ≈15.5 h（YOLO26n 同预算 ~14 h）。
物理 batch 由显存定即可——**梯度语义与 lr 由 nbs 保证（default 64），调 batch 不用动 lr**。
吞吐别拍脑袋，用 `scripts/bench_io.py` 实测（`train.md` §6）。

### 支持多卡 / 分布式训练吗？

暂不支持，当前是单卡训练。

### 换自己的数据集要改什么？

写一个描述符 yaml（`dataset.md` §1）+ `--data <描述符>`：nc/names 随描述符走（模型头按 nc
建），索引 = 类别 id。自定义数据集的检查清单见 `dataset.md` §5；分钟级迭代用 `coco-tiny`
路线（`dataset.md` §6）。

### yolov3-tiny 换输入尺寸，锚点要重算吗？

**官方权重：不用**——锚点是 darknet 固定像素值，多尺度训练下由网络自补偿；按 imgsz 线性缩放
是错的（实测 500 图 mAP50：640 原值 0.412 → 线性缩放 0.154）。**从零训练**换输入尺寸才需要
重聚类：`python scripts/compute_anchors.py --data <描述符>`。

## 3. 数据

### 扫描时告警 train2017 缺 1 张图？

官方 train2017 确实少 `000000391895.jpg`。加载器初始化时按文件存在性过滤并告警——预期行为，
不是你的数据坏了。

### YOLO 标签文件名必须和图片同名吗？

支持两级回退：`<stem>.txt` 优先，缺失时回退 `str(int(stem)).txt`——同时兼容"按 COCO image id
命名"（`000000397133.jpg` → `397133.txt`）与 `1.txt ↔ 000001.jpg`。整目录零命中会**直接报错**
（全背景守卫）——否则会安静地训练 100% 背景。`dataset.md` §4。

### crowd 标注怎么处理？为什么两种格式的 val 数字有差？

训练侧两边都剔除 crowd（统计逐项相等）；val 侧 COCO 保留（官方口径）、YOLO 无 crowd 概念，
val 数字差异只来自这一项（coco-tiny 全量实测：0.4080 vs 0.4024）。

### 图坏了会怎样？

扫描期不逐张解码（惰性）；`load_image` 读失败时计数并告警，训练不中断：train 侧返回空白图 +
**空标签**（不教假框），val 侧返回空白图 + **保留 GT**（诚实记为漏检）。

### `--limit` 抽样能跨格式比较吗？

不能直接比：COCO 按 json 顺序取前 N，YOLO 按文件名排序取前 N——是两批不同的图。跨格式比较
要用全量或对齐集合。

## 4. 评估

### eval 和 infer 的 conf 默认值为什么不一样？

`eval.py` 用 0.001（对齐官方评估口径），`infer.py` 用 0.25（出图省事）。别拿推理图上的数量
去心算 mAP。

### 早期/未训练模型评估特别慢（NMS 秒级每图）？

历史坑：conf 0.001 下未训练模型 2535 锚 × 80 类几乎全过筛，逐类贪心 NMS 实测 7.2 s/图。现在
NMS 是**类感知 + 早停**（凑满 max_det 即返回），~12 ms/图，结果与"逐类全量 NMS 再全局取
top"规范化等价。别改回逐类全量做。

### E2E 和 NMS 差多少？该看哪个？

同权重一般差 0.6~1 点（官方 40.89 vs 40.27），两个都报。E2E 是默认路径（部署省一步、NMS-free
是 YOLO26 的卖点）；NMS 走 o2m 头。`val.md` §4。

## 5. 导出与部署

### 导出的 onnx 旁边怎么多了一个 `.onnx.data`？

torch 2.9 起 `torch.onnx.export` 默认走 dynamo 导出器，其 `external_data=True` 默认会把 ~10MB
的模型也拆成两片——`.onnx` 单独不可用。`scripts/export.py` 固定 `dynamo=False`（权重内嵌、单
文件）；若未来 torch 移除旧路径，退路是 dynamo + `external_data=False`（同样单文件）。
`export.md` §1。

### TensorRT 引擎能换到别的机器用吗？

不能：引擎绑定**构建机的 GPU 架构 + TensorRT 版本**，换机器/换版本必须在目标机上重跑
`scripts/export.py --trt [--fp16]`（几十秒到几分钟）。仓库不下发也不能下发预构建引擎。

### TRT 11 怎么开 fp16？为什么没有 FP16 flag？

TRT 11 移除了 `BuilderFlag.FP16`，也没有 precision-constraint 类标志——精度完全由网络张量
类型决定。所以 `--fp16` 的做法是：先 `model.half()` 另导一份半精度 onnx，再用
**strongly-typed** 网络构建 `<stem>.fp16.engine`（与 fp32 引擎共存）。把 fp32 图交给 fp16
构建会直接报错，不静默产出"看着像 fp16"的 fp32 引擎。收益在 batch：b8 每图 −41~44%；精度无
可测损失（E2E 输出经 head 里的类型提升仍是 fp32）。`export.md` §2/§4。

### onnxruntime 能跑半精度 onnx 吗？

CPU 的 ORT 没有 fp16 kernel，跑不了半精度图；半精度的验收锚是 **torch half 模型**（实测
TRT fp16 引擎与它 raw 输出逐位一致）。ORT-CPU 的定位是 fp32 参考口径。

### pip 安装 TensorRT 卡住 / 装不上？

PyPI 上的 `tensorrt` 同名包是壳（会挂起）；按 Dockerfile 的路线直连 NVIDIA 索引装
（`tensorrt_cu13_libs` / `_bindings` / `tensorrt_cu13`），或直接用本仓库 Docker 镜像（已含
TRT 11.4）。`docker.md`。

## 6. 复现与环境

### 同一个 seed 两跑，结果不一样？

按四件事排查：① `deterministic` 是否开（默认开；`--no-deterministic` 会引入末位差）；
② `workers`/`batch` 是否一致（它们是复现条件的一部分，不是纯性能参数）；③ 是否有人新增了裸的
全局 `np.random`/`random` 调用；④ 是否 resume 场景（逐位等价是保证的）。**跨机器到不了逐位**
（kernel 不同），跨机器只承诺统计一致。`train.md` §5。

### 拿到一个权重文件，怎么知道是哪版代码、什么设置训的？

权重 metadata 自带 `arch/scale/nc/imgsz/names` + 出处 `version/repo/commit/describe`。读取
三级回退：**CLI 显式 > metadata > 文件名/默认**；缺键 = 未知（不猜）。`version` 是发布线、
`commit` 是精确坐标，两者不要求一致——`describe`（如 `v0.5.0-17-g984d198-dirty`）把落差写明。

### 容器里 `nproc` 显示 208 核，该开多少 workers？

容器常见的 CPU 谎报：affinity mask 报宿主核数，实际是 cgroup 配额（本机实测 25 核）。workers
按**实测**选：`scripts/bench_io.py`（两阶段：loader-only 网格 → 端到端）——**加载器快 ≠ 训练
快**，且 workers 不是越多越好（16→20 反而更慢：worker 占满核、主进程被饿）。`train.md` §6。

### WSL2 下耗时/速度偶尔显示异常（负值）？

历史坑：WSL2 时钟同步会让 `time.time()` 跳变（产生过负速度和负总时长）。本项目统一用
`time.monotonic()`；改计时相关代码时别用 `time.time()`。

### Docker 构建失败怎么办？

按 `docker.md` §5 的检查点顺序排查：base tag 是否存在、base 的 Python 版本、TensorRT 直连索引
安装、apt 包名随发行版变化。构建成功后建议顺序：`pytest tests/` → 单图推理 →
`export.py --trt` → 短跑一次训练。

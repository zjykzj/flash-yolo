# 训练（Train）

> 入口：`python scripts/train.py --data <name|.yaml> [选项]`。数据见 `dataset.md`，
> 评估口径见 `val.md`。

## 1. 四条路径

```bash
# 从零训练（默认；内置 default 配方 100 轮）
python scripts/train.py --data coco --name my-run --batch 64 --workers 16

# 微调（用已有权重初始化；可配 --recipe 换微调配方）
python scripts/train.py --data coco --weights weights/yolo26n.safetensors --name ft

# 续训（恢复完整训练状态：优化器/EMA/轮次/调度）
python scripts/train.py --data coco --resume runs/train/my-run/resume.pt

# 换架构（yolov3-tiny 结构固定，不要传 --scale）
python scripts/train.py --data coco --model yolov3-tiny --name v3 --imgsz 416
# 算子替换版（LeakyReLU→SiLU；键集/参数量/GFLOPs 与原版一致）
python scripts/train.py --data coco --model yolov3-tiny-silu --name v3s --imgsz 416

# 实例分割（COCO json 多边形或 YOLO-seg txt 多边形；bbox 行按矩形回退）
python scripts/train.py --data coco --model yolo26-seg --name seg --mask-ratio 4
# flash-yolo 的分割变体（单档 flash，不要传 --scale；结构见 flash-yolo-seg-spec.md）
python scripts/train.py --data coco --model flash-yolo-seg --name flash-seg --mask-ratio 4

# 多卡 DDP（`--batch` 是**全局** batch：2 卡时每卡 32；仅训练链路，见 §7）
python scripts/train.py --data coco --name ddp --device 0,1 --batch 64 --workers 8
# 等价 torchrun 入口（自动读 RANK/LOCAL_RANK/WORLD_SIZE，设备按 LOCAL_RANK 选）
torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco --name ddp --batch 64 --workers 8
```

## 2. 配方（recipe）

优先级：`config/train.yaml` 基础值 → 配方文件覆盖 → CLI 覆盖。

| 配方 | 阶段 | 前置条件 | 轮数（n/s/m/l/x） |
|---|---|---|---|
| *（不传）* | 从零 / 自有数据微调 | — | 100 |
| `yolo26-coco-ft` | 官方 COCO 阶段（微调） | **需 O365 预训练权重初始化** | 245/70/80/60/40 |
| `yolo26-seg-coco-ft` | 官方 seg COCO 阶段（微调） | 需 `yolo26*-objv1-seg`（365 类）初始化 + COCO json / YOLO-seg 描述符；n/s/m/l/x 全部按官方 seg ckpt 的 `train_args` 核对 | 245/50/80/60/40 |
| `yolo26-o365-pt` | 官方 Objects365 预训练（检测） | 需 Objects365 数据集（未在本仓库验证） | 150 |
| `yolo26-seg-o365-pt` | 官方 Objects365 预训练（分割） | 需 Objects365v1 分割数据集（未在本仓库验证）；数值取自五个 `*-objv1-seg` ckpt 的 `train_args` | 150 |

> `yolo26-seg-coco-ft` 的数值取自五个官方 seg ckpt 的 `train_args`：m/l/x 与官方检测表一致；
> s 来自另一提交的早期基线（epochs 50 / lr0 0.00125），n 的 mixup/degrees/shear/bgr 与检测表不同——
> 均按 seg ckpt 保留。配方显式写入 `mask_ratio: 1`、`overlap_mask: true`、`copy_paste_mode: flip`，
> semseg 辅助损失始终启用；其余实验分支参数见配方头部（cls_w/o2m 已写入配方，erasing/auto_augment
> 为分类任务专用、检测/分割管线不消费）。从零训练仍用 default。

```bash
python scripts/train.py --data coco --recipe yolo26-coco-ft --weights yolo26n-objv1-150.safetensors

# 分割官方 COCO 阶段（n 档；mask_ratio=1 / copy_paste_mode=flip / semseg）
python scripts/convert_weights.py --src weights/yolo26n-objv1-seg.pt \
    --dst weights/yolo26n-objv1-seg.safetensors
python scripts/train.py --data coco --recipe yolo26-seg-coco-ft \
    --weights weights/yolo26n-objv1-seg.safetensors
```

> 分割初始化是 **365 类**的 O365 权重，COCO 训练建 80 类头：`--weights` 走
> `load_weights_finetune` 的 key+shape 交集加载——backbone/neck/proto/cv2 逐位加载，
> cv3/one2one_cv3/semseg 等类别相关头按 80 类重建（n 档实测 740/844 张量加载、104 个重初始化）。
> 实测（RTX 4080 32GB，mask_ratio=1，640，batch 1，AMP）训练约 **5.2 img/s**、峰值 **1.41G**；
> 全量 COCO 245 轮约 **65 天/单卡**。DDP 已实现（§7），可直接多卡扩；吞吐上限通常在数据管线
> 而不是 GPU（本机 2×5090 检测实测约 1.2× 单卡——loader 先饱和；batch/模型越大比例越高）。
>
> **单卡闭环验收（coco-tiny：train 2000 / val 1000）**：官方 n 档配方 + O365 初始化，
> 1 epoch / batch 4 / AMP / 640 —— 训练 7m20s、峰值 5.31G；pycocotools Box mAP50-95
> 0.0017 / Mask 0.0015（person 0.073 / 0.062）；导出双输出 ONNX 后 pt/onnx 掩码逐像素一致。
> 1 epoch 只用于验收管线，不代表精度（类别头重建后仅 ~15 个优化步）。

> v3 没有专用配方：**推荐配置 = default + 两个开关**（`--model yolov3-tiny-silu --obj-iou-target`），
> 依据与数字见 `yolov3-tiny-runs.md` §4.5；旧的 `yolov3-tiny-darknet` 配方已定案退役（§4.4）。

> **从零训练只有 `default` 一档。** `yolo26-coco-ft` 的 lr0 是基线的一半、衰减更陡，是
> 进化搜索针对"已收敛起点"调的——从零跑它是把模型按在慢速档上（实测同轮次相对低 ~43%）。
> 同一个配方下的**轮数**是硬预算：100 轮远未收敛（见 `yolo26-runs.md`）。

## 3. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | —（必填） | 描述符（见 `dataset.md`） |
| `--name` | 空 → `runs/train/train` | run 目录：`--name foo` → `runs/train/train-foo`；目录已存在自动加序号 |
| `--recipe` | 内置 default | 配方名（`config/recipes/<name>.yaml`）或 `.yaml` 路径 |
| `--model` | `yolo26` | `yolo26` / `yolo26-seg` / `flash-yolo` / `flash-yolo-seg` / `yolov3-tiny` / `yolov3-tiny-silu`（算子替换版） |
| `--scale` | `n` | yolo26 系（含 yolo26-seg）档位 n/s/m/l/x；其他架构固定（flash-yolo / flash-yolo-seg 单档 flash） |
| `--mask-ratio` | 4 | 分割 GT 掩码下采样倍率（`imgsz/mask_ratio`）；4 = Proto26 分辨率，1 = 官方 seg ckpt 记录的全分辨率 |
| `--overlap-mask` / `--no-overlap-mask` | true | 分割 GT：每图实例索引图（官方默认）vs 每实例二值掩码 |
| `--cfg` | — | 模型 yaml 覆盖（自定义结构 / 筛选变体） |
| `--epochs` | 100 | 总轮数（配方的调度也按它算） |
| `--batch` | 16 | **物理** batch（**全局**口径）；DDP 下每卡 = batch/world_size（必须整除）。机器相关，用 `bench_io.py` 实测后给 |
| `--nbs` | 64 | 名义 batch；`accum = max(round(nbs/batch),1)`，梯度语义与物理 batch 无关 |
| `--imgsz` | 640 | 训练输入（v3 常用 416）。batch 需整除 nbs（否则 wd 因子 ≠ 1.0） |
| `--workers` | 8 | 加载进程数；机器相关 |
| `--seed` | 0 | 全局种子（数据/增强/初始化） |
| `--deterministic` / `--no-deterministic` | true | 逐位可复现的 kernel 开关（关 cuDNN autotune + 确定性算法）；代价实测在噪声内，见 §5 |
| `--weights` | — | 初始化权重（微调） |
| `--resume` | — | `resume.pt` 或 run 目录（恢复优化器/EMA/轮次） |
| `--device` | auto | `cuda:0` / `cpu` / `0,1`（DDP，仅训练；见 §7） |
| `--amp` / `--no-amp` | false | fp16+GradScaler；**从零训练会 NaN，仅微调用** |
| `--optimizer` | `musgd` | `musgd` = Muon+SGD 混合（官方 yolo26 口径）；`sgd` = 全参数走 SGD 路径（darknet 口径复刻：配 `--momentum 0.9 --no-nesterov --lr0 0.001` + 台阶 LR） |
| `--nesterov` / `--no-nesterov` | true | SGD 分支的 Nesterov 修正；darknet 的更新是 heavy-ball，复刻时用 `--no-nesterov` |
| `--obj-iou-target` | false | v3 系 IoU-aware objectness（参考 zjykzj/YOLOv3）：正样本 obj 目标 = IoU(预测框, 责任 GT)、权重 ×5；false = darknet 硬目标 0/1 |
| `--close-mosaic` | 10 | 最后 N 轮关 mosaic/mixup/cutmix/copy_paste（自动 `min(N, epochs//5)`） |
| `--multi-scale` | 0.0 | 输入尺寸抖动 ±r（ultralytics 口径，按 32 量化） |
| `--copy-paste-mode` | `off` | `off` / `box`（检测矩形贴块近似）/ `flip`（官方分割 CopyPaste） |
| `--perspective` | 0.0 | 透视畸变幅度（官方 P 矩阵；0 = 关闭且不消耗 RNG） |
| `--cutmix` | 0.0 | CutMix 概率（官方语义：候选切块避开已有目标，实例裁剪后拼接） |
| `--cls-w` | 1.0 | o2o 分类项额外权重（官方 ckpt 记录参数；1.0 = released 代码行为） |
| `--freeze` | 0 | 冻结前 N 层（model.0..N-1，冻结层 BN 保持 eval） |
| `--single-cls` | false | 单类训练：标签全部映射到 0，模型 nc=1（评估配 `--single-cls`） |
| `--plots` / `--no-plots` | true | 训练结束写 results/PR/混淆矩阵（需 matplotlib；缺失自动跳过） |
| `--channels-last` | true | NHWC 训练（cuDNN 反向快 ~33%）；推理/导出不受影响 |
| `--stop-after` | 0 | 只跑前 N 轮（lr/close_mosaic 仍按 `epochs` 调度 → 可与全长 run 逐轮对齐） |
| `--val-epochs` | 1 | 0 = 全程不验证（冒烟）；`--val-limit` 限制验证图数 |
| `--limit` | 0 | 训练子集（前 N 张） |

其余全部超参（优化器 / 损失 / 增强 / 产物周期）都有同名 CLI 覆盖项，`--help` 列出全量；
默认值即 `config/train.yaml`（= ultralytics `default.yaml`，逐项一致），文件内每项都有注释。

## 4. 运行产物（`runs/train/train-<name>/`）

| 产物 | 内容 |
|---|---|
| `weights/best.safetensors` | 按训练中指标选的 EMA 权重（**正式数字请另行 eval**）；分割的 fitness = box mAP@[.5:.95] + mask mAP@[.5:.95] |
| `weights/last.safetensors` · `best_raw.safetensors` | 最后一轮 / 非 EMA 最佳 |
| `weights/epochNNN.safetensors` | 每 `save_period`（默认 20）轮一存，留最近 `keep_periodic` 个——**中途分叉实验用** |
| `results.csv` | 逐轮训练损失 + 训练中指标 + **val 损失**（val_box/val_cls/val_l1/val_o2m/val_o2o）；分割另有 `seg`/`val_seg`、`sem`/`val_sem` 与 `P(M)/R(M)/mAP50(M)/mAP@[.5:.95](M)`（`val_sem` 在验证模式恒 0，官方同口径） |
| `diag/train_diag.csv` | 每 `diag_interval` 步：**裁剪前**梯度范数、分组 lr、损失分解 |
| `samples/*.png` | 增强抽样网格（首轮 / 关 mosaic 后 / 末轮各一张，肉眼校验增强口径） |
| `results.png` · `PR_curve.png` · `confusion_matrix*.png` | 训练结束报告（`--no-plots` 关闭；matplotlib 缺失时自动跳过）：损失/mAP 曲线、IoU=0.5 逐类 PR、conf 0.25 / IoU 0.45 的混淆矩阵 |
| `meta.json` | 环境（git commit / torch / CUDA / GPU）+ 解析后的完整配置快照 |
| `run.log` | 该 run 的控制台 + 文件日志 |
| `resume.pt` | 续训状态 |

**权重 metadata**：每个 checkpoint 的 safetensors header 带一份可选 JSON（`arch` / `scale` /
`nc` / `imgsz` / `names`，含锚模型另有 `anchors`），所以 eval / infer / export 可以直接从文件
读模型配置（CLI 显式覆盖 > metadata > 文件名/默认）。另带出处 `version` / `repo` / `commit` /
`describe`（哪版代码产的这份权重）：`version` 是**声明版本（发布线）**、`commit` 是**精确坐标**，
两者不要求对应（版本号只在发布时 bump，两次发布之间的提交都顶着旧版本号）；`describe` 是
`git describe` 的原始输出（如 `v0.5.0-17-g984d198-dirty` = "v0.5.0 之后 17 个提交、落在 984d198、
工作区脏"），把这段落差显式写出来。`commit` / `describe` 是 best-effort——读不到就不写该键
（如镜像内 `.git` 被 `.dockerignore` 排除）。只写 header，不碰张量。

## 5. 复现性口径

三层，按"要求多严"递增——**只有训练有随机性**（权重初始化 + 数据增强），评估/导出/推理全程无随机源，
它们的复现靠的是数值验收阈值（见 `val.md` / `export.md`），不是这套。

| 层 | 含义 | 今天的保证 |
|---|---|---|
| **稳定** | 任何配置下不发散不 NaN | fp32 默认（从零训练不开 AMP）+ 梯度裁剪 max_norm=10；历史三次发散路径都已定位并默认关闭 |
| **可复现（逐位）** | 同代码 commit + 同 seed + 同硬件/驱动 + 同 workers/batch → **权重逐位相同** | 默认达成（`deterministic: true`）；`FLASH_YOLO_REPRO=1 pytest tests/test_train_repro.py` 是闸门 |
| **可复现（统计）** | 跨硬件/驱动/GPU 型号 → 曲线与指标在噪声内一致 | 未量化（换机复现本就到不了逐位：kernel 不同）；A/B 比较仍用同机 + `--stop-after` 逐轮对齐 |

**逐位复现需要同时成立的四件事**（缺一即退化到统计口径）：

1. `deterministic: true`（默认）——kernel 侧：关 cuDNN autotune、确定性算法（`warn_only`，遇无确定性
   实现的算子只告警不炸）、固定 CUBLAS 工作区。**实测代价在噪声内**：yolo26n@640 b16 channels_last
   纯算力口径 ~3-5%（57.4 → 60 ms/step），端到端（真实加载器）不可辨——管线本来就是数据瓶颈。
   `--no-deterministic` = 纯速度优先，同 seed 两跑有末位差异。
2. **`workers` 与 `batch` 是复现条件的一部分**（机器相关值，故不写进默认配置）：增强流的种子按
   (seed, epoch, worker_id) 派生，改 worker 数会重排"哪张图落到哪个 worker"。换机器时按
   `scripts/bench_io.py` 重选，然后把它当实验参数固定下来。
3. **RNG 流域每 epoch 重播**（不依赖历史消费量）：DataLoader generator 与主进程增强 RNG 的种子
   都是 `seed + epoch × world_size + rank`（单卡即 `seed + epoch`，逐位不变）；worker 内由
   `worker_init_fn` 播种、主进程（`workers=0`）由 `seed_main_rng(同一派生种子)` 播种；stdlib
   `random`（multi_scale 抽尺寸）也随之播种。DDP 的抽样顺序由 `DistributedSampler(seed=cfg.seed)`
   按 `set_epoch(epoch)` 决定（各 rank 是同一 permuted 索引表的跨步切片，无重叠）。
   **新增随机性时不要裸调全局 `np.random`/`random`**——会同时打破第 4 条。
4. **resume 逐位等价**：`resume.pt` 存了 torch/cuda/numpy/random 四套 RNG 状态 + `ema.steps`，且抽样流
   按 epoch 重建——实测"1 轮后续训"与"一次跑完"逐位相同。

**配合权重 metadata 用**：交付物里带 `commit` / `describe`，正是为了让人拿到文件就能定位"哪版代码 +
什么设置"。完整复现命令 = 描述符 + `--seed` + `--workers`/`--batch` + 权重 metadata 里的 commit。

## 6. 先把 I/O 调对（`scripts/bench_io.py`）

`--batch` / `--workers` 是机器相关值，仓库**刻意不写进默认配置**。用基准脚本决定：

```bash
python scripts/bench_io.py --data coco            # 两阶段：loader-only 网格 -> 端到端只跑前 K
python scripts/bench_io.py --data coco --loader-only
```

- **加载器快 ≠ 训练快**：只有端到端数字能决策（实测本机 loader-only 496 img/s 而端到端 ~230）。
- workers 不是越多越好（实测 16→20 反而更慢：worker 占满核，主进程被饿）。
- worker 里的 cv2/torch 线程数由框架统一管理（`data/loader.py` 导入时设定）——
  **不要在 worker 里调 `cv2.setNumThreads`**（OpenCV 线程池非 fork 安全，会整体卡死）。

## 7. 多卡训练（DDP）

只覆盖**训练**：`scripts/train.py` 一条路径支持检测与分割（yolo26 / yolo26-seg / flash-yolo /
flash-yolo-seg / yolov3-tiny），eval / export / infer 仍是单进程，CLI 不变。

### 7.1 启动与 batch 语义

```bash
python scripts/train.py --data coco --device 0,1 --batch 64 --workers 8 --name ddp
# 或 torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco --batch 64 ...
```

- `--device 0,1`：父进程为每张卡起一个 worker（设置 `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*`），
  任一 worker 非 0 退出即终止其余。`torchrun` 已设好这些环境变量时不再 spawn。
- **`--batch` 是全局 batch**（ultralytics 口径）：每卡 = `batch // world_size`，必须整除。
  `accumulate = max(round(nbs / batch), 1)` 仍按全局 batch 计算，`nbs` 的梯度语义不变。
  `--workers` 是**每卡**的加载进程数（2 卡 × 8 = 16 个 worker，按机器核数分配）。
- 单卡（world_size=1）路径逐位不变：所有 rank 分支为真、barrier 为 no-op。

### 7.2 梯度口径（为什么要乘 world_size）

本工程的损失是 **Σ_img 口径**（`total = weight × Σ_img L_img`，见 `train/loss/yolo26_detect.py`
文件头）。DDP 在 backward 时对梯度做 **按 rank 平均**：每 rank 的局部 Σ 平均后是全局 Σ 的 1/W。
所以 backward 前把损失乘 `world_size`（`train/distributed.py::ddp_scale_loss`，world=1 原样返回），
"每 rank 局部 Σ → 平均 → ×W" 才等于单卡同一 global batch 的梯度。实测（2×5090，同数据同 global
batch，aug 关闭）：2 卡/1 卡裁剪前 gnorm 中位数比 = **1.03**；把 ×W 临时去掉后同一对比 = **0.50**
（回归信号：2 卡 gnorm 约为单卡一半）。

### 7.3 哪些东西按 rank 分、哪些只 rank 0

- **训练数据**：`DistributedSampler(shuffle=True)` 每 epoch `set_epoch(epoch)`，各 rank 拿互不
  重叠的索引切片（不整除时官方口径补样）。每卡一个 DataLoader generator（种子含 rank）。
- **验证只 rank 0 且全量**：不按 rank 分片，指标口径与单卡完全一致（检测数秒、分割百秒级）。
  其他 rank 在每轮末的 barrier 等待；barrier 在验证/保存之后、且每个 epoch 无条件执行。
- **产物只 rank 0**：checkpoint / results.csv / diag / samples / plots / meta.json / run.log。
  非 0 rank 写 `run.rankN.log`（文件仍 INFO），控制台抬到 ERROR。
- **`self.model` / `self.head` 始终是未包裹的 raw module**：DDP 只放在 `self.ddp_model`，
  EMA（deepcopy raw）、state_dict 保存、resume 都不带 `module.` 前缀。
- **梯度累积**：非累积边界的 micro-batch 把 forward+backward 整体包进 `no_sync()`（不发
  all-reduce），边界步同步；单卡是 `nullcontext`，代码顺序与历史一致。

### 7.4 与单卡的数值差异

- **不转换 SyncBatchNorm**：BN 前向统计按卡独立（官方 DDP 口径）。DDP 默认的 buffer 同步保留
  （每 forward 从 rank0 广播 running stats），因此 running stats 以 rank0 为准，但每卡的
  **批次统计**仍不同 → 多卡结果与单卡**不逐位一致**（同 world_size/rank 数的重复运行可逐位复现）。
- `find_unused_parameters=True`：覆盖 freeze 与损失内的条件分支（某些 step 参数不参与前向）。
  首个 iteration 会看到 torch 的 `find_unused_parameters=True ... did not find any unused
  parameters` 告警（本模型全参数参与前向时 torch 自己提示），属预期。
- **AMP**：每个 rank 独立 GradScaler；同步 DDP 的梯度 all-reduce 会让任一 rank 的 inf/nan
  传播到所有 rank，`found_inf` 一致、各 rank 一起 skip step，代码不需要额外分支。
  从零训练仍默认 fp32（AMP 的 NaN 机制与单卡相同，DDP 不解决）。
- **resume**：`resume.pt` 由 rank0 写，各 rank 读同一文件（模型/EMA/优化器/RNG 状态一致），
  DDP 包装在恢复之后，可直接 `--resume` + `--device 0,1` 续训。

### 7.5 吞吐

DDP 只并行训练步；**数据管线是常见瓶颈**（本机 2×5090 / yolo26n@640 / batch 64 / workers 8：
2 卡约 258 img/s vs 单卡 218 img/s ≈ 1.2×，而纯 GPU fwd+bwd（dummy loss）约 450-770 img/s）。先用
`scripts/bench_io.py` 定单卡 batch/workers；多卡时按“每卡 worker 数 × 卡数”分配，别把总 worker
数翻倍到核数超订。分割等重 GPU 步的任务，训练步占比更高，扩展比会更好。

**20 轮全量 COCO 实测**（2026-10-25，2×5090，yolo26n / fp32 / batch 64 / workers 8 / 全量 val）：
**2 h 38 min**（每轮训练 ~430 s + rank0 全量 val ~40 s，约 258 img/s），产物齐全（best/last/
best_raw/epoch020 · results.csv · diag · samples · plots · run.rank1.log）；best 权重经
`scripts/eval.py` 为 **E2E mAP50-95 0.2808 / mAP50 0.4073**（20 轮从零，只作 DDP 规模与吞吐
验收，不是精度结论；训练中 FastMetrics 报 0.2781，与正式口径同量级）。

## 8. 常见坑

1. **`--amp` 从零训练必炸**（fp16 在 640² 从零 + BN eps=0.001 的组合下第 1 轮内 NaN）。默认
   fp32；微调可以试 `--amp`。
2. **`--close-mosaic` 是"最后 N 轮"**（关闭 mosaic/mixup/cutmix/copy_paste），不是"从第 N 轮开始"；取值会自动 `min(N, epochs//5)`，
   短跑时小心它把 mosaic 全程关掉（`--close-mosaic 999` 这种写法等于全程关）。
3. **`best.safetensors` 由训练中快速指标（FastMetrics）选取**，只用于挑权重；对外数字一律走
   `scripts/eval.py`（见 `val.md`）。
4. **val 损失不可与训练损失直接比**（增强分布 / EMA vs raw / BN 模式三重差异），只做趋势。
5. 训练日志的损失列名由损失函数 `item_keys` 派生：yolo26 为 `box/cls/l1/o2m/o2o`，
   yolo26-seg / flash-yolo-seg 为 `box/seg/cls/l1/o2m/o2o/sem`，yolov3-tiny 为 `box/obj/cls`。
6. 换输入尺寸（非 640）**从零训练**时，模型先验偏置按训练 imgsz 计算（`bias_init(imgsz)`）；
   有锚模型（v3）则要先用 `compute_anchors.py` 重聚类（锚点不随 imgsz 缩放）。
7. **分割训练接受 COCO json 与 YOLO-seg txt**：COCO 的 RLE/缺失 segmentation、YOLO 的
   5 列 bbox 行都退回矩形并计入 `seg fallback`；COCO 全量 fallback 会报错（多半是描述符/
   标注选错）。YOLO 多边形行由 `cls x1 y1 x2 y2 ...` 解析，框取多边形外接框；
   COCO→YOLO-seg 用 `scripts/convert_coco_seg.py`（官方 merge 口径）。
   训练中 Mask 指标在 `imgsz/mask_ratio` 网格上做快速近似，**正式数字仍以 `scripts/eval.py`
   的 pycocotools 结果为准**。分割 CopyPaste 已实现官方 **flip** 模式（`copy_paste_mode: flip`），
   官方 seg 的 semseg 辅助损失（BCE-Dice）也已参与训练；复现官方 seg 配方用
   `--recipe yolo26-seg-coco-ft`。

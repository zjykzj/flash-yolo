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
```

## 2. 配方（recipe）

优先级：`config/train.yaml` 基础值 → 配方文件覆盖 → CLI 覆盖。

| 配方 | 阶段 | 前置条件 | 轮数（n/s/m/l/x） |
|---|---|---|---|
| *（不传）* | 从零 / 自有数据微调 | — | 100 |
| `yolo26-coco-ft` | 官方 COCO 阶段（微调） | **需 O365 预训练权重初始化** | 245/70/80/60/40 |
| `yolo26-seg-coco-ft` | 官方 seg COCO 阶段（微调） | 需 `yolo26*-objv1-seg`（365 类）初始化 + COCO json / YOLO-seg 描述符；n/s/m/l/x 全部按官方 seg ckpt 的 `train_args` 核对 | 245/50/80/60/40 |
| `yolo26-o365-pt` | 官方 Objects365 预训练 | 需 Objects365 数据集（未在本仓库验证） | 150 |

> `yolo26-seg-coco-ft` 的数值取自五个官方 seg ckpt 的 `train_args`：m/l/x 与官方检测表一致；
> s 来自另一提交的早期基线（epochs 50 / lr0 0.00125），n 的 mixup/degrees/shear/bgr 与检测表不同——
> 均按 seg ckpt 保留。配方显式写入 `mask_ratio: 1`、`overlap_mask: true`、`copy_paste_mode: flip`，
> semseg 辅助损失始终启用；未实现的 ckpt 参数（cls_w/o2m/erasing 等）见配方头部。从零训练仍用 default。

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
> 全量 COCO 245 轮约 **65 天/单卡**——完整复现需要多卡（当前 Trainer 单设备，DDP 未实现）。
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
| `--batch` | 16 | **物理** batch；机器相关，用 `bench_io.py` 实测后给 |
| `--nbs` | 64 | 名义 batch；`accum = max(round(nbs/batch),1)`，梯度语义与物理 batch 无关 |
| `--imgsz` | 640 | 训练输入（v3 常用 416）。batch 需整除 nbs（否则 wd 因子 ≠ 1.0） |
| `--workers` | 8 | 加载进程数；机器相关 |
| `--seed` | 0 | 全局种子（数据/增强/初始化） |
| `--deterministic` / `--no-deterministic` | true | 逐位可复现的 kernel 开关（关 cuDNN autotune + 确定性算法）；代价实测在噪声内，见 §5 |
| `--weights` | — | 初始化权重（微调） |
| `--resume` | — | `resume.pt` 或 run 目录（恢复优化器/EMA/轮次） |
| `--device` | auto | `cuda:0` / `cpu` |
| `--amp` / `--no-amp` | false | fp16+GradScaler；**从零训练会 NaN，仅微调用** |
| `--optimizer` | `musgd` | `musgd` = Muon+SGD 混合（官方 yolo26 口径）；`sgd` = 全参数走 SGD 路径（darknet 口径复刻：配 `--momentum 0.9 --no-nesterov --lr0 0.001` + 台阶 LR） |
| `--nesterov` / `--no-nesterov` | true | SGD 分支的 Nesterov 修正；darknet 的更新是 heavy-ball，复刻时用 `--no-nesterov` |
| `--obj-iou-target` | false | v3 系 IoU-aware objectness（参考 zjykzj/YOLOv3）：正样本 obj 目标 = IoU(预测框, 责任 GT)、权重 ×5；false = darknet 硬目标 0/1 |
| `--close-mosaic` | 10 | 最后 N 轮关 mosaic/mixup/copy_paste（自动 `min(N, epochs//5)`） |
| `--multi-scale` | 0.0 | 输入尺寸抖动 ±r（ultralytics 口径，按 32 量化） |
| `--copy-paste-mode` | `off` | `off` / `box`（检测矩形贴块近似）/ `flip`（官方分割 CopyPaste） |
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
3. **RNG 流域每 epoch 重播**（不依赖历史消费量）：worker 内由 `worker_init_fn` 播种、主进程
   （`workers=0`）由 `seed_main_rng(seed + epoch)` 播种；stdlib `random`（multi_scale 抽尺寸）也随之播种。
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

## 7. 常见坑

1. **`--amp` 从零训练必炸**（fp16 在 640² 从零 + BN eps=0.001 的组合下第 1 轮内 NaN）。默认
   fp32；微调可以试 `--amp`。
2. **`--close-mosaic` 是"最后 N 轮"**，不是"从第 N 轮开始"；取值会自动 `min(N, epochs//5)`，
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

# 训练（Train）

> 入口：`python scripts/train.py --data <name|.yaml> [选项]`。
> 数据见 [../datasets/README.md](../datasets/README.md)，模型见
> [../models/README.md](../models/README.md)，正式评估见 [eval.md](eval.md)。

## 1. 四条路径

```bash
# 从零训练（默认；内置 default 配方 100 轮）
python scripts/train.py --data coco --name my-run --batch 32 --workers 8

# 微调（用已有权重初始化；可配 --recipe 换微调配方）
python scripts/train.py --data coco --weights weights/yolo26n.safetensors --name ft

# 续训（恢复优化器 / EMA / 轮次 / 调度 / RNG）
python scripts/train.py --data coco --resume runs/train/train-my-run/resume.pt

# 换架构
python scripts/train.py --data coco --model flash-yolo --name flash-coco
python scripts/train.py --data coco --model yolo26-seg --name seg-coco --mask-ratio 4
python scripts/train.py --data coco --model yolov3-tiny --imgsz 416 --name v3tiny
python scripts/train.py --data coco --model yolov3-tiny-silu --imgsz 416 \
    --obj-iou-target --name v3tiny-silu
```

## 2. 配方（recipe）

优先级：`config/train.yaml` 基础值 → 配方文件覆盖 → CLI 覆盖。

| 配方 | 阶段 | 前置条件 | 轮数（n/s/m/l/x） |
|---|---|---|---|
| *（不传）* | 从零 / 自有数据微调 | — | 100 |
| `yolo26-coco-ft` | 官方 COCO 阶段（微调） | **需 O365 预训练权重初始化** | 245/70/80/60/40 |
| `yolo26-seg-coco-ft` | 官方 seg COCO 阶段 | 需 `yolo26*-objv1-seg`（365 类）+ COCO 多边形 | 245/50/80/60/40 |
| `yolo26-o365-pt` | 官方 Objects365 预训练（检测） | 需 Objects365（未验证） | 150 |
| `yolo26-seg-o365-pt` | 官方 Objects365 预训练（分割） | 需 Objects365v1 seg（未验证） | 150 |

```bash
python scripts/train.py --data coco --recipe yolo26-coco-ft \
    --weights weights/yolo26n-objv1-150.safetensors

python scripts/train.py --data coco --recipe yolo26-seg-coco-ft \
    --weights weights/yolo26n-objv1-seg.safetensors
```

- **从零训练只用 `default`**。`yolo26-coco-ft` 的 lr0 是基线的一半、衰减更陡，是针对
  “已收敛起点”调的；从零跑它等于把模型按在慢速档上（实测同轮次相对低 ~43%）。
- `yolo26-seg-coco-ft` 显式写 `mask_ratio: 1`、`overlap_mask: true`、
  `copy_paste_mode: flip`；semseg 辅助损失始终启用。
- 分割初始化是 365 类 O365 权重，`--weights` 走 key+shape 交集加载（类别头重建）。

## 3. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | —（必填） | 描述符 |
| `--name` | 空 → `runs/train/train` | run 目录后缀；已存在自动加序号 |
| `--recipe` | 内置 default | 配方名或 `.yaml` 路径 |
| `--model` | `yolo26` | `yolo26` / `yolo26-seg` / `flash-yolo` / `flash-yolo-seg` / `yolov3-tiny` / `yolov3-tiny-silu` |
| `--scale` | `n` | yolo26 系档位；其他架构固定 |
| `--mask-ratio` | 4 | 分割 GT 下采样倍率（`imgsz/mask_ratio`） |
| `--overlap-mask` | true | 实例索引图 vs 每实例二值掩码 |
| `--cfg` | — | 模型 yaml 覆盖 |
| `--epochs` | 100 | 总轮数 |
| `--batch` | 16 | **物理** batch（DDP 下为全局 batch） |
| `--nbs` | 64 | 名义 batch；`accum = max(round(nbs/batch),1)` |
| `--imgsz` | 640 | 训练输入（v3 常用 416） |
| `--workers` | 8 | 加载进程数（机器相关） |
| `--seed` | 0 | 全局种子 |
| `--deterministic` | true | 逐位可复现 kernel 开关 |
| `--weights` | — | 微调初始化 |
| `--resume` | — | `resume.pt` 或 run 目录 |
| `--device` | auto | `cuda:0` / `cpu` / `0,1`（DDP） |
| `--amp` | false | fp16+GradScaler；**从零训练会 NaN，仅微调可试** |
| `--optimizer` | `musgd` | `musgd`（Muon+SGD）或 `sgd`（darknet 口径） |
| `--obj-iou-target` | false | v3 IoU-aware objectness |
| `--close-mosaic` | 10 | 最后 N 轮关 mosaic/mixup/cutmix/copy_paste（自动 `min(N, epochs//5)`） |
| `--multi-scale` | 0.0 | 输入尺寸抖动 ±r |
| `--copy-paste-mode` | `off` | `off` / `box` / `flip`（官方分割 CopyPaste） |
| `--freeze` | 0 | 冻结前 N 层 |
| `--single-cls` | false | 单类训练 |
| `--plots` | true | 结束写 results/PR/混淆矩阵（需 matplotlib；`requirements-dev.txt` 已含，运行时按需安装） |
| `--channels-last` | true | NHWC 训练 |
| `--stop-after` | 0 | 只跑前 N 轮（调度仍按 `epochs`） |
| `--val-epochs` | 1 | 0 = 全程不验证 |
| `--limit` | 0 | 训练子集（前 N 张） |

其余超参都有同名 CLI 覆盖项，`--help` 列出全量；默认值即 `config/train.yaml`。

## 4. 运行产物（`runs/train/train-<name>/`）

| 产物 | 内容 |
|---|---|
| `weights/best.safetensors` | 按训练中指标选的 EMA 权重（正式数字请另行 eval） |
| `weights/last.safetensors` · `best_raw.safetensors` | 最后一轮 / 非 EMA 最佳 |
| `weights/epochNNN.safetensors` | 每 `save_period` 轮一存（中途分叉实验用） |
| `results.csv` | 逐轮训练损失 + 训练中指标 + val 损失 |
| `diag/train_diag.csv` | 每 `diag_interval` 步：裁剪前梯度范数、分组 lr、损失分解 |
| `samples/*.png` | 增强抽样网格 |
| `results.png` / `PR_curve.png` / `confusion_matrix*.png` | 训练结束报告 |
| `meta.json` | 环境 + 完整配置快照 |
| `run.log` | 控制台 + 文件日志 |
| `resume.pt` | 续训状态 |

权重 metadata 见 [../models/README.md](../models/README.md)。

## 5. 复现性

三层口径：

| 层 | 含义 | 保证 |
|---|---|---|
| **稳定** | 任何配置不发散不 NaN | fp32 默认 + 梯度裁剪 max_norm=10 |
| **逐位** | 同 commit + 同 seed + 同硬件/驱动 + 同 workers/batch → 权重逐位相同 | 默认达成（`deterministic: true`） |
| **统计** | 跨硬件/驱动 → 曲线在噪声内一致 | 未量化；A/B 用同机 + `--stop-after` 对齐 |

逐位复现需要同时成立：

1. `deterministic: true`（默认）——关 cuDNN autotune + 确定性算法 + 固定 CUBLAS 工作区。
2. `workers` / `batch` 是复现条件的一部分（增强种子按 `(seed, epoch, worker_id)` 派生）。
3. RNG 每 epoch 重播：种子 = `seed + epoch × world_size + rank`（单卡即 `seed + epoch`）。
4. resume 逐位等价：`resume.pt` 存四套 RNG 状态 + `ema.steps`；前提是同卡数同设备。

跨机器到不了逐位（kernel 不同），只承诺统计一致。新增随机性不要裸调全局
`np.random` / `random`。闸门：`FLASH_YOLO_REPRO=1 pytest tests/test_train_repro.py`。

## 6. 先把 I/O 调对

`--batch` / `--workers` 是机器相关值，仓库刻意不写进默认配置：

```bash
python scripts/bench_io.py --data coco            # loader-only 网格 -> 端到端前 K
python scripts/bench_io.py --data coco --loader-only
```

- **加载器快 ≠ 训练快**：只有端到端数字能决策。
- workers 不是越多越好（占满核会把主进程饿死）。
- 不要在 worker 里调 `cv2.setNumThreads`（OpenCV 线程池非 fork 安全，会整体卡死）；
  线程数由 `data/loader.py` 导入时在父进程设定。

## 7. 多卡训练（DDP）

只覆盖训练；eval / export / infer 仍是单进程。

```bash
python scripts/train.py --data coco --device 0,1 --batch 32 --workers 8 --name ddp
# 或 torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco --batch 32
```

- `--batch` 是**全局** batch：每卡 = `batch // world_size`，必须整除。
- `--workers` 是**每卡**进程数。
- 本工程损失是 Σ_img 口径，DDP backward 按 rank 平均 → 损失需乘 `world_size`
  （`train/distributed.py::ddp_scale_loss`；world=1 原样返回）。
- 验证只 rank 0 且全量；checkpoint/CSV/diag/samples/plots/meta.json/run.log 只 rank 0 写。
- `self.model` / `self.head` 始终是 raw module，DDP 只存 `self.ddp_model`。
- 无 SyncBatchNorm：BN 统计按卡独立；多卡与单卡不逐位一致，同 world_size 重复运行可逐位复现。
- `find_unused_parameters=True`（覆盖 freeze 与条件分支）。
- 梯度累积的非边界步把 forward+backward 包进 `no_sync()`。
- 换卡数/换设备续训支持（`resume.pt` 记录 world_size/device），但轨迹不再逐位一致。
- 吞吐：数据管线常先饱和；2×5090 / yolo26n@640 / batch 64 / workers 8 实测约 258 img/s
  vs 单卡 218（~1.2×）。先用 `bench_io.py` 定单卡值。

## 8. 常见坑

1. **`--amp` 从零训练必炸**（fp16 在 640² + BN eps=0.001 下第 1 轮内 NaN）。默认 fp32；
   微调可试 `--amp`。
2. **`--close-mosaic` 是“最后 N 轮”**，不是“从第 N 轮开始”；短跑会被 `min(N, epochs//5)` 缩小。
3. `best.safetensors` 由 FastMetrics 选取；对外数字走 `scripts/eval.py`。
4. val 损失不可与训练损失直接比（增强分布 / EMA vs raw / BN 模式）。
5. 训练日志的损失列名由 `item_keys` 派生：yolo26 为 `box/cls/l1/o2m/o2o`，
   分割多 `seg/sem`，v3 为 `box/obj/cls`。
6. 换输入尺寸从零训练时，先验偏置按训练 imgsz 计算；v3 需重聚类锚点。
7. 分割接受 COCO json 与 YOLO-seg txt；全量 fallback 会报错。

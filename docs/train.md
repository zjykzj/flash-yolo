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
```

## 2. 配方（recipe）

优先级：`config/train.yaml` 基础值 → 配方文件覆盖 → CLI 覆盖。

| 配方 | 阶段 | 前置条件 | 轮数（n/s/m/l/x） |
|---|---|---|---|
| *（不传）* | 从零 / 自有数据微调 | — | 100 |
| `yolo26-coco-ft` | 官方 COCO 阶段（微调） | **需 O365 预训练权重初始化** | 245/70/80/60/40 |
| `yolo26-o365-pt` | 官方 Objects365 预训练 | 需 Objects365 数据集（未在本仓库验证） | 150 |

```bash
python scripts/train.py --data coco --recipe yolo26-coco-ft --weights yolo26n-objv1-150.safetensors
```

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
| `--model` | `yolo26` | `yolo26` / `flash-yolo` / `yolov3-tiny` / `yolov3-tiny-silu`（算子替换版） |
| `--scale` | `n` | yolo26 档位 n/s/m/l/x；其他架构固定 |
| `--cfg` | — | 模型 yaml 覆盖（自定义结构 / 筛选变体） |
| `--epochs` | 100 | 总轮数（配方的调度也按它算） |
| `--batch` | 16 | **物理** batch；机器相关，用 `bench_io.py` 实测后给 |
| `--nbs` | 64 | 名义 batch；`accum = max(round(nbs/batch),1)`，梯度语义与物理 batch 无关 |
| `--imgsz` | 640 | 训练输入（v3 常用 416）。batch 需整除 nbs（否则 wd 因子 ≠ 1.0） |
| `--workers` | 8 | 加载进程数；机器相关 |
| `--seed` | 0 | 全局种子（数据/增强/初始化） |
| `--weights` | — | 初始化权重（微调） |
| `--resume` | — | `resume.pt` 或 run 目录（恢复优化器/EMA/轮次） |
| `--device` | auto | `cuda:0` / `cpu` |
| `--amp` / `--no-amp` | false | fp16+GradScaler；**从零训练会 NaN，仅微调用** |
| `--optimizer` | `musgd` | `musgd` = Muon+SGD 混合（官方 yolo26 口径）；`sgd` = 全参数走 SGD 路径（darknet 口径复刻：配 `--momentum 0.9 --no-nesterov --lr0 0.001` + 台阶 LR） |
| `--nesterov` / `--no-nesterov` | true | SGD 分支的 Nesterov 修正；darknet 的更新是 heavy-ball，复刻时用 `--no-nesterov` |
| `--obj-iou-target` | false | v3 系 IoU-aware objectness（参考 zjykzj/YOLOv3）：正样本 obj 目标 = IoU(预测框, 责任 GT)、权重 ×5；false = darknet 硬目标 0/1 |
| `--close-mosaic` | 10 | 最后 N 轮关 mosaic/mixup/copy_paste（自动 `min(N, epochs//5)`） |
| `--multi-scale` | 0.0 | 输入尺寸抖动 ±r（ultralytics 口径，按 32 量化） |
| `--channels-last` | true | NHWC 训练（cuDNN 反向快 ~33%）；推理/导出不受影响 |
| `--stop-after` | 0 | 只跑前 N 轮（lr/close_mosaic 仍按 `epochs` 调度 → 可与全长 run 逐轮对齐） |
| `--val-epochs` | 1 | 0 = 全程不验证（冒烟）；`--val-limit` 限制验证图数 |
| `--limit` | 0 | 训练子集（前 N 张） |

其余全部超参（优化器 / 损失 / 增强 / 产物周期）都有同名 CLI 覆盖项，`--help` 列出全量；
默认值即 `config/train.yaml`（= ultralytics `default.yaml`，逐项一致），文件内每项都有注释。

## 4. 运行产物（`runs/train/train-<name>/`）

| 产物 | 内容 |
|---|---|
| `weights/best.safetensors` | 按训练中指标选的 EMA 权重（**正式数字请另行 eval**） |
| `weights/last.safetensors` · `best_raw.safetensors` | 最后一轮 / 非 EMA 最佳 |
| `weights/epochNNN.safetensors` | 每 `save_period`（默认 20）轮一存，留最近 `keep_periodic` 个——**中途分叉实验用** |
| `results.csv` | 逐轮训练损失 + 训练中指标 + **val 损失**（val_box/val_cls/val_l1/val_o2m/val_o2o） |
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

## 5. 先把 I/O 调对（`scripts/bench_io.py`）

`--batch` / `--workers` 是机器相关值，仓库**刻意不写进默认配置**。用基准脚本决定：

```bash
python scripts/bench_io.py --data coco            # 两阶段：loader-only 网格 -> 端到端只跑前 K
python scripts/bench_io.py --data coco --loader-only
```

- **加载器快 ≠ 训练快**：只有端到端数字能决策（实测本机 loader-only 496 img/s 而端到端 ~230）。
- workers 不是越多越好（实测 16→20 反而更慢：worker 占满核，主进程被饿）。
- worker 里的 cv2/torch 线程数由框架统一管理（`data/loader.py` 导入时设定）——
  **不要在 worker 里调 `cv2.setNumThreads`**（OpenCV 线程池非 fork 安全，会整体卡死）。

## 6. 常见坑

1. **`--amp` 从零训练必炸**（fp16 在 640² 从零 + BN eps=0.001 的组合下第 1 轮内 NaN）。默认
   fp32；微调可以试 `--amp`。
2. **`--close-mosaic` 是"最后 N 轮"**，不是"从第 N 轮开始"；取值会自动 `min(N, epochs//5)`，
   短跑时小心它把 mosaic 全程关掉（`--close-mosaic 999` 这种写法等于全程关）。
3. **`best.safetensors` 由训练中快速指标（FastMetrics）选取**，只用于挑权重；对外数字一律走
   `scripts/eval.py`（见 `val.md`）。
4. **val 损失不可与训练损失直接比**（增强分布 / EMA vs raw / BN 模式三重差异），只做趋势。
5. 训练日志的损失列名由损失函数 `item_keys` 派生：yolo26 为 `box/cls/l1/o2m/o2o`，
   yolov3-tiny 为 `box/obj/cls`。
6. 换输入尺寸（非 640）**从零训练**时，模型先验偏置按训练 imgsz 计算（`bias_init(imgsz)`）；
   有锚模型（v3）则要先用 `compute_anchors.py` 重聚类（锚点不随 imgsz 缩放）。

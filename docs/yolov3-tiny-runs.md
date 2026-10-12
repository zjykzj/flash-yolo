# YOLOv3-tiny 训练与评估记录

> 数字口径：**正式** = `scripts/eval.py`（pycocotools，COCO val2017）；**训练中** = FastMetrics
> （纯 numpy 快速近似，只用于趋势与选 `best.safetensors`）。
> 本文是 v3 全部训练 run 的唯一留档（`runs/` 不入库）。架构与实现口径见 `yolov3-tiny-spec.md`。
> 三份关键产物的**完整 run 包**已复制归档到 `runs/v3-tiny-models/`（含 README 索引；同样不入库）。

## 1. 口径说明（数字怎么读）

- **两套指标并存**：训练中每轮打印 FastMetrics（mAP / mAP50），正式数字一律以 `eval.py` 为准。
  实测两者贴合良好（300 轮 run：训练中 0.1379 / 0.2658 → 正式 14.23 / 27.50）。
- **输入尺寸是 v3 的一等变量**：锚值是 darknet 固定像素单位（不随 imgsz 缩放），同一个权重
  换输入尺寸是"换口径"而不是"换配置"。官方发布的 33.1 AP50 是 **COCO test-dev @416**；
  本仓库全部数字是 **val2017**。
- **数据集**：本仓库用 COCO train2017（118,287 图 / 849,949 实例，训练侧剔 crowd 10,052）；
  darknet 发布的 `coco.data` 用 `trainvalno5k`（2014 train+val 减去 5k val 图）≈ 与本仓库
  同一批图，差异已考证排除。
- **轮数不同 ≠ 直接可比**：跨 run 比较要求 recipe 与预算同参（本页 run 均 seed 0、同增强；
  `v3-darknet-e3002` 是 batch 16、其余 batch 64 —— 梯度语义由 `nbs=64` 保证，但 b16 的小
  惩罚未分离，见 §4.2）。

## 2. 官方权重基准（配对齐点）

`scripts/download_weights.py --model yolov3-tiny` →
`scripts/convert_weights.py --src weights/yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors`
→ eval（转换器的字节对账见 spec 第 7 节）：

| 输入 | mAP@[.5:.95] | mAP@50 | P | R |
|---|---|---|---|---|
| 640 | **17.16** | **35.90** | 0.4883 | 0.3713 |
| 416 | 16.94 | 34.37 | 0.5040 | 0.3470 |

- 官方发布 33.1 AP50（test-dev @416）与上表 34.37（val2017 @416）的差 = 数据集/口径差，
  不是实现差；同权重 640 比 416 高是锚值不缩放 + 有效分辨率更高的结果。
- 这个基准是全部从零 run 的参照系：**训练侧的任务是把 14.23 与 16.94 之间的缝收掉**。

## 3. 全部训练 run

| run | 起始 | 配置（@416，seed 0） | 状态 | 训练中 best mAP / mAP50 | 正式 mAP@[.5:.95] / mAP@50 |
|---|---|---|---|---|---|
| `v3-coco`, `v3-coco2`–`coco5` | 10-08 晚 | 100ep，640/416 反复试 | 开发期中断（均未跑完第 1 轮） | — | — |
| `v3-coco20` | 10-08 20:54 | 20ep | 6 轮后让位（改 30 轮重跑） | 0.0396 | — |
| `v3-coco6` | 10-08 21:25 | 30ep，2 h 04 m | 完成（首个跑完的 run） | 0.0900 / 0.1859 | — |
| `v3-coco-e100` | 10-09 00:06 | 100ep，default 配方，6 h 49 m | 完成 | 0.1231 / 0.2401 | **12.70 / 24.84** |
| `v3-coco-e300` | 10-09 10:24 | 300ep，default 配方，22 h 41 m | 完成 | 0.1379 / 0.2658 | **14.23 / 27.50** |
| `v3-darknet-e300` | 10-10 14:03 | 300ep，darknet 配方，batch 64 | 4 分钟后中断（与 flash 同跑时 reserved 冲到 30.35G，改 batch 16 重启） | — | — |
| `v3-darknet-e3002` | 10-10 14:13 | 300ep，darknet 配方，batch 16，43 h 53 m | 完成 | 0.1233 / 0.2416（best 在末轮 ep300） | **12.76 / 25.09** |

（`v3-darknet-e3002` 的正式数字取 `weights/best.safetensors` = EMA 副本，评估口径与其余
run 同：`scripts/eval.py --data coco --model yolov3-tiny`，imgsz 由 metadata 还原 416。）

```bash
# 100 轮（正式口径 12.70 / 24.84）
python scripts/train.py --data coco --model yolov3-tiny --name v3-coco-e100 --batch 64 --workers 16 --imgsz 416
python scripts/eval.py --weights runs/train/v3-coco-e100/weights/best.safetensors --data coco

# 300 轮（正式口径 14.23 / 27.50）
python scripts/train.py --data coco --model yolov3-tiny --name v3-coco-e300 --batch 64 --workers 16 --imgsz 416 --epochs 300
python scripts/eval.py --weights runs/train/v3-coco-e300/weights/best.safetensors --data coco

# darknet 配方（多尺度 + 台阶 LR；batch/workers 为机器相关值）
python scripts/train.py --data coco --model yolov3-tiny --recipe yolov3-tiny-darknet \
    --name v3-darknet-e300 --batch 16 --workers 8
```

说明：

- 训练日志/csv 的损失列名由 `ComputeLossV3.item_keys` 派生（`box / obj / cls`，
  取代 yolo26 的 `box / cls / l1 / o2m / o2o`）。
- `v3-coco20` 与 `v3-darknet-e300` 的"中断"是**主动重启**（配置调整），不是崩溃 ——
  两者的 `run.log` 里都没有 traceback，只有训练在某一轮后戛然而止。
- darknet 配方的中途值（0.0948 @150 轮）**远低于**同轮次的 default run。当时归因于"收益
  集中在末段（台阶在 240/270 轮）+ 多尺度在固定 416 上验证吃亏"——**跑完后半句被证伪**：
  末段确实追得快（见 §4.2），但平段被白扔的 140 轮追不回来，终点 12.76 / 25.09 ≈ default
  **100 轮**（12.70 / 24.84）的水平。

## 4. 结论：预算 vs 配方（残差归因）

### 4.1 预算口径修正：darknet 的"一次迭代"是 32 张图

darknet `yolov3-tiny.cfg` 训练档（AlexeyAB 仓库同款）：`batch=64 / subdivisions=2` →
**每次权重更新只吃 32 张**；`max_batches=500200`、`steps=400000,450000`（= 80% / 90%）、
`scales=.1,.1`、`lr=0.001`、`momentum=0.9`、`decay=0.0005`、`burn_in=1000`、416²、
`random=1` 多尺度（{320..608}、每 10 batch 换一次、随机交换宽高比）。

换算成"看过的图"：官方 tiny 训练 = 500200 更新 × 32 图 = **16.0M 图 ≈ 135 个 COCO epoch**
（train2017 = 118,287 图/轮）。我们：100 轮 = 11.8M 图 / 185k 更新；300 轮 = **35.5M 图 /
554k 更新**。→ 图片数口径下 **300 轮已经是官方 tiny 的 2.2×，更新次数（554k vs 500k）也
相当**。此前"184.9k vs 500.2k iters（2.7× 欠训练）"的算法漏了 subdivisions —— 预算解释
不了 14.23 vs 16.94 的 2.71 点缝，剩下的候选只有**配方/增强/优化器口径**。

### 4.2 darknet 配方实验（`v3-darknet-e3002`）：决定性变量是平段 lr

配方 = default 基线 + 多尺度（±0.46）+ 台阶 LR（80%/90% ×0.1），单次两变量。300 轮跑完，
正式 **12.76 / 25.09** ≈ default **100 轮**水平，且同轮次**全线落后**：

| ep | default（linear，无 ms，b64） | darknet 配方（step+ms，b16） |
|---|---|---|
| 20 | 0.0694 / 0.1521 | 0.0614 / 0.1358 |
| 100 | 0.1079 / 0.2173 | 0.0920 / 0.1913 |
| **120 → 240** | **+0.0164** | **+0.0040（基本没动）** |
| 240 | 0.1271 / 0.2512 | 0.0969 / 0.1990 |
| 240 → 250 | +0.0014 | **+0.0088（台阶落地）** |
| 300 | 0.1379 / 0.2658 | 0.1233 / 0.2416 |

（列格式：训练中 mAP / mAP50；两 run 同 seed 0、同增强、同 416、同 nbs=64。）

机制（三条互相印证）：

1. **平段 lr=0.01 把模型按在噪声底**：ep120→240（占预算 40%）mAP 净增 +0.004，train box
   损失 0.4290→0.4268、val box 损失 0.3576→0.3559 同步平——不是容量/数据到顶的形貌。
2. **台阶一落就阶跃解锁**：ep241 lr 0.01→0.001，10 轮内 mAP +9%（0.0969→0.1057）、box
   损失 0.4268→0.3982；台阶后 30 轮斜率 +0.0193 是 default 同期（+0.0049）的 **4×**
   ——230 轮里 140 轮被白扔，模型离自身上界还很远。
3. **它不是 darknet 的配方数值**：darknet 平段 lr 是 0.001（SGD，见 4.1），我们沿用 default
   的 lr0=0.01（MuSGD），只抄了形状（fracs/gamma）。多尺度也不同：我们对称 ±46% →
   [224,608]、**每 batch** 换（ultralytics `preprocess_batch` 口径），darknet 是
   {320..608}、每 10 batch 换、下界高 96px。
   → 这次 A/B 实际测的是"平段 lr 10× 偏高的 step 调度"，**并没有否证 darknet 配方**。

**未分离变量（诚实记账）**：① 早期 -14%（ep5：0.0281 vs 0.0326，当时两边 lr 都 ≈0.01、
warmup 一致）在台阶差异生效前就出现，只能来自多尺度（和/或 b16），本次未做单变量分离；
② 本 run 是 b16（OOM 改的）而 default 是 b64 —— ep1 两条 run 近乎逐位一致
（0.0007/0.0007，val 损失差 1~4%），管线等价无疑，但 b16 的小惩罚不能排除。

### 4.3 当前位置

| 模型（全部 @416，val2017） | 正式 mAP50-95 / mAP50 |
|---|---|
| 官方 darknet 权重（同管线） | 16.94 / 34.37 |
| **default 300 轮（当前最优自训）** | **14.23 / 27.50** |
| darknet 配方 300 轮（step@lr0=0.01 + ms） | 12.76 / 25.09 |
| default 100 轮 | 12.70 / 24.84 |

### 4.4 下一步（暂缓）

首选单变量：`yolov3-tiny-darknet.yaml` 只把平段 lr0 降到 0.001~0.002
再跑。先探针后跑满：`--lr0 0.002 --stop-after 100`（b64 单跑约 7 h），在 ep60 / ep100 对标
default 的 0.0983 / 0.1079 与本次的 0.0852 / 0.0920（`stop_after` 不改调度，可逐轮次直接
对齐）；有信号再 `--resume` 跑满 300。

## 5. 对照实验（已完成）

- **输入尺寸 vs 锚值缩放**（官方权重，500 图 FastMetrics）：416 原生 0.384 ·
  640 + 原始锚值 0.412 · 640 + 线性缩放锚值 **0.154** —— 锚值必须原样保留，详见 spec 第 4 节。
- **锚点重聚类**（`scripts/compute_anchors.py`，COCO@640）：官方锚 coverage 97.0% /
  bestIoU 0.520 vs 重聚类 96.1% / 0.522 → 对 COCO 形数据**不必重算**；自定义数据集先跑
  verdict 再定。
- **输入尺寸对 v3 近乎不敏感**：官方权重 640 → 416 只掉 0.22 点（17.16 → 16.94），
  而 yolo26n 同样换到 416 会掉 5.96 点（40.27 → 34.31）。所以第 4 节的"从零 vs 官权"差距
  （双方同为 416 口径）与输入尺寸无关，也已被 §4.1 排除预算，是**配方/增强/优化器口径**问题。
  （参考系：官方 YOLO26n 权重本管线 416 → 34.31 / 48.61、512 → 37.49 / 52.59。）

## 6. 常用命令（v3 全套）

```bash
python scripts/train.py --data <name|.yaml> --model yolov3-tiny --batch 64 --workers 16 --imgsz 416
python scripts/eval.py  --weights weights/yolov3-tiny.safetensors --data coco            # E2E 开关不适用
python scripts/eval.py  --weights w.safetensors --data coco --imgsz 416                  # 输入尺寸：CLI > metadata
python scripts/infer.py --weights w.safetensors --image assets/bus.jpg
python scripts/export.py --weights w.safetensors --model yolov3-tiny                    # 单输出 (B, NA, 5+nc)
python scripts/compute_anchors.py --data <name|.yaml>                                    # 锚点先验评估/重聚类
```

`--nms` / `--raw` 是 yolo26 专属开关，对 v3 直接报错；v3 只有一条"解码 + obj×cls + 类感知
NMS"路径。

## 7. 待补（暂缓，非阻塞）

- [ ] 分离"多尺度税"：`--lr-schedule linear`（保留 ms、去掉台阶）短跑，逐轮次对标 default
- [ ] darknet 配方的修正版：`--lr0 0.001~0.002`（平段降到可持续档），可选再把 ms 下界对齐 320
- [ ] 缝若能再收，回填 §2 / §4.3 的对照表与 README 的 v3 数字

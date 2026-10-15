# YOLOv3-tiny 训练与评估记录

> 数字口径：**正式** = `scripts/eval.py`（pycocotools，COCO val2017）；**训练中** = FastMetrics
> （纯 numpy 快速近似，只用于趋势与选 `best.safetensors`）。
> 本文是 v3 全部训练 run 的唯一留档（`runs/` 不入库）。架构与实现口径见 `yolov3-tiny-spec.md`。
> 关键产物的**完整 run 包**已复制归档到 `runs/v3-tiny-models/`（含 README 索引；同样不入库）。

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
| `v3-softobj-e300`（`-e3003`） | 10-14 13:36 | 300ep，default + IoU-aware objectness | 完成（≈31 h，两条并发） | 0.1586 / 0.2913（best 在末轮 ep300） | **16.07 / 29.56** |
| `v3-silu-softobj-e300`（`-e3004`） | 10-14 13:36 | 300ep，SiLU + IoU-aware objectness（**推荐配置**） | 完成（≈31 h，两条并发） | 0.1630 / 0.3000（best 在末轮 ep300） | **16.52 / 30.44** |

（`v3-darknet-e3002` 的正式数字取 `weights/best.safetensors` = EMA 副本，评估口径与其余
run 同：`scripts/eval.py --data coco --model yolov3-tiny`，imgsz 由 metadata 还原 416。）

```bash
# 100 轮（正式口径 12.70 / 24.84）
python scripts/train.py --data coco --model yolov3-tiny --name v3-coco-e100 --batch 64 --workers 16 --imgsz 416
python scripts/eval.py --weights runs/train/v3-coco-e100/weights/best.safetensors --data coco

# 300 轮（正式口径 14.23 / 27.50）
python scripts/train.py --data coco --model yolov3-tiny --name v3-coco-e300 --batch 64 --workers 16 --imgsz 416 --epochs 300
python scripts/eval.py --weights runs/train/v3-coco-e300/weights/best.safetensors --data coco

# 推荐配置（§4.5；也即 §3.1 组合轮探针的命令，把 --epochs 提到 300 即可）
python scripts/train.py --data coco --model yolov3-tiny-silu --obj-iou-target \
    --imgsz 416 --epochs 300 --batch 64 --nbs 64 --name v3-best-e300
```

（darknet 配方的跑法已随配方退役——`config/recipes/yolov3-tiny-darknet.yaml` 2026-10-14
删除，定案见 §4.4。）

说明：

- 训练日志/csv 的损失列名由 `ComputeLossV3.item_keys` 派生（`box / obj / cls`，
  取代 yolo26 的 `box / cls / l1 / o2m / o2o`）。
- `v3-coco20` 与 `v3-darknet-e300` 的"中断"是**主动重启**（配置调整），不是崩溃 ——
  两者的 `run.log` 里都没有 traceback，只有训练在某一轮后戛然而止。
- darknet 配方的中途值（0.0948 @150 轮）**远低于**同轮次的 default run。当时归因于"收益
  集中在末段（台阶在 240/270 轮）+ 多尺度在固定 416 上验证吃亏"——**跑完后半句被证伪**：
  末段确实追得快（见 §4.2），但平段被白扔的 140 轮追不回来，终点 12.76 / 25.09 ≈ default
  **100 轮**（12.70 / 24.84）的水平。

### 3.1 100 轮单变量探针波次（wave-1 + wave-2，2026-10-12 → 10-14）

对照 ref = `v3-coco-e100`（训练中 0.1231 / 0.2401）；Δ = **同轮次训练中读数差**（末轮 vs 对照
同轮次），正式数字只对需要盖章的 run 做（`scripts/eval.py`）。

| run | 单变量 | 停止 | 末轮 Δ vs 对照 | 正式 | 判读 |
|---|---|---|---|---|---|
| `v3-step100` | lr_schedule step（形状） | ep49 主动中断 | −0.0067 / −0.0091 | — | 🔴 |
| `v3-nomosaic-p100` | mosaic 0 | ep60 主动中断 | −0.0065 / −0.0126 | — | 🔴 |
| `v3-cosine-p100` | lr_schedule cosine | 跑满 | +0.0000 / +0.0015 | — | 🔴 平手 |
| `v3-sched-lr5e3-p100` | darknet 配方 + lr0 0.005 | ep53 主动中断 | 同配方 +0.0055/+0.0063；vs ref −0.0121/−0.0220 | — | 🔴 修配方 ≠ 超 default |
| `v3-musgd-b32-p100`（F4） | b32/nbs32 步数×2（lr0 0.005） | 跑满 | −0.0005 / −0.0034 | 12.65 / 24.49 | 🔴 平手 |
| `v3-sgd-p100`（E） | 官方 SGD 口径，b64 | ep60 砍 | −0.0574 / −0.1044 | — | 🔴🔴 |
| `v3-official-b32-p100`（F1） | 官方 SGD 口径，b32 | 跑满 | −0.0494 / −0.0815 | — | 🔴🔴 |
| `v3-sgd01-p100`（B） | `--optimizer sgd` lr0 0.01（关 Muon） | ep40 判官砍 | −0.0349 / −0.0637 | — | 🔴 |
| `v3-ms-b32-p100`（C） | + multi_scale 0.46（配对 F4） | ~ep41 主动中断 | −0.0099 / −0.0161（vs F4） | — | 🔴 未收敛 |
| `v3-lr15b64-p100`（G′） | lr0 0.015 | ~ep30 主动砍 | −0.0012 / −0.0026 | — | 🔴 平手 |
| `v3-lr5e3b64-p100`（H） | lr0 0.005 | ep60 判官砍 | −0.0011 / −0.0011 | — | 🔴 平手 |
| **`v3-softobj-p100`（A）** | **`--obj-iou-target` IoU-aware objectness** | 跑满 | **+0.0234 / +0.0352** | **14.85 / 27.97** | 🟢 |
| **`v3-silu-b32-p100`** | **SiLU 算子替换（配对 F4）** | 跑满 | **+0.0045 / +0.0105** | **13.06 / 25.47** | 🟢 |
| **`v3-silu-softobj-p100`（组合）** | **SiLU + IoU-aware objectness（= 推荐配置）** | 跑满 | **+0.0274 / +0.0418** | **15.23 / 28.58** | 🟢 |

- 全部探针 seed 0 / @416 / 同增强；b64 档 batch 64 nbs 64，b32 档 batch 32 nbs 32（F4 系列
  "步数×2" = 与 b64 每轮位移对齐）；未标明的 lr 调度均为 linear。
- **组合轮两增益可加**：combo−A（同为 b64）末轮 +0.0040 / +0.0066（训练中）· +0.0038 / +0.0061
  （正式）≈ SiLU 在 b32 配对的增量（+0.0045/+0.0105 训练中 · +0.0041/+0.0098 正式）。
- 另有 `v3-lr15-p100`：lr0.015 的重复启动（让位于 `v3-lr15b64-p100`），4 轮即停，未计入。
- **判读口径**：|Δ| ≤ 0.003 记平手；≥ +0.003 且跨轮次保持记 🟢；中途砍走的走"结构化提前
  结束"（`--stop-after` 保持调度可比，resume.pt 可补跑）。

## 4. 结论：预算 vs 配方（残差归因）

### 4.1 预算口径：官方 = 500,200 次更新 × 64 图 = 32.0M 图 ≈ 270 轮

darknet `yolov3-tiny.cfg` 训练档（AlexeyAB 仓库同款）：`batch=64 / subdivisions=2`、
`max_batches=500200`、`steps=400000,450000`（= 80% / 90%）、`scales=.1,.1`、`lr=0.001`、
`momentum=0.9`、`decay=0.0005`、`burn_in=1000`、416²、`random=1` 多尺度（{320..608}、
每 10 batch 换一次、随机交换宽高比）。

**`subdivisions` 是显存切分 + 梯度累积，不是"每次更新吃几张"**（2026-10-12 本地 darknet
源码三处互证，此前本文"32 图/更新 → 135 轮"的算法把累积当成了减半，已作废）：
`src/parser.c` 的 `net->batch /= subdivs` → 单次前向 32 张；`src/network.c` 的
`train_network_datum` 每 chunk `seen += net->batch`（32），而
`if((seen/batch) % subdivisions == 0) update_network(net)` → **每 2 个 chunk 更新一次，
一次更新吃满 64 张**；`network.c` 的 `get_current_batch` 用 `seen/(batch*subdivisions)`
计数、`examples/detector.c` 的 `imgs = batch*subdivisions`（= 64）→ **`max_batches=500200`
数的就是更新次数**。

| 口径 | 更新次数 | 看过的图 | ≈ COCO 轮数（118,287 图/轮） |
|---|---|---|---|
| 官方 tiny | 500,200 | 500200 × 64 = **32.0M** | **270.6 轮** |
| 我们 100 轮 @b64/nbs64 | 184.9k | 11.8M | 0.37×（**确实欠训练**） |
| 我们 300 轮 @b64/nbs64 | 554.7k | 35.5M | 1.11×（同量级） |

`steps` 换算：400k / 450k 更新 = 第 **216.4 / 243.5** 轮（80% / 90% 位置）；`burn_in=1000`
更新 ≈ 0.54 轮。**官方口径的对照预算 = 270 轮 @b64/nbs64**（135 轮 @b32/nbs32 的更新数
相同，但每次更新的梯度只来自 32 张，不等价）。

结论（方向不变、理由更正）：**300 轮与官方是 1.11× 同量级**（不是 2.2× 超出），
14.23 vs 16.94 的 2.71 点缝仍只能记在**配方/增强/优化器口径**上；同时恢复一条被错误
"修正"过的事实——**100 轮 = 官方的 0.37×，是真的欠训练**，这解释了 100 轮那批探针为何
始终在低位徘徊。

### 4.2 darknet 配方实验（`v3-darknet-e3002`）：决定性变量是平段 lr

配方 = default 基线 + 多尺度（±0.46）+ 台阶 LR（80%/90% ×0.1），单次两变量（配方文件
2026-10-14 退役删除，定案见 §4.4）。300 轮跑完，
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

### 4.3 当前位置（2026-10-15）

| 模型（全部 @416，val2017） | 正式 mAP50-95 / mAP50 |
|---|---|
| 官方 darknet 权重（同管线） | 16.94 / 34.37 |
| **自训最优：SiLU + IoU-aware objectness 300 轮（推荐配置，§4.5）** | **16.52 / 30.44** |
| 原版 + IoU-aware objectness 300 轮 | 16.07 / 29.56 |
| SiLU + IoU-aware objectness 100 轮 | 15.23 / 28.58 |
| IoU-aware objectness 单轴 100 轮 | 14.85 / 27.97 |
| default 300 轮 | 14.23 / 27.50 |
| darknet 配方 300 轮（step@lr0=0.01 + ms） | 12.76 / 25.09 |
| default 100 轮 | 12.70 / 24.84 |

### 4.4 定案：darknet 配方族退役（2026-10-14）

§4.1 排除了预算、§4.2 记录了 darknet 配方 300 轮落败；此后四条探针（表 §3.1）把剩余嫌疑
闭环，**darknet 配方方向（`yolov3-tiny-darknet.yaml` 及其修正版）不再推进，配方文件已删**：

1. **官方 SGD 数值在本管线不可复刻**：本仓库训练循环无条件 clip(max_norm=10)（yolo26 官方
   训练循环的一部分）。v3 实测裁剪前 ‖g‖ ≈ 1.1k–1.6k → SGD 路径实际步长被锁成
   lr×10×单位方向；darknet 原版无裁剪、步长 ∝ ‖g‖ —— 同 lr 差 ~100×。E/F1 用 darknet 的
   lr=0.001 进裁剪管线 = 把动力学压扁 100 倍（表 §3.1）——**这不构成"darknet 配方差"的证据，
   只说明"在本工程（含裁剪）内复刻官方 SGD 口径不可行"**。
2. **裁剪口径的正确 lr（0.01）+ 关 Muon（B）仍无收益**（ep40 −0.0349）→ Muon+Momentum 系
   （default）在 v3-tiny 尺度不劣于朴素 SGD。
3. **多尺度（C，严格配对 F4）全程落后且未收敛；lr0 双括号 0.005 / 0.015（H / G′）均打平**
   → lr0 0.01 是真实局部最优。用户记忆里"lr 0.01→0.005 有效"也核清了：那是
   `v3-sched-lr5e3-p100` 修 darknet 配方的坏档（其平段 lr0 0.01 偏高），对 default 无此效应。
4. 增强与数据差异此前已排除（§1、§4.1）。

→ 主线收敛为：**default 调参已是局部最优，继续收益必须来自模型/损失算子**（表 §3.1 的两条
绿线，正是"纯算子替换、精度至少不降"原则下的结果）。

### 4.5 最终交付：推荐配置（2026-10-15 实测定稿）

**最佳模型配置**：`--model yolov3-tiny-silu` + `--obj-iou-target`
（SiLU 算子替换 + IoU-aware objectness。参数量 8,852,366 / GFLOPs 5.56 @416 与原版逐位一致，
仅激活函数与物体度目标变化；推理延迟不变）。

**最佳训练参数配置**：内置 default 配方原样（`config/train.yaml`：lr0 0.01 · linear 退火 ·
MuSGD momentum 0.937 · wd 0.0005 · mosaic 1.0 …，逐项 = ultralytics default.yaml），
`--imgsz 416 --batch 64 --nbs 64`。

```bash
# 最终 300 轮（2026-10-15 已完成：16.52 / 30.44；run = v3-silu-softobj-e3004）
python scripts/train.py --data coco --model yolov3-tiny-silu --obj-iou-target \
    --imgsz 416 --epochs 300 --batch 64 --nbs 64 --name v3-silu-softobj-e300
```

100 轮证据（正式 val2017 @416）：**15.23 / 28.58** vs default 100 轮 12.70 / 24.84
（+2.53 / +3.74）——IoU-aware objectness 单轴贡献 +2.15 / +3.13、SiLU 单轴 +0.41 / +0.98（b32 配对口径），
组合 ≈ 两轴之和；**100 轮即超 default 300 轮**（14.23 / 27.50）。

**300 轮实测（最终数字）**：**16.52 / 30.44**（P 0.4612 / R 0.3216，best 在末轮 ep300）——
vs default 300 轮 **+2.29 / +2.94**，与官方权重的 **mAP50-95 缝只剩 0.42**（16.52 vs 16.94），
mAP50 差 3.93（30.44 vs 34.37）。激活轴 A/B（同 300 轮、只差 `--model`）：原版 + objectness =
**16.07 / 29.56**（vs default +1.84 / +2.06，P 0.4436 / R 0.3150）——silu 增量 **+0.45 / +0.88**
在长预算下保持（100 轮配对 +0.41 / +0.98）。默认 300 轮的正式数字复核复现一致（14.23 / 27.50）。

## 5. 对照实验（已完成）

- **输入尺寸 vs 锚值缩放**（官方权重，500 图 FastMetrics）：416 原生 0.384 ·
  640 + 原始锚值 0.412 · 640 + 线性缩放锚值 **0.154** —— 锚值必须原样保留，详见 spec 第 4 节。
- **锚点重聚类**（`scripts/compute_anchors.py`，COCO@640）：官方锚 coverage 97.0% /
  bestIoU 0.520 vs 重聚类 96.1% / 0.522 → 对 COCO 形数据**不必重算**；自定义数据集先跑
  verdict 再定。
- **输入尺寸对 v3 近乎不敏感**：官方权重 640 → 416 只掉 0.22 点（17.16 → 16.94），
  而 yolo26n 同样换到 416 会掉 5.96 点（40.27 → 34.31）。所以第 4 节的"从零 vs 官权"差距
  （双方同为 416 口径）与输入尺寸无关，也已被 §4.1 排除预算，是**配方/增强/优化器口径**问题。
  （2026-10-14 补注：这些旋钮此后被 §3.1 波次逐一检验——LR 形状/数值、多尺度、优化器、
  batch 步数均无绿项；剩余缝的归因收窄到"darknet 训练实现口径 vs 本仓库"本身，见 §4.4。）
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

## 7. 待补（非阻塞）

- [x] 最终 300 轮 run（§4.5 命令）——**2026-10-15 完成：16.52 / 30.44**（silu+softobj）与
      16.07 / 29.56（原版+softobj），已回填 §3 / §4.3 / §4.5
- [ ] 416 锚点重聚类（prior-fit 差 0.024，可选；COCO 形数据收益有限，见 spec 第 8 节）
- [ ] 缝若能再收，回填 §2 / §4.3 的对照表与 README 的 v3 数字

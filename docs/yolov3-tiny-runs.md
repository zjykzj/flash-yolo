# YOLOv3-tiny 训练与评估记录

> 数字口径：**正式** = `scripts/eval.py`（pycocotools，COCO val2017）；**训练中** = FastMetrics
> （纯 numpy 快速近似，只用于趋势与选 `best.safetensors`）。
> 本文是 v3 全部训练 run 的唯一留档（`runs/` 不入库）。架构与实现口径见 `yolov3-tiny-spec.md`。

## 1. 口径说明（数字怎么读）

- **两套指标并存**：训练中每轮打印 FastMetrics（mAP / mAP50），正式数字一律以 `eval.py` 为准。
  实测两者贴合良好（300 轮 run：训练中 0.1379 / 0.2658 → 正式 14.23 / 27.50）。
- **输入尺寸是 v3 的一等变量**：锚值是 darknet 固定像素单位（不随 imgsz 缩放），同一个权重
  换输入尺寸是"换口径"而不是"换配置"。官方发布的 33.1 AP50 是 **COCO test-dev @416**；
  本仓库全部数字是 **val2017**。
- **数据集**：本仓库用 COCO train2017（118,287 图 / 849,949 实例，训练侧剔 crowd 10,052）；
  darknet 发布的 `coco.data` 用 `trainvalno5k`（2014 train+val 减去 5k val 图）≈ 与本仓库
  同一批图，差异已考证排除。
- **轮数不同 ≠ 直接可比**：跨 run 比较要求 recipe 与预算同参（本页 run 均 seed 0、同增强）。

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
| `v3-darknet-e300` | 10-10 14:03 | 300ep，darknet 配方，batch 64 | 4 分钟后中断（改 batch 16 重启） | — | — |
| `v3-darknet-e3002` | 10-10 14:13 | 300ep，darknet 配方，batch 16 | **运行中（~150/300）** | 0.0948（中途值，勿当结论） | `--` |

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
- darknet 配方的中途值（0.0948 @150 轮）**远低于**同轮次的 default run，属预期：台阶 LR
  的两档衰减在 240 / 270 轮（`lr_step_fracs: [0.8, 0.9]`），收益集中在末段；
  同时多尺度训练在固定 416 上验证本身吃亏。**不要拿中途值下结论。**

## 4. 结论：预算 vs 配方（残差归因）

1. **预算是第一变量**：100 → 300 轮，正式 +1.53 mAP（12.70 → 14.23）。但 300 轮末 10 轮
   仍 +0.0034（FastMetrics）——曲线"放缓未平"，不完全是收敛态。
2. **预算不是全部**：300 轮止步 14.23，距官方权重同管线同输入（16.94 @416）还有 2.71 点。
   预算已被排除（+200 轮只买到 1.5 点且没跑干净），剩余候选 = 官方权重自带的
   **O365 式先验 + 多尺度训练（darknet `random=1`）+ 台阶 LR（steps 400k/450k of 500k）**。
   前者不可得，后两者落在 `config/recipes/yolov3-tiny-darknet.yaml` 里做单变量归因：
   `--multi-scale 0` 只留 step / `--lr-schedule linear` 只留多尺度。
3. 该归因实验 = `v3-darknet-e3002`（运行中），完成后补正式数字并回填上表。

## 5. 对照实验（已完成）

- **输入尺寸 vs 锚值缩放**（官方权重，500 图 FastMetrics）：416 原生 0.384 ·
  640 + 原始锚值 0.412 · 640 + 线性缩放锚值 **0.154** —— 锚值必须原样保留，详见 spec 第 4 节。
- **锚点重聚类**（`scripts/compute_anchors.py`，COCO@640）：官方锚 coverage 97.0% /
  bestIoU 0.520 vs 重聚类 96.1% / 0.522 → 对 COCO 形数据**不必重算**；自定义数据集先跑
  verdict 再定。
- **输入尺寸对 v3 近乎不敏感**：官方权重 640 → 416 只掉 0.22 点（17.16 → 16.94），
  而 yolo26n 同样换到 416 会掉 5.96 点（40.27 → 34.31）。所以第 4 节的"从零 vs 官权"差距
  （双方同为 416 口径）与输入尺寸无关，是预算/配方问题。
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

## 7. 待补

- [ ] `v3-darknet-e3002` 跑完 → 正式 eval → 填入第 3 节表格与第 4 节归因结论
- [ ] （可选）单变量 A/B：`--multi-scale 0` / `--lr-schedule linear` 各一个短跑筛一遍

# 评估（Eval）

> 入口：`python scripts/eval.py --weights <w> --data <name|.yaml>`。
> **对外数字只有一个来源就是本脚本**：COCO 协议走 pycocotools，LVIS 协议走 faster_coco_eval
> （`lvis_style`）；训练中打印的 FastMetrics 只用于趋势与选 best。

## 1. 用法

```bash
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco                # E2E
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms          # o2m + NMS
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data coco         # 换运行时
python scripts/eval.py --weights w.safetensors --data coco --imgsz 416                  # 指定输入
python scripts/eval.py --weights w.safetensors --data coco --split val --limit 100      # 子集
python scripts/eval.py --weights weights/yolo26n-seg.safetensors --data coco            # Box + Mask
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose      # Pose (OKS)
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose --nms  # Pose o2m+NMS
python scripts/eval.py --weights weights/yoloe-26n-seg-pf.safetensors --data coco       # pf 词汇映射
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis          # 文本 prompt（1203 类）
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival  # LVIS minival（5k）
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco --visual-prompt prompt.json  # 视觉 prompt
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --refer-data lvis --refer-split train --refer-limit 1000                            # refer 数据集视觉 prompt
python scripts/eval.py --weights w.safetensors --data lvis-minival --eval-max-dets 100  # 对齐 ultralytics 口径
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--data` | — | 描述符（见 [../datasets/README.md](../datasets/README.md)）；yoloe 文本模式用其类别名做 prompt |
| `--split` | `val` | 角色键（train/val/test） |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--model` / `--scale` | 从权重推断 | 推不出时显式给 |
| `--nms` | off | o2m + NMS（yolo26 系含 pose + yoloe-26 文本模式；v3 会报错，pf 仅 E2E；onnx/trt 需导出时加 `--nms`，metadata 会校验） |
| `--conf` / `--iou` / `--max-det` | 0.001 / 0.7 / 300 | 与官方 val 口径一致 |
| `--imgsz` | metadata > 640 | onnx 以图内 shape 为准（冲突报错） |
| `--limit` | 0（全量） | 只评前 N 张（GT 同步限制到该子集） |
| `--eval-style` | `auto` | `auto`（读描述符 `eval_style`）/ `coco` / `lvis` |
| `--eval-max-dets` | LVIS 300 | 评估器 maxDets；`100` 用于对齐 ultralytics 当前 val 行为 |
| `--visual-prompt` | — | yoloe-26 视觉模式：单 refer 图 JSON（bbox/mask；仅 pt；可与 `--classes` 混合） |
| `--classes` | — | yoloe-26 pt 文本 prompt 子集；与 `--visual-prompt` 组合成混合 prompt |
| `--text-encoder` | 权重同目录 | MobileCLIP2-B `.safetensors`；run 目录的微调产物用它指定 |
| `--refer-data` | — | yoloe-26 视觉模式：参考集描述符，逐类 GT 框提取 vpe |
| `--refer-split` / `--refer-limit` | train / 0 | 参考集角色 / 只取前 N 张（0=全量） |
| `--summary-only` | off | 只打印 all 行 |
| `--single-cls` | off | 单类评估：GT 类别映射到 0（pose 不适用，CLI 会报错） |
| `--device` | auto | pt 引擎设备 |

## 2. 指标口径与评估协议

| 口径 | 实现 | 用途 | 近似点 |
|---|---|---|---|
| **正式 · COCO 协议**（默认） | pycocotools | COCO 对外数字、验收 | — |
| **正式 · LVIS 协议** | faster_coco_eval（`lvis_style=True`） | LVIS / YOLOE 开放词汇对比 | — |
| 训练中 | 纯 numpy FastMetrics / FastPoseMetrics | 每轮趋势 + 选 `best.safetensors` | 忽略 crowd、maxDets=100（pose 20）、无面积分组；贪婪匹配 + 101 点插值 |

FastMetrics 与正式口径贴合良好，但**跨 run 比较/对外引用一律用正式口径**。

### 2.1 LVIS 协议（`eval_style: lvis`）

LVIS 的标注不是穷举的，评估必须处理两个图级字段，pycocotools 不认识它们：

- `neg_category_ids`：该图**确认不存在**的类别——出现预测记 FP；
- `not_exhaustive_category_ids`：该图该类标注**不完整**——该类预测忽略；
- 既不在 GT、也不在 neg/not_exhaustive 列表的 `(图, 类)` 预测直接丢弃。

因此 LVIS 数据集描述符使用 `eval_style: lvis`，评估器切到 faster_coco_eval 的
`lvis_style=True`，官方 maxDets=300，并额外输出 `APr / APc / APf`。**直接用 pycocotools
评 LVIS 得到的不是 LVIS 指标**（实测同一份预测可差数倍），不能和官方数字对比。

- `--eval-max-dets 100`：ultralytics 当前 val 只设置 `lvis_style`，没有改 faster_coco_eval
  的默认 `maxDets=[1,10,100]`；要与其代码逐字对齐时用这个值。产品默认仍是官方 300。
- LVIS minival = COCO val2017 的 5,000 张图；把 minival 注解 + 同一批图的描述符放到
  `config/datasets/local/lvis-minival.yaml`（结构同 `config/datasets/lvis.yaml`），
  `--data lvis-minival` 即可使用。
- `lvis_style` 要求 categories 带 `frequency`（r/c/f），缺失会在构造评估器时报错。

## 3. 输出解读

```text
      Class           Images Instances         P         R     mAP50  mAP50-95    AR@100
      all               5000     36781    0.6220    0.4594    0.4863    0.3377    0.5561
      person            2693     11004    0.7892    0.6300    0.7221    0.4776    0.6174
      Size buckets   mAP_small ... · mAP_medium ... · mAP_large ...
      Thresholds     mAP50 ... · mAP75 ... · mAP50-95 ...
Speed: 27.0 ms/image = load 2.5 + preprocess 1.8 + inference 7.5 + postprocess 14.5 + other 0.7
       model-only inference 7.5 ms/image (133 img/s) at shape (1, 3, 640, 640)
```

- per-class 表只列评估范围内有 GT 的类别；评估器的 `-1` 无效组合在聚合时剔除。
- LVIS 协议下最后一列是 `AR@300`（或 `--eval-max-dets` 指定的值），并多一行
  `LVIS freq  APr ... · APc ... · APf ...`。
- `Speed` 是评估管线分段耗时（含加载与后处理）；`model-only` 是纯推理。
- 结果落在 `runs/val/valN/`（含 `run.log`）；`results.json` 是 COCO result 列表，可直接交给
  faster_coco_eval 重新打分。

## 4. E2E vs NMS

- **E2E（默认）**：o2o 头、无 NMS，输出 `(B,300,6)`；官方 40.1 走这条。
- **NMS**：o2m 头 + 类感知 NMS，通常高 ~0.9 点、略慢；`--agnostic-nms` 改为跨类抑制。
  yolo26 系与 **yoloe-26 文本模式**都支持；yoloe pf（lrpc）仍只有 E2E。
- **单类模型**：`--single-cls` 把 GT 类别映射到 0（模型 nc 必须为 1）。
- YOLOv3-tiny 只有“解码 + obj×cls + NMS”一条路径，`--nms` / `--raw` 直接报错。

## 5. 分割（yolo26-seg / flash-yolo-seg）

分割权重自动走双表：同一份结果分别按 **框 IoU（Box）** 与 **掩码 IoU（Mask）** 评估。

- 掩码由 `utils/postprocess.process_mask` 解码到原图分辨率；评估器现编码 RLE 喂给
  pycocotools（`iou_type="segm"`）。
- `runs/val/valN/results.json` 写 Mask 组（含 RLE），可直接当 COCO 结果提交。
- 掩码解码与官方 val 管线逐像素一致（`tests/test_seg_inference.py` 对拍）。
- 数据集支持 COCO json 与 YOLO-seg txt；YOLO 无 crowd，GT 会少 crowd 实例。

### 5.1 Pose（yolo26-pose）

- `--data coco-pose`：COCO `person_keypoints_*.json`；评估器 `iou_type="keypoints"`，
  pycocotools OKS，`maxDets=20`。**val 只评估 2,346 张有有效关键点的图**（不是 5,000 张）；
  `scripts/eval.py` 自动按 `num_keypoints > 0` 过滤。
- 表头是 `Class Images Instances AP AP50 AP75 AP(M) AP(L) AR`；没有 bbox 表的 P/R 与 small
  bucket。`results.json` 写 COCO keypoints 结果（`keypoints` + `score`）。
- `--nms` 走 pose 的 o2m 分支；`--single-cls` 对 pose 不适用。
- 训练中 FastPoseMetrics 与 pycocotools 的 pose AP 可差约 0.1（10 轮微调实测 0.4367 vs
  0.5476）；训练日志只用于趋势，正式数字以本脚本为准。
- 官方 `yolo26n-pose` 参考（同一 2,346 图）：本仓库 E2E **0.5687 / 0.8250**，
  NMS 0.5670 / 0.8280；ultralytics E2E 0.572 / 0.833、NMS 0.570 / 0.831。完整口径见
  [../models/yolo26-pose.md](../models/yolo26-pose.md)。
- 训练中验证同时算 Box + Mask（低分辨率近似），best fitness = box + mask mAP50-95；
  正式数字仍以本脚本为准。

**官方 yolo26n-seg @640 验收**：Box 0.3967 / 0.5548，Mask 0.3367 / 0.5276（mAP50-95 / mAP50）。
官方训练期 validator 记录 box 0.3975 / mask 0.3386，差 ~0.002 属指标实现噪声。

## 6. YOLOE（开放词汇）

- **prompt-free（`-pf` 权重）**：头内烘焙的 4,585 名词汇映射到数据集类别——精确名优先、
  `vocab_synonyms` 回退（COCO 描述符内置 11 条）、缺失类别直接报错；数据集类别之外的预测
  被丢弃（`CocoEvaluator` 契约不变）。
- **文本 prompt（`-seg` 权重）**：数据集自身的类别名即 prompt（`set_classes`）；`--data lvis`
  对 1,203 个 LVIS 类别评估（`config/datasets/lvis.yaml`，取每类名第一个同义词——官方
  LVIS prompt 口径）。官方 LVIS minival 对比用 `--data lvis-minival`，走 faster_coco_eval
  `lvis_style`（见 §2.1）。pt 路径默认取模型权重同目录的 `mobileclip2_b.safetensors`
  （run 目录的微调产物用 `--text-encoder` 指定）；`onnx` / `trt` 消费类锁定图
  （类别名来自部署 metadata，按名字映射到数据集类别；锁定类别
  必须覆盖数据集类别，子集导出请用对应子集的描述符）。
- **视觉 prompt（`-seg` 权重）**：`--visual-prompt`（单 refer 图 JSON，bbox/mask）或
  `--refer-data`（参考集逐类 exemplar；`--refer-split` 默认 train，`--refer-limit` 限流）。
  参考类别按名字对齐评估集，缺失类零嵌入（AP 0）；仅 pt 引擎。`--classes` 可指定文本类
  子集，与 `--visual-prompt` 组合成混合 prompt（文本类在前）；类名不能重复。类锁定导出图
  按文本类锁定路径消费。
- 三种模式都像其他分割架构一样自动出 Box + Mask 双表。
- 模型细节：[../models/yoloe.md](../models/yoloe.md)。

## 7. 参考基准（COCO val2017）

| 权重 | 输入 | E2E mAP50-95 / mAP50 | NMS mAP50-95 / mAP50 |
|---|---|---|---|
| YOLO26n 官方 | 640 | 40.27 / 55.80 | 40.89 / 56.87 |
| YOLO26n 从零 300 ep | 640 | 37.82 / 53.24 | 38.65 / 54.46 |
| Flash-YOLO 从零 300 ep | 640 | 35.14 / 50.15 | 36.09 / 51.56 |
| YOLOv3-tiny 官方 | 640 | —（单路径） | 17.16 / 35.90 |

完整权重清单、下载链接与延迟见各模型页（[../models/README.md](../models/README.md)）。

## 8. 坑

- **`--limit` 必须同时限制 GT**：评估器按"本次遍历的全部 image_id"设置 `params.imgIds`
  （含零检出图）；自己改指标代码时注意不要退回到"只用有预测的图"——那会把漏检排除掉。
- **空结果**：`loadRes([])` 会 IndexError；评估器对空结果短路返回全 0。
- **LVIS 的 `frequency`**：`lvis_style=True` 要求每个 category 带 `frequency`（r/c/f）；
  普通 COCO json 用 `eval_style: lvis` 会在构造评估器时报错。
- **LVIS maxDets 默认值**：faster_coco_eval 的 `Params.maxDets` 默认 `[1,10,100]`；
  LVIS 官方是 300，本工程显式设置为 300。与 ultralytics 对比时用 `--eval-max-dets 100`。
- **`--eval-max-dets` 只对 LVIS 生效**：COCO 协议的 maxDets 固定 100（pycocotools 口径），
  传入会直接报错。
- **crowd**：COCO 侧保留（官方口径）；YOLO 侧无 crowd，val 差异只来自这一项。
- **segm 缺掩码直接报错**：`iou_type="segm"` 收到无 `masks` 的 Detections 会抛 ValueError，
  不静默退化成框评估。
- pt 的 `--imgsz` 覆盖 metadata 时会重新 letterbox；onnx/trt 输入尺寸由导出图决定。
- `eval` 默认 conf 0.001，`infer` 默认 0.25，两者不要混用。

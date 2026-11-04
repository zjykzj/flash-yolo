# 评估（Eval）

> 入口：`python scripts/eval.py --weights <w> --data <name|.yaml>`。
> **对外数字只有一个来源就是本脚本**（pycocotools 管线）；训练中打印的 FastMetrics 只用于
> 趋势与选 best。

## 1. 用法

```bash
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco                # E2E
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms          # o2m + NMS
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data coco         # 换运行时
python scripts/eval.py --weights w.safetensors --data coco --imgsz 416                  # 指定输入
python scripts/eval.py --weights w.safetensors --data coco --split val --limit 100      # 子集
python scripts/eval.py --weights weights/yolo26n-seg.safetensors --data coco            # Box + Mask
python scripts/eval.py --weights weights/yoloe-26n-seg-pf.safetensors --data coco       # pf 词汇映射
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis          # 文本 prompt（1203 类）
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--data` | — | 描述符（见 [../datasets/README.md](../datasets/README.md)）；yoloe 文本模式用其类别名做 prompt |
| `--split` | `val` | 角色键（train/val/test） |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--model` / `--scale` | 从权重推断 | 推不出时显式给 |
| `--nms` | off | o2m + NMS（yolo26 专属；v3 会报错） |
| `--conf` / `--iou` / `--max-det` | 0.001 / 0.7 / 300 | 与官方 val 口径一致 |
| `--imgsz` | metadata > 640 | onnx 以图内 shape 为准（冲突报错） |
| `--limit` | 0（全量） | 只评前 N 张（GT 同步限制到该子集） |
| `--summary-only` | off | 只打印 all 行 |
| `--device` | auto | pt 引擎设备 |

## 2. 两套指标口径

| 口径 | 实现 | 用途 | 近似点 |
|---|---|---|---|
| **正式**（本脚本） | pycocotools | 对外数字、验收 | — |
| 训练中 | 纯 numpy FastMetrics | 每轮趋势 + 选 `best.safetensors` | 忽略 crowd、maxDets=100、无面积分组；贪婪匹配 + 101 点插值 |

FastMetrics 与正式口径贴合良好，但**跨 run 比较/对外引用一律用正式口径**。

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

- per-class 表只列评估范围内有 GT 的类别；pycocotools 的 `-1` 无效组合在聚合时剔除。
- `Speed` 是评估管线分段耗时（含加载与后处理）；`model-only` 是纯推理。
- 结果落在 `runs/val/valN/`（含 `run.log`）。

## 4. E2E vs NMS

- **E2E（默认）**：o2o 头、无 NMS，输出 `(B,300,6)`；官方 40.1 走这条。
- **NMS**：o2m 头 + 类感知 NMS，通常高 ~0.9 点、略慢；`--agnostic-nms` 改为跨类抑制。
- **单类模型**：`--single-cls` 把 GT 类别映射到 0（模型 nc 必须为 1）。
- YOLOv3-tiny 只有“解码 + obj×cls + NMS”一条路径，`--nms` / `--raw` 直接报错。

## 5. 分割（yolo26-seg / flash-yolo-seg）

分割权重自动走双表：同一份结果分别按 **框 IoU（Box）** 与 **掩码 IoU（Mask）** 评估。

- 掩码由 `utils/postprocess.process_mask` 解码到原图分辨率；评估器现编码 RLE 喂给
  pycocotools（`iou_type="segm"`）。
- `runs/val/valN/results.json` 写 Mask 组（含 RLE），可直接当 COCO 结果提交。
- 掩码解码与官方 val 管线逐像素一致（`tests/test_seg_inference.py` 对拍）。
- 数据集支持 COCO json 与 YOLO-seg txt；YOLO 无 crowd，GT 会少 crowd 实例。
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
  LVIS prompt 口径）。目前仅 `--engine pt`，且 `mobileclip2_b.safetensors` 需与模型权重
  同目录。
- 两种模式都像其他分割架构一样自动出 Box + Mask 双表。
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

- **`--limit` 必须同时限制 GT**：评估器已按 `params.imgIds` 对齐子集；自己改指标代码时注意。
- **空结果**：`loadRes([])` 会 IndexError；评估器对空结果短路返回全 0。
- **crowd**：COCO 侧保留（官方口径）；YOLO 侧无 crowd，val 差异只来自这一项。
- **segm 缺掩码直接报错**：`iou_type="segm"` 收到无 `masks` 的 Detections 会抛 ValueError，
  不静默退化成框评估。
- pt 的 `--imgsz` 覆盖 metadata 时会重新 letterbox；onnx/trt 输入尺寸由导出图决定。
- `eval` 默认 conf 0.001，`infer` 默认 0.25，两者不要混用。

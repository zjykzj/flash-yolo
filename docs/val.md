# 评估（Val）

> 入口：`python scripts/eval.py --weights <w> --data <name|.yaml>`。
> **对外数字只有一个来源就是本脚本**（pycocotools 管线）；训练中打印的 FastMetrics 只用于
> 趋势与选 best。

## 1. 用法

```bash
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco                # E2E -> 40.27
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms          # o2m+NMS -> 40.89
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data coco         # 换运行时
python scripts/eval.py --weights w.safetensors --data coco --imgsz 416                  # 指定输入尺寸
python scripts/eval.py --weights w.safetensors --data coco --split val --limit 100      # 冒烟子集
python scripts/eval.py --weights weights/yolo26n-seg.safetensors --data coco            # 分割（Box + Mask 双表）
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` 或 `.onnx` |
| `--data` | — | 描述符（见 `dataset.md`） |
| `--split` | `val` | 角色键（描述符里的 train/val/…） |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--model` / `--scale` | 从权重文件名推断 | 推不出时显式给（`--model yolov3-tiny` 等） |
| `--nms` | off | 走 o2m + NMS 路径（yolo26 专属；v3 会直接报错） |
| `--conf` / `--iou` / `--max-det` | 0.001 / 0.7 / 300 | 与官方 val 口径一致 |
| `--imgsz` | 权重 metadata > 640 | 优先级：CLI > metadata > 640；**onnx 以图内 shape 为准**（显式冲突报错不静默） |
| `--limit` | 0（全量） | 只评前 N 张（评估器会把 GT 限制到同一子集） |
| `--summary-only` | off | 只打印 all 行 |
| `--device` | auto | pt 引擎的设备 |

## 2. 两套指标口径

| 口径 | 实现 | 用途 | 近似点 |
|---|---|---|---|
| **正式**（本脚本） | pycocotools | 对外数字、验收 | — |
| 训练中（`train/metrics.py`） | 纯 numpy FastMetrics | 每轮趋势 + 选 `best.safetensors` | 忽略 crowd、maxDets=100、无面积分组；贪婪匹配 + 101 点插值 |

FastMetrics 与正式口径实测贴合良好（如 300 轮 v3 run：训练中 0.1379 → 正式 14.23），
但**跨 run 比较/对外引用一律用正式口径**。

## 3. 输出解读

```
      Class           Images Instances         P         R     mAP50  mAP50-95    AR@100
      all               5000     36781    0.6220    0.4594    0.4863    0.3377    0.5561
      person            2693     11004    0.7892    0.6300    0.7221    0.4776    0.6174
      Size buckets   mAP_small ... · mAP_medium ... · mAP_large ...   (COCO area <32² / 32²-96² / >96²)
      Thresholds     mAP50 ... · mAP75 ... · mAP50-95 ...   (IoU 0.50 / 0.75 / 0.50:0.95)
Speed: 27.0 ms/image = load 2.5 + preprocess 1.8 + inference 7.5 + postprocess 14.5 + other 0.7 → 37.0 img/s
       model-only inference 7.5 ms/image (133 img/s) at shape (1, 3, 640, 640)
```

- per-class 表**只列评估范围内有 GT 的类别**（该图集没有的类不出现）；被 pycocotools 标 `-1`
  的无效组合在聚合时剔除（整类都无效时该项记 0.0000）；
- `Speed` 行是**评估管线**的分段耗时（含加载与 pycocotools 后处理）；模型纯推理看
  `model-only` 那行；
- 结果同时落在 `runs/val/valN/`（含 `run.log`）。

## 4. E2E vs NMS

- **E2E（默认）**：o2o 头、无 NMS，输出 (B,300,6)。官方 40.1 就是这条路径。
- **NMS**：o2m 头 + 类感知 NMS，通常高 ~0.9 点、略慢（官方 40.9）。
- yolov3-tiny 只有一条"解码 + obj×cls + NMS"路径，`--nms` / `--raw` 对它直接报错。

## 5. 分割（yolo26-seg / flash-yolo-seg）

分割权重（`--weights weights/yolo26n-seg.safetensors`）自动走双表口径：同一份结果分别按
**框 IoU（Box）** 与 **掩码 IoU（Mask）** 评估，与官方发布的 Box/Mask 双表一致。

```
      Box metrics
      Class           Images Instances         P         R     mAP50  mAP50-95    AR@100
      all               5000     36781    ...
      Mask metrics
      all               5000     36781    ...
      Size buckets (Box)  ... / (Mask) ...
      Thresholds (Box)    ... / (Mask) ...
```

- 掩码由 `utils/postprocess.process_mask` 解码到**原图分辨率**（与 GT 掩码同尺度），评估器现编码
  RLE 喂给 pycocotools（`iou_type="segm"`）；
- `runs/val/valN/results.json` 写 Mask 组（含 `segmentation` RLE 字符串），可直接当 COCO 结果提交；
- 掩码解码与官方 val 管线**逐像素一致**（`tests/test_seg_inference.py` 对拍）；
- 数据集支持 COCO json 与 YOLO-seg txt：YOLO 的 GT dict 由多边形即时生成
  （`segmentation` 为绝对像素多边形、`area` 为多边形面积）；YOLO 无 crowd 概念，
  Mask/Box GT 会少 crowd 实例（与检测侧同一条口径差异）；
- **官方分割权重同管线对照**：`scripts/compare_official.py --weights yolo26n-seg.pt --data coco`
  解包官方双输出（检测+系数、原型），同时输出 Box/Mask 两表；同子集与 `scripts/eval.py`
  逐项一致（dev-time 工具，需要 ultralytics）。
- **训练中验证**（`train/validator.py` + `FastMetrics`）：同一轮同时算 Box 与 Mask，Mask 在
  `imgsz/mask_ratio` 网格上做快速近似（低分辨率 `process_mask_lowres`，不落盘掩码），best 选取
  用官方 seg fitness = box mAP@[.5:.95] + mask mAP@[.5:.95]；正式数字仍以本节 pycocotools 为准。
  每张图的 GT 多边形只栅格化一次（`seg_val_multi`，loss 网格与指标网格共用）：默认
  `mask_ratio=4` 实测 1.79 -> 0.98 ms/图（全量 5000 图 8.9 -> 4.9 s），输出逐像素不变。
  训练列 `sem` 是 semseg 辅助损失（BCE-Dice）；验证模式不返回 semantic，所以 `val_sem` 恒 0
  （官方同口径）；
- 开销：conf 0.001 下每图最多 300 检出，掩码解码是此后处理的大头——`process_mask` 按 8 张
  一批做（`_MASK_CHUNK`），让中间张量落进缓存：实测 p50 128ms -> 59ms/图，逐位不变。
  RLE 编码交给 pycocotools 的 C 实现（自写向量化版反而慢，见下）。

**验收（官方 yolo26n-seg @640，COCO val2017 全量，本管线 pycocotools 口径）**：

| 口径 | mAP@50 | mAP@[.5:.95] |
|---|---|---|
| Box | 0.5548 | 0.3967 |
| Mask | 0.5276 | 0.3367 |

官方训练期 validator 记录为 box 0.3975 / mask 0.3386——两者差 ~0.002，属指标实现噪声
（同检测侧的 40.1 vs 40.27）。全量耗时 15.1 min（181.6 ms/图：推理 12.9 + 后处理 111.1，GPU；
后处理分批后见上）。

**RLE 编码为什么不用自研向量化版**：实测 pycocotools 逐张 C 编码 33ms（300 掩码），而
numpy 整批向量化（trans + nonzero + lexsort）反而 85ms——列主序展平与变化点扫描都是一次
全像素遍历，numpy 的多次大临时量抵消了向量化收益；逐张 numpy 提取也要 20ms 且仍要接
varint 编码。**瓶颈在掩码解码，不在编码**，故保留 pycocotools。

## 6. 参考基准（COCO val2017，本管线）

| 权重 | 输入 | E2E | NMS |
|---|---|---|---|
| YOLO26n 官方权重 | 640 | 40.27 / 55.80 | 40.89 / 56.87 |
| YOLO26n 从零 100 轮 | 640 | 36.22 / 51.32 | 37.15 / 52.67 |
| flash-yolo 从零 100 轮 | 640 | 33.77 / 48.63 | 34.72 / 50.16 |
| YOLOv3-tiny 官方权重 | 640 | —（单路径） | 17.16 / 35.90 |

（mAP@[.5:.95] / mAP@50。历史与实验细节见 `yolo26-runs.md` / `flash-yolo-spec.md` /
`yolov3-tiny-runs.md`。）

## 7. 坑

- **`--limit` 必须同时限制 GT**：评估器已按 `params.imgIds` 对齐子集；自己改指标代码时注意，
  否则未预测的图会把 mAP 稀释。
- **空结果**：`loadRes([])` 会 IndexError——评估器对空结果短路返回全 0，属预期行为。
- **crowd**：COCO 侧保留（官方口径）；YOLO 侧无 crowd，两者 val 数字差异只来自这一项。
- **segm 缺掩码直接报错**：`iou_type="segm"` 的评估器收到无 `masks` 的 Detections 会抛
  ValueError——不静默退化成框评估（否则会得到一份"看着像 mask mAP"的框数字）。
- pt 的 `--imgsz` 覆盖 metadata 时会**重新 letterbox**；onnx/trt 输入尺寸由导出图决定。

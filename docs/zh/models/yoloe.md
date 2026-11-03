# YOLOE-26（开放词汇实例分割）

> 开放词汇分割：类别不在训练期固定——推理时按文本 / 视觉 prompt 或内置 **4,585 名词汇**作答。
> v1.1.0 支持 **prompt-free（-pf）** 推理 / 评估 / 导出；文本与视觉 prompt 模式在后续版本提供。
> 结构定义：`config/models/yoloe-26-seg.yaml`；骨架 = YOLO26（backbone/neck 与
> [yolo26-seg.md](yolo26-seg.md) 逐行相同，只换最后一行头）。

## 1. 口径声明

- 官方 26 家族只发布 **seg 权重**：`yoloe-26*-seg.pt`（文本模式）/ `yoloe-26*-seg-pf.pt`
  （prompt-free，本页支持范围）。
- 官方 LVIS minival 数字（论文口径、**检测配置**、640 输入）：YOLOE-26n-pf **16.6 E2E /
  17.7 Non-E2E** mAP50-95；文本 prompt 24.7 / 视觉 prompt 21.9（Non-E2E）。论文的参数量/FLOPs
  是检测配置——发布的 seg checkpoint 额外带 mask 分支 / SAVPE / 文本投影，实际更大
  （n-pf 7.84M），官方文档已注明此差异；FLOPs 不含区域-文本相似度，真实成本随 prompt 集增大。
- 本仓库的复现口径 = 同权重、同前后处理重跑：与官方 fp32 前向**逐位一致**（§4）；COCO 数字
  为管线冒烟口径（20 图），不代表精度结论。
- 文中「官方」指 Ultralytics 发布的 YOLOE-26 权重/文档，本仓库不分发。

## 2. 结构（prompt-free 布局，n 档）

yaml 24 个模块（`model.model` 0–23），346 层模块树，**7,837,694 参数 / 16.0 GFLOPs @640**
（s 档 18,419,972 参数 / 44.2 GFLOPs；比 yolo26n-seg 的 9.3 GFLOPs 高——vocab Linear 对
8400 锚点 × 4585 词汇的前向是 pf 的固有开销）：

| 段 | idx | 内容 |
|---|---|---|
| Backbone | 0–10 | 与 yolo26-seg 逐行相同（Conv → C3k2 → SPPF → C2PSA） |
| Neck | 11–22 | 与 yolo26-seg 逐行相同（P3/P4/P5 三尺度） |
| Head | 23 | YOLOESegment26（pf 布局）：截断 cv2/cv3 + lrpc 烘焙词汇 + cv5/proto |

### Head（YOLOESegment26，pf 布局）

- `cv2[i]` box 分支（Conv+Conv，无末层——拆到 `lrpc.loc`）；`cv3[i]` 嵌入分支（DWConv+Conv
  两级，无末层——拆到 `lrpc.pf`）；`cv5[i]` mask 系数与 `proto`（Proto26）与 yolo26-seg 同构。
- `lrpc[i]` = `vocab`（Linear(c3→4585)，烘焙词汇）+ `pf`（提案过滤，只取第 0 通道）+
  `loc`（框回归）。**P5（level 2）不参与提案过滤**（vocab 保持 Conv2d、全锚点通过——
  官方发布权重的结构事实）。
- 词汇 = **4,585 名（RAM++）**，随权重 metadata 走（`names`）；类别语义由词汇决定，
  与数据集无关。
- `one2one_*` 副本 = end2end 双头；`reprta` / `savpe` 在权重里但不参与 pf 前向（strict load
  对齐，同 yolo26-seg 的 semseg 先例）。
- 提案过滤：conf=0.001 逐图 sigmoid 阈值 + 批内并集掩码；**导出 = conf=0 静态口径**（官方
  同款：全锚点通过、无动态掩码）。E2E 输出 `(B, 1000, 6+nm)`——**max_det=1000** 是 YOLOE
  口径（yolo26 系为 300）。
- 档位差异（官方资产的结构事实）：n 档 pf 卷积 512 通道（复用 cv3 尾卷积）、s 档 1 通道
  （专用单通道）——行为一致（都只取第 0 通道），由 yaml 档位数据的 `pf_channels` 表达。

## 3. 用法

```bash
# 下载官方 .pt 后转换（一次性）：解包成纯 safetensors + 词汇/模式 metadata
python scripts/convert_weights.py --src yoloe-26n-seg-pf.pt --dst weights/yoloe-26n-seg-pf.safetensors

# 推理（类别名 = 词汇名，随权重；检测框 + 掩码同 yolo26-seg 管线）
python scripts/infer.py --weights weights/yoloe-26n-seg-pf.safetensors --image assets/traffic.jpg

# 评估（数据集类别名 -> 词汇映射：精确匹配优先、vocab_synonyms 回退、缺失报错）
python scripts/eval.py --weights weights/yoloe-26n-seg-pf.safetensors --data coco

# 导出（E2E 融合图 + 原型，双输出；conf=0 静态口径）
python scripts/export.py --weights weights/yoloe-26n-seg-pf.safetensors --out weights/yoloe-26n-seg-pf.onnx
```

- `--model/--scale` 可从文件名推（`yoloe-26n-seg-pf` → yoloe-26 / n）；`--nms` / `--raw`
  不适用（直接报错——pf 原始输出列数随提案过滤变化）。
- COCO 类名与 RAM++ 词汇的命名差异（如 `airplane` → `plane`、`tv` → `television`）经数据集
  描述符的 `vocab_synonyms` 字段解析（`config/datasets/coco.yaml` 已内置 11 条，含官方词汇
  自身的拼写 `dinning table`）；评估只保留数据集类别的预测，`CocoEvaluator` 契约不变。
- 官方权重下载（ultralytics assets）：`yoloe-26n-seg-pf.pt` 等见
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/`。

## 4. 验收（凭什么可信）

- **strict load**：官方 n/s 两档 -seg-pf 转换后全 key 匹配（928 键、零多余零缺失）；
  头构建 key 集与官方 ckpt 一致（pf / 文本两种布局）。
- **逐位一致**：与官方 fp32 前向（E2E 检测 + 原型）在 zeros/random 双输入下逐位一致
  （n/s 两档；E2E 输出 (1,1000,38) 与原型 (1,32,160,160) 均 torch.equal）。
- **导出对拍**：onnx 与 pt（同 conf=0 静态口径）在 conf>0.25 下有效检测完全一致——框/分数/
  系数 < 1e-3、零类别翻转（4585 类的近并列分数只出现在低分尾部，由 torch/ORT 后端决胜）。
- **COCO 冒烟**（20 图，管线验证口径）：Box 0.3485 / Mask 0.2865 mAP50-95。
- 回归闸门 `tests/test_yoloe.py`（结构 / 转换 metadata / strict load / 逐位对齐 / 词汇映射 /
  导出对拍）。

## 5. 已知边界

- 文本 / 视觉 prompt 前向未实现（文本布局权重可构建与加载，前向报错提示）。
- 导出图与运行时 pt 的提案过滤口径不同（官方同款：导出 conf=0 静态、运行时 conf=0.001）——
  少数低分检测可能不同；conf>0.25 的有效检测实测一致。
- 官方 yaml 注释的参数量（5,615,540）是陈旧值，不含 savpe/reprta；本页数字是 strict-load
  一致的实值（`parameters()` 口径）。

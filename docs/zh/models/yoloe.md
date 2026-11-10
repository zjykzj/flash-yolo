# YOLOE-26（开放词汇实例分割）

> 开放词汇分割：类别不在训练期固定——推理时按文本 prompt、视觉 prompt（refer 图 + bbox/mask）
> 或内置 **4,585 名词汇**作答。当前版本支持 **prompt-free（-pf）**、**文本 prompt**
> （自定义类别名，经 MobileCLIP2-B 文本编码器编码）与 **视觉 prompt**（refer 图示例经
> SAVPE 提取嵌入）三种模式的推理 / 评估 / 导出。
> 结构定义：`config/models/yoloe-26-seg.yaml`；骨架 = YOLO26（backbone/neck 与
> [yolo26-seg.md](yolo26-seg.md) 逐行相同，只换最后一行头）。

## 1. 口径声明

- 官方 26 家族只发布 **seg 权重**：`yoloe-26*-seg.pt`（文本/视觉模式）/
  `yoloe-26*-seg-pf.pt`（prompt-free）；本页覆盖三种 prompt 模式。
- 官方 LVIS minival 数字（论文口径、**检测配置**、640 输入）：YOLOE-26n-pf **16.6 E2E /
  17.7 Non-E2E** mAP50-95；文本 prompt 24.7 / 视觉 prompt 21.9（Non-E2E）。论文的参数量/FLOPs
  是检测配置——发布的 seg checkpoint 额外带 mask 分支 / SAVPE / 文本投影，实际更大
  （n-pf 7.84M），官方文档已注明此差异；FLOPs 不含区域-文本相似度，真实成本随 prompt 集增大。
- 文本模式需要 MobileCLIP2-B 文本编码器权重（`mobileclip2_b.safetensors`，由 Apple 的
  `mobileclip2_b.ts` 转换而来，随本仓库 Releases 提供）与模型权重同目录——只在
  `set_classes`（推理/评估）时加载，不进导出图（导出时嵌入已融合进分类卷积）。视觉模式
  不需要文本编码器：refer 图的 bbox/mask 在 P3 网格（1/8）上光栅化，SAVPE 从三尺度特征
  聚合出每类 vpe（L2 归一化）。
- 本仓库的复现口径 = 同权重、同前后处理重跑：与官方 fp32 前向**逐位一致**（§4）；20 图 COCO
  冒烟是管线检查，5,000 图全量跨实现数字见 §4。
- 文中「官方」指 Ultralytics 发布的 YOLOE-26 权重/文档，本仓库不分发。

## 2. 结构（n 档）

yaml 24 个模块（`model.model` 0–23），346 层模块树。**pf 布局** **7,837,694 参数 /
16.0 GFLOPs @640**（s 档 18,419,972 参数 / 44.2 GFLOPs；比 yolo26-seg 的 9.3 GFLOPs 高——
vocab Linear 对 8400 锚点 × 4585 词汇的前向是 pf 的固有开销）；**文本布局**
**5,615,540 参数 / 10.5 GFLOPs @640**（K=80 prompt 口径——区域-文本相似度随 prompt 数线性
缩放；官方 yaml 注释的 11.7 含 semseg 前向，官方文档的数字是检测配置）：

| 段 | idx | 内容 |
|---|---|---|
| Backbone | 0–10 | 与 yolo26-seg 逐行相同（Conv → C3k2 → SPPF → C2PSA） |
| Neck | 11–22 | 与 yolo26-seg 逐行相同（P3/P4/P5 三尺度） |
| Head | 23 | YOLOESegment26（pf 布局：截断 cv2/cv3 + lrpc 烘焙词汇；文本布局：cv2/cv3 完整尾卷积 + cv4 对比头）+ cv5/proto |

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
- 档位差异（官方资产的结构事实）：n 档 pf 卷积 512 通道（复用 cv3 尾卷积）、s/m/l/x 档
  1 通道（专用单通道）——行为一致（都只取第 0 通道），由 yaml 档位数据的 `pf_channels` 表达。

### Head（YOLOESegment26，文本布局）

- `cv2[i]` / `cv3[i]` 保留完整尾卷积（框回归 / 512-d 区域嵌入）；`cv4[i]` =
  BNContrastiveHead：区域嵌入过 BN（eps=0.001，官方 ckpt 实测值）、文本嵌入 L2 归一化后求
  余弦相似度，乘可学习 `exp(logit_scale)` + bias；`cv5`/`proto` 与 pf 布局同构。无 `lrpc`。
- 文本 prompt 流程（`set_classes`）：CLIP BPE tokenize（77 上下文）→ MobileCLIP2-B 编码 →
  `reprta`（SwiGLUFFN 残差）→ L2 归一化 → 缓存到 head。类别数 = prompt 数，随每次前向变。
- 视觉 prompt 流程：refer 图 bbox/mask 在 P3 网格（输入 1/8）光栅化 → `SAVPE`（三尺度
  特征 + 掩码注意力聚合）→ 每类一个 L2 归一化 vpe → 直接作为类别嵌入送入同一
  `BNContrastiveHead`（不再过 `reprta`）。同图同类多框 OR；类别数 = prompt 类数。
- 文本布局 **max_det=300**（官方文本 ckpt 带的是 Detect 类默认；pf ckpt 带的是 1000）。
- 文本编码器是外部件（63.4M 参数，不计入 FLOPs——官方同口径）；导出时融合进分类卷积，
  导出图即类锁定、单输入。

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

# ---- 文本模式（yoloe-26*-seg 权重；需要 mobileclip2_b.safetensors 与模型权重同目录）----

# 推理：自定义类别名 / 文本 prompt，经 MobileCLIP2-B 文本编码器编码
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors --image assets/traffic.jpg \
    --classes "person, bicycle, car, bus, truck"

# 评估：数据集自身的类别名即 prompt
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis            # LVIS 全量 val
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival    # 官方 minival 口径

# 导出：prompt 融合进分类卷积——单输入、类锁定 ONNX
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --out weights/yoloe-26n-3cls.onnx

# 导出 Non-E2E 部署图（o2m 密集输出 + 外部 NMS）
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --nms --out weights/yoloe-26n-3cls-nms.onnx

# TensorRT（pf / text 均适用；engine 绑定构建机 GPU + TRT 版本）
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --trt [--fp16]

# ---- 视觉模式（yoloe-26*-seg 权重；refer 图 + bbox/mask，JSON）----

# prompt.json: {"image": "refer.jpg",
#               "prompts": [{"class": "person", "bbox": [x1,y1,x2,y2]},
#                           {"class": "person", "mask": "person.png"}]}
# 每个示例 bbox / mask 二选一；mask 与 refer 图同尺寸（非 0 即前景），同类多示例 OR。

# 推理：refer 图/框提取 vpe，再跑目标图（pt 引擎）
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors \
    --visual-prompt prompt.json --image assets/traffic.jpg

# 评估：单 refer 图 JSON，或 refer 数据集（每类 exemplar）
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco \
    --visual-prompt prompt.json
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --refer-data lvis --refer-split train --refer-limit 1000

# 导出：vpe 融合进分类卷积，类锁定（可配 --nms / --trt）
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --visual-prompt prompt.json --out weights/yoloe-26n-visual.onnx

# ---- 文本 + 视觉混合（文本类在前、视觉类在后——官方 get_cls_pe 顺序）----
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "bicycle" --visual-prompt prompt.json --image assets/traffic.jpg
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco \
    --classes "bicycle" --visual-prompt prompt.json
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "bicycle" --visual-prompt prompt.json --out weights/yoloe-26n-mixed.onnx

# ---- PE 线性探测微调（文本模式；需要官方文本权重 + mobileclip2_b.safetensors）----
# 数据集类别名作文本 prompt：嵌入融合进 cv3 尾卷积后只训练这 6 个卷积；
# backbone / neck / BN / 掩码分支 / 原型全部冻结。--text-encoder 可覆盖编码器路径。
python scripts/train.py --data coco --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 32 --imgsz 640 --lr0 0.1

# LVIS（federated 掩码自动启用；跨图增强自动关闭；lr0 用 0.01）
python scripts/train.py --data lvis --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 16 --imgsz 640 --lr0 0.01

# 产物是类锁定权重（metadata: pe/fused/struct_nc/names）：eval / export / infer 直接消费
python scripts/eval.py --weights runs/train/train-yoloe-pe-coco/weights/best.safetensors --data coco
python scripts/export.py --weights runs/train/train-yoloe-pe-coco/weights/best.safetensors \
    --out weights/yoloe-26n-coco-pe.onnx

# ---- 全量微调（文本模式；从官方文本权重出发，全部参数训练）----
# 数据集类名的原始 CLIP 嵌入缓存一次（reprta 留在图内随训练更新），mask/proto 也训练；
# 学习率比 PE 低两个数量级。产物未 fused，prompt 名随 metadata 走。
python scripts/train.py --data coco --model yoloe-26 --scale n --full-finetune \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 32 --imgsz 640 --lr0 0.001

# LVIS（federated 掩码自动启用；跨图增强自动关闭）
python scripts/train.py --data lvis --model yoloe-26 --scale n --full-finetune \
    --weights weights/yoloe-26n-seg.safetensors \
    --epochs 10 --batch 16 --imgsz 640 --lr0 0.001

# eval 用同一数据集描述符；infer/export 未给 --classes 时自动用训练 names。
# run 目录的产物旁边没有编码器：显式 --text-encoder（或把它拷到权重同目录）。
python scripts/eval.py --weights runs/train/train-yoloe-ft-coco/weights/best.safetensors \
    --text-encoder weights/mobileclip2_b.safetensors --data coco
python scripts/infer.py --weights runs/train/train-yoloe-ft-coco/weights/best.safetensors \
    --text-encoder weights/mobileclip2_b.safetensors --image assets/traffic.jpg
python scripts/export.py --weights runs/train/train-yoloe-ft-coco/weights/best.safetensors \
    --text-encoder weights/mobileclip2_b.safetensors \
    --out weights/yoloe-26n-coco-ft.onnx
```

- PE 训练只接受文本模式权重（prompt-free 拒绝）；`--model yoloe-26` 不带 `--pe` /
  `--full-finetune` 也会报错。结构 nc 取权重 metadata（官方 80），prompt 数 = 数据集类数，
  因此 LVIS 1203 类不会改变 cv3 主体结构。`--freeze` 不能与 `--pe` 同用；融合后的产物
  类锁定，eval/infer/export 再传 `--classes` / `--visual-prompt` 会被拒绝。PE 的
  `--resume` 必须同时带 `--weights`（用原始文本权重重建融合结构，再由 resume.pt 覆盖状态）。
  学习率建议：COCO 80 类 lr0=0.1；LVIS 1203 类 lr0=0.01（federated 分类掩码会自动启用，
  mosaic 等跨图增强自动关闭）。
- 全量微调（`--full-finetune`）与 PE 的边界：不 fuse、不冻结，文本侧 reprta 与视觉侧一起
  训练，损失走分割三项（含 mask/proto），结构 nc 仍取权重 metadata（官方 80），活 nc =
  数据集类数。学习率要比 PE 低两个数量级（COCO 探针 0.0005–0.001 与零样本持平，
  ≥0.005 明显漂移；LVIS 同量级）——PE 的 0.1/0.01 会让预训练特征漂移，1 epoch 探针即
  明显退化。`--pe` 与 `--full-finetune` 互斥，两者都要求文本权重；`--full-finetune`
  允许 `--freeze N` 冻结浅层。`--resume` 时两者都要重复对应 flag 与 `--weights`
  （与 PE 同口径）。PE 产物类锁定；
  全量微调产物未 fused，eval 用数据集 names 做 prompt（名字不一致报错），infer/export 在
  未显式给 `--classes` 时自动用 metadata 的训练 names。
- 选择建议：只换词汇、数据量小（几百到几千张）、要分钟级产出 → PE；域偏移明显或数据量
  上万、追求精度 → 全量微调。全量微调会重排开放词汇的特征空间，产物只适合训练时的词汇，
  不要拿它当新的零样本模型（PE 的漂移小一些，但同样建议只用于训练词汇）。
- `--model/--scale` 可从文件名推（`yoloe-26n-seg-pf` → yoloe-26 / n）；`--raw` 不适用
  （直接报错——pf 原始输出列数随提案过滤变化）；`--nms`（o2m + 类感知 NMS）支持 **text 模式**，
  pf 仍只有 E2E。官方文本/视觉 prompt 表的 24.7 就是 Non-E2E 口径。
- COCO 类名与 RAM++ 词汇的命名差异（如 `airplane` → `plane`、`tv` → `television`）经数据集
  描述符的 `vocab_synonyms` 字段解析（`config/datasets/coco.yaml` 已内置 11 条，含官方词汇
  自身的拼写 `dinning table`）；评估只保留数据集类别的预测，`CocoEvaluator` 契约不变。
- 文本模式说明：`--classes` 逗号分隔，每个名字即一个 prompt/类别；LVIS prompt 取每类名第一
  个同义词（`name.split("/", 1)[0]`，官方口径——`config/datasets/lvis.yaml` 已按此存储）；
  类锁定 ONNX 拒收新 prompt（`is_fused` 守卫）。类锁定产物自带部署 metadata
  （arch/scale/names/end2end），导出后可直接 `--engine onnx|trt`，无需再传
  `--model/--scale/--classes`；TRT 旁车为 `<engine>.json`。
- 视觉模式说明：JSON 的类别顺序 = 首次出现顺序（vpe 行序一致），bbox 为 refer 图原始像素，
  mask 与 refer 图同尺寸；相对路径按 JSON 所在目录解析；同类多个示例 OR 成一张掩码。
  `--classes` 可与 `--visual-prompt` 组合成混合 prompt（文本类在前、视觉类在后；同名报错）。
  推理/评估只支持 pt 引擎（SAVPE 在图内），导出后的类锁定图可走 onnx/trt。`--refer-data`
  用参考集 GT 框逐类提取 vpe，缺失类别零嵌入（AP 0）；官方公开 API 无混合入口，本工程按
  `get_cls_pe` 的拼接语义扩展。
- LVIS 评估走 **faster_coco_eval `lvis_style`**（`eval_style: lvis` 描述符字段）：处理
  `neg_category_ids` / `not_exhaustive_category_ids`、maxDets=300，并输出 APr/APc/APf。
  用 pycocotools 评 LVIS 得到的不是 LVIS 指标；与 ultralytics 代码逐字对齐时加
  `--eval-max-dets 100`（它没有覆盖 faster_coco_eval 的默认 maxDets）。
- 官方权重下载（ultralytics assets）：`yoloe-26n-seg-pf.pt` 等见
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/`。

## 4. 验收（凭什么可信）

- **strict load**：官方 -seg-pf 全五档转换后全 key 匹配（零多余零缺失）；头构建 key 集与
  官方 ckpt 一致（pf / 文本两种布局）。
- **逐位一致（pf）**：与官方 fp32 前向（E2E 检测 + 原型）在 zeros/random 双输入下逐位一致
  （全五档；E2E 输出 (1,1000,38) 与原型 (1,32,160,160) 均 torch.equal）。
- **逐位一致（文本）**：完整 prompt 链——CLIP BPE tokenize → MobileCLIP2-B 编码 → reprta+L2
  ——与 head 前向（E2E 检测 + 原型）与官方管线逐位一致，全五档；文本编码器本身与官方
  `mobileclip2_b.ts` 逐位一致（L=77，官方输入契约）。tokenizer 与官方 CLIP fork 逐位一致，
  除 ftfy 乱码修复（§5）。
- **导出对拍（pf）**：onnx 与 pt（同 conf=0 静态口径）在 conf>0.25 下有效检测完全一致——
  框/分数/系数 < 1e-3、零类别翻转（4585 类的近并列分数只出现在低分尾部，由 torch/ORT 后端决胜）。
- **导出对拍（文本）**：fuse 后的 pt 与未 fuse 差 1.8e-6、onnx 差 6e-5——conf>0.25 的检出
  完全一致、零类别翻转。
- **Non-E2E 导出对拍（文本）**：`--nms` 导出的 o2m 图与 pt 同路径在 conf>0.25 下检出一致
  （框/分数 <1e-3、零类别翻转）；coco-tiny 2 图评估 Box/Mask 四位一致。
- **视觉 prompt**：同一 refer 图 + bbox 的 vpe 与官方环境（`.pt` + `YOLOEVPSegPredictor`）
  逐位一致（maxdiff 0.0）；fuse 导出后 pt vs onnx 2/2 检出、框差 6.1e-5、分数差 1.1e-6、
  零类别翻转；refer 数据集逐类聚合与单图提取逐位一致。同图 top-2 与官方 Non-E2E 口径框差
  <0.6px、分数差 ≤0.011（低分尾部候选数受官方 predictor 的 NMS 默认值影响）。
- **mask prompt**：mask 的 1/8 光栅化与官方 `LoadVisualPrompt(masks=...)` `torch.equal`；
  类锁定导出 vs pt 检出一致。
- **混合 prompt**：缓存嵌入 = `cat([tpe, vpe])`（文本在前，官方 `get_cls_pe` 顺序）；
  同名类别明确报错；类锁定导出 vs pt 检出一致。
- **视觉 val（官方默认协议）**：官方 `load_vp=True` 用**评估集自身**当 refer。官方 8.4.175
  在 LVIS minival 的 COCO 表 Box 0.187 / Mask 0.141；用同一 LVIS 口径重打分官方预测：
  Box 0.3557 / Mask 0.2826。本工程同协议（同一 YOLO 参考集、`--nms`、maxDets=100）：
  Box **0.3534** / Mask **0.2805**（差 0.0023 / 0.0021），APr/APc/APf 差 ≤0.008。
  参考集为同一批 minival 图的 YOLO 格式 GT（本地 refer 描述符，同图同标注）。
  COCO 表的 Mask 差 0.015 归因于官方表后端（其 faster-coco-eval 报错后走 fallback）；
  LVIS 同口径是实现对齐的主证据。该 self-refer 是 transductive（测试集 GT 参与 vpe），
  不是论文 21.9 的独立 refer 口径；论文的 refer 数据/采样未在公开文档给出，不把 0.35 当 21.9。
- **TensorRT（fp32 / fp16）**：pf E2E、text E2E、text Non-E2E 的 engine 均可构建并
  `infer/eval --engine trt`；部署 metadata 写旁车 `<engine>.json`。text E2E fp16（@320
  冒烟）与 pt fp32：6/6 检出、框差 <0.35px、分数差 <3e-3、掩码 IoU ≥0.996；text NMS
  fp32 的 2 图评估 Box 0.2651 vs pt 0.2656、Mask 完全一致。E2E top-k 对累加差敏感
  （fp32 也会在低分尾部产生不同候选），跨实现比较仍以 Non-E2E 为准。
- **COCO 冒烟**（20 图，管线验证口径，pf）：Box 0.3485 / Mask 0.2865 mAP50-95。
- **LVIS 协议**：`tests/test_lvis_evaluator.py` 在合成数据上验证 neg / not_exhaustive /
  未列出 pair 三条 LVIS 规则，以及零检出图计入、APr/APc/APf、segm RLE 路径；`--data
  lvis-minival` 端到端跑通 Box + Mask。
- 回归闸门 `tests/test_yoloe.py`（结构 / 转换 metadata / strict load / 逐位对齐 / 词汇映射 /
  导出对拍 / 文本模式对拍 / fuse 等价 / 类锁定消费 / Non-E2E / 视觉 prompt 含 mask 与混合）、
  `tests/test_visual_prompt.py`（JSON 契约 / 1/8 光栅化 / refer 聚合）、
  `tests/test_text_encoder.py`（编码器结构 + ts 逐位对拍、tokenizer 黄金序列）、
  `tests/test_yoloe_train.py`（训练前向契约 / fused 前后数值等价 / PE 参数边界 / fused
  权重重建）与 `tests/test_yoloe_trt.py`（`FLASH_YOLO_TRT=1` 门控的 engine 验收）。

### 跨实现对齐（LVIS minival / COCO val2017）

同一份官方 `yoloe-26n-seg` 权重、同一组 text prompts、640 输入；两边使用同一评分协议
（LVIS：faster_coco_eval `lvis_style`，maxDets=100；COCO：COCO 协议，maxDets=100）。
`--nms` 选择 Non-E2E（o2m + 类感知 NMS），默认 E2E。

| 数据集 | 协议 | 本工程 Box | 本工程 Mask | Ultralytics Box | Ultralytics Mask |
|---|---|---:|---:|---:|---:|
| LVIS minival | E2E | 0.2125 | 0.1754 | 0.2231 | 0.1828 |
| LVIS minival | Non-E2E | **0.2218** | **0.1817** | **0.2228** | **0.1826** |
| COCO val2017 | E2E | 0.2832 | 0.2337 | 0.2952 | 0.2408 |
| COCO val2017 | Non-E2E | **0.2952** | **0.2407** | **0.2952** | **0.2407** |

复现命令：

```bash
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --eval-max-dets 100 [--nms]
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data coco [--nms]
```

- Non-E2E 与 Ultralytics 的差距：LVIS ≤0.001，COCO 四位小数一致；AP50、APr/APc/APf、
  COCO 面积分组差 ≤0.003。
- E2E 全量验证对 300 候选 top-k 更敏感：即使同输入 fp32 前向逐位一致（见上），同一权重也可能
  差 ~0.01；跨实现比较以 Non-E2E 为准。
- 官方 24.7（LVIS minival，text prompts）是论文**检测配置**的 Non-E2E 数字；Ultralytics 用
  发布的 **seg** checkpoint 自己的 val 是 0.2228，本工程同口径 0.2218——实现对齐，剩余与
  24.7 的差是“论文检测配置 vs 发布分割权重”。
- LVIS maxDets 100/300 在 minival 上结果相同；COCO 用标准 maxDets=100。

### PE 线性探测（COCO val2017）

以官方 `yoloe-26n-seg` 文本权重为起点，`fuse_text` 后只训练 6 个 cv3 尾卷积（全量 COCO
train2017，10 轮，lr0 0.1 / batch 32 / 640）。正式评估（pycocotools，全量 5000 图）：

| 协议 | Box | Mask | 零样本文本基线 Box / Mask |
|---|---:|---:|---:|
| E2E | **0.3089** | **0.2509** | 0.2832 / 0.2337 |
| Non-E2E | **0.3211** | **0.2570** | 0.2952 / 0.2407 |

- fuse 初始化与未 fuse 文本路径在前 500 图上一致（Box 0.3338 / Mask 0.2790）：PE 的起点
  就是零样本基线，上述增益全部来自 6 个尾卷积。
- **10 轮是推荐预算**：COCO 与 LVIS 都按 10 轮训练；只有验证曲线明显未平台时才增加
  预算（更长的轮数收益很小）。
- 部署链路：同一 fused 权重在 pt 与导出 ONNX 上评估 COCO 前 100 图，Box 0.4031 vs
  0.4032（E2E）；类锁定图可直接 `--engine onnx` 消费，无需 `--classes`。
- 规模与单卡选项：n/s/m/l/x 五档均通过「官方文本权重 strict load → fuse → 只训 6 个尾卷积」
  的 backward 契约；GPU 上 `--amp --multi-scale --single-cls --channels-last` 组合 1 epoch
  冒烟通过（PE 默认 fp32，AMP 为可选）。
- **LVIS（1203 类，federated）**：每图只标注部分类别。训练自动生成「正类 ∪
  `neg_category_ids`」分类掩码，`not_exhaustive_category_ids` 与未标注类从 BCE 忽略
  （不再当负样本）；federated 数据自动关闭 mosaic/mixup/cutmix/copy_paste。PE 10 轮
  （lr0=0.01）minival 正式评估：E2E Box **0.2346** / Mask **0.1905**，Non-E2E
  **0.2436** / **0.1955**（零样本 0.2125 / 0.1754 与 0.2218 / 0.1817）。E2E 频率分组：
  APr 0.1180 / APc 0.2146 / APf 0.2733（零样本 0.1923 / 0.1999 / 0.2272）——总体超过
  基线，rare 类从 0 恢复但仍低于零样本，增益集中在 common/frequent；负类采样 / 类别平衡
  是 rare 类的后续改进项。fused 初始化与文本路径一致（E2E 0.2124 / 0.1753、APr 0.1922）。
- 复现：`python scripts/train.py --data coco --model yoloe-26 --scale n --pe
  --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 32 --lr0 0.1`；评估把
  `scripts/eval.py` 的 `--weights` 换成产物的 `best.safetensors`（E2E / `--nms` 两口径）。

### 全量微调（COCO val2017）

从官方 `yoloe-26n-seg` 文本权重出发，全部参数训练（全量 COCO train2017，10 轮，
lr0 0.001 / batch 32 / 640；单卡 3090 4 h 40 min）。正式评估（pycocotools，全量 5000 图）：

| 协议 | Box | Mask | PE 10 轮 Box / Mask | 零样本 Box / Mask |
|---|---:|---:|---:|---:|
| E2E | **0.3597** | **0.2980** | 0.3089 / 0.2509 | 0.2832 / 0.2337 |
| Non-E2E | **0.3710** | **0.3044** | 0.3211 / 0.2570 | 0.2952 / 0.2407 |

- 比 PE：E2E +0.0508 Box / +0.0471 Mask；Non-E2E +0.0499 / +0.0474。
- **10 轮是推荐起点**：验证曲线明显未平台时可续训（更长预算的收益递减）。
- **lr0 用 0.001**（COCO / LVIS 一致）；不要沿用 PE 的 0.1 / 0.01——全量参数会让预训练
  特征漂移。默认 fp32；AMP 在该模型上没有速度收益。
- 消费：run 目录产物用 `--text-encoder` 指向 MobileCLIP2-B；`infer/export` 未给
  `--classes` 时自动用 metadata 的训练 names，`eval` 用同一数据集描述符。
- 复现：`python scripts/train.py --data coco --model yoloe-26 --scale n --full-finetune
  --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 32 --lr0 0.001
  --warmup-epochs 3`；评估加 `--text-encoder weights/mobileclip2_b.safetensors`
  （E2E / `--nms` 两口径）。
- **LVIS（1,203 类，federated）**：10 轮，batch 16、lr0 0.001；
  minival 正式评估（LVIS 协议 maxDets=100）：E2E Box **0.2578** / Mask **0.2124**
  （Box APr / APc / APf 0.1453 / 0.2373 / 0.2962）、Non-E2E Box **0.2712** / Mask
  **0.2230**（0.1362 / 0.2539 / 0.3109）。PE 10 轮为 0.2346 / 0.1905（APr 0.1180）与
  0.2436 / 0.1955（APr 0.1215）——全量微调 E2E +0.0232 / +0.0219、Non-E2E +0.0276 /
  +0.0275；rare 类 APr 从 0.1180 升到 0.1453，但仍低于零样本 0.1923 / 0.1962，
  增益继续集中在 common/frequent。10 轮是当前推荐预算。
- LVIS 复现：`python scripts/train.py --data lvis --model yoloe-26 --scale n --full-finetune
  --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 16 --lr0 0.001
  --warmup-epochs 3 --val-limit 5000`；评估用 `--data lvis-minival --eval-max-dets 100
  --text-encoder weights/mobileclip2_b.safetensors`（E2E / `--nms` 两口径）。

## 5. 已知边界

- 视觉 prompt 的 mask 必须与 refer 图同尺寸（二值 PNG）；bbox/mask 二选一，同类多示例 OR。
- 文本+视觉混合：infer/export 直接支持；eval 用 `--classes` 指定文本类名（子集，其余零
  嵌入），文本与视觉类名不能重复。官方公开 API 没有混合入口，本工程按 `get_cls_pe` 的
  拼接语义扩展。
- `--refer-data` 参考集需与评估集同类别命名，缺失类零嵌入（AP 0）。官方视觉 val 默认
  `load_vp=True` 用评估集自身抽 vpe，本工程对应 `--refer-data <同一数据集>`（§4 有对照）。
- 文本模式的 pt 路径需要 `--classes` 与 MobileCLIP2-B 文本编码器；`onnx` / `trt` 只能消费
  类锁定图（`export --classes` / `--visual-prompt` / 混合；类别名随 ONNX `metadata_props` /
  TRT 旁车 JSON 走，`--classes` 可省略，给出时必须与锁定列表一致）。评估时锁定类别必须覆盖
  数据集类别（子集导出请配对应子集的描述符；缺失类会报错，不静默丢类）。旧的无 metadata
  导出需重新导出。
- tokenizer 跳过官方 fork 的 `ftfy` 清理（乱码修复 / 全角折叠）：正常 UTF-8 文本逐位一致，
  乱码输入不承诺。
- 导出图与运行时 pt 的提案过滤口径不同（官方同款：导出 conf=0 静态、运行时 conf=0.001）——
  少数低分检测可能不同；conf>0.25 的有效检测实测一致。
- 参数量以 strict-load 一致的 `parameters()` 为准，文本布局的 5,615,540 已含 savpe/reprta；
  官方文档表中的较小数字是**检测配置**口径，与本页的分割布局不是同一结构。
- **训练支持 PE 线性探测与全量微调**：`--pe` 只训练 cv3 / one2one_cv3 的 6 个尾卷积
  （backbone、neck、BN、掩码分支、原型全部冻结，损失只算检测三项），产物类锁定；
  `--full-finetune` 训练全部参数（reprta 在图内），产物未 fused。两者都从官方文本权重
  出发、需要 `--weights`；从零训练尚未接入，prompt-free 权重仅推理（训练会显式报错）。
  全量微调评估必须使用与训练一致的数据集描述符（类名不一致会报错）。
- **LVIS 的 federated 掩码（PE / 全量微调）**：LVIS 每图只标注部分类别；训练自动把
  「正类 ∪ `neg_category_ids`」作为有效类，`not_exhaustive_category_ids` 与未标注类从
  BCE 忽略，并自动关闭 mosaic/mixup/cutmix/copy_paste。总体指标可超过零样本（全量微调
  10 轮 E2E 0.2578 / 0.2124），但 rare 类 AP 仍可能低于零样本（全量微调 APr 0.1453、
  PE 0.1180 vs 零样本 0.1923；增益集中在 common/frequent）。负类采样 / 类别平衡是
  rare 类的后续改进项。详见 §4。

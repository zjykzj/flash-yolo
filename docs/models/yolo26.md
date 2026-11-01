# YOLO26（检测）

> 复现对象：YOLO26n 官方权重经本仓库管线 **40.27** mAP50-95 E2E / **40.89** NMS。
> 结构定义：`config/models/yolo26.yaml`；分割版见 [yolo26-seg.md](yolo26-seg.md)。

## 1. 口径声明

- 官方 40.1（E2E）/ 40.9（NMS）**不是从零训练**：所有官方 checkpoint 都是 Objects365v1
  预训练 150 轮 + COCO 微调（n 档 245 轮、batch 128）。官方文档明确：
  "No YOLO26 checkpoint was trained on COCO from random weights … a from-scratch comparison
  against those numbers is not like for like"。
- 本仓库的复现口径 = 同权重、同前后处理、同 pycocotools 指标、同硬件重跑官方权重；
  从零训练结果见 §4。
- 代码为独立重写（Apache-2.0），不复制 ultralytics 代码；ultralytics 仅作开发期数值对照。

## 2. 结构（n 档）

24 个 yaml 模块（`model.model` 0–23），260 层模块树，**2,572,280 参数 / 5.48 GFLOPs @640**：

| 段 | idx | 内容 |
|---|---|---|
| Backbone | 0–10 | Conv k3 s2 下采样 → C3k2 → SPPF（k=5, n=3）→ C2PSA（1× PSABlock） |
| Neck | 11–22 | Upsample ×2 → Concat → C3k2，P3/P4/P5 三尺度 |
| Head | 23 | Detect 双头（`cv2/cv3` + `one2one_cv2/one2one_cv3`），reg_max=1，E2E top-k 图内 |

- 检测层 P3/P4/P5（stride 8/16/32），输入 640×640，最多 300 检测/图。
- 通道：n 档 3 → 16 → 32 → 64 → 128 → 256；m/l/x 档 C3k2 强制 `c3k=True`。
- BN eps=0.001、momentum=0.03；scale 规则见 [README.md](README.md)。

### Detect 头

- Box：`cv2[i]` = Conv 3×3 → Conv 3×3 → Conv2d(c2→4)，`c2 = max(16, ch[0]//4, 4·reg_max)`；
  reg_max=1 时 4 通道 ltrb 距离（无 DFL）。
- Cls：`cv3[i]` = DWConv+Conv 两级 → Conv2d(c3→nc)，`c3 = max(ch[0], min(nc,100))`。
- `one2one_cv2/one2one_cv3 = deepcopy(cv2/cv3)`。
- bias_init：box 末层 bias=2.0；cls 末层 bias = `log(5/nc/(S/stride)²)`，S = 训练输入边长
  （官方写死 640；非 640 从零训练时先验偏 `(640/S)²` 倍，只影响起点）。

### 解码与后处理

- **E2E**：ltrb → `xyxy = anchor ± dist × stride`；scores=sigmoid；两阶段 top-k(300)：
  ① 每 anchor 取 max 类分 → top-300 anchor；② 在 300×80 候选上再取 top-300 →
  输出 `(B, 300, 6) = [x1,y1,x2,y2,conf,cls]`。顺序必须逐位一致。
- **NMS**：o2m 解码 `(B, 4+nc, 8400)`，**xywh**（官方 `xywh=not end2end`）→ 转 xyxy →
  类感知贪心 NMS（conf 0.001 / iou 0.7 / max_det 300），候选全局按分数降序、跨类坐标偏移隔离、
  凑满 max_det 早停。
- 前处理：letterbox 到 640（灰度 114）、BGR→RGB、/255，无 mean/std。

## 3. 权重下载与初始化

下载链接固定到 **v1.0.0**（下一个发布版本；发布前会 404，可先用 Releases 页面的当前版本）。
资产名固定，发新版时统一替换 tag 即可。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `yolo26n-from-scratch-e300-640` | 300 ep 从零 | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolo26n-from-scratch-e300-640.safetensors) |

官方权重（O365 预训练 + COCO 微调）不随 Release 分发，需自行下载并转换：

```bash
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco
```

- 官方 ckpt 的 `ckpt["model"]` 是完整模块对象（`ema` 为 None）；转换器解包一次写成纯
  safetensors，运行时零 ultralytics 依赖。
- `strict=True` 全 key 匹配是结构对齐的第一道闸门；成功即证明拓扑一致。
- 本地产物 metadata 自带 `arch=yolo26` / `scale` / `nc` / `imgsz` / `names`，
  eval/infer/export 可自动解析。

## 4. 正式指标（COCO val2017 @640）

| 训练 | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 |
|---|---:|---:|---:|---:|
| 官方权重（O365 + COCO） | **40.27** | 55.80 | **40.89** | 56.87 |
| 从零 300 ep | **37.82** | **53.24** | **38.65** | **54.46** |

- 官方发布值 40.1 / 40.9 与本管线 40.27 / 40.89 的 ~0.1 差是指标实现噪声。

## 5. 推理延迟

> 口径：**model-only**（不含前后处理），输入 640；b1 = 单图，b8 = batch 8 每图平均。
> 引擎 init 已做 warmup，数字为重复测量的稳态值；同一架构不同权重的延迟相同。

### 云服务器 / 工作站

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.37 ms | 0.245 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 1.28 ms | 0.135 ms |
| Xeon 8470Q（25 核配额） | ONNX Runtime 1.30 · fp32 | 14.6 ms | 6.86 ms |

- ORT 条件：intra_op=16 / inter_op=1；b1 固定图、b8 动态图；300 样本取 best-case min。
- TRT 条件：b1 固定引擎、b8 batch-8 引擎；30 次单发 + 100 次流水线。
- b1 贴近启动开销地板，b8 才能反映算力差；ORT 绝对值受宿主邻居负载影响。

### 本地笔记本

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| （CPU 型号待填） | ONNX Runtime · fp32 | 待测 | 待测 |

## 6. 用法

```bash
python scripts/train.py --data coco --model yolo26 --name yolo26n-coco --batch 32 --workers 8
python scripts/eval.py --weights yolo26n-from-scratch-e300-640.safetensors --data coco
python scripts/export.py --weights yolo26n-from-scratch-e300-640.safetensors --trt
python scripts/infer.py --weights yolo26n-from-scratch-e300-640.safetensors \
    --image assets/traffic.jpg
```

训练参数、recipe、DDP：见 [../pipeline/train.md](../pipeline/train.md)；评估口径见
[../pipeline/eval.md](../pipeline/eval.md)。

## 7. 注意

- 属性树（`cv1/cv2/cv3/m`、`one2one_*`）必须保持官方命名；重命名会破坏 strict load。
- 组装器元数据用 `m_.repeats`，**不要挂 `m_.n`**（SPPF 的 `self.n` 是池化次数）。
- E2E 的两阶段 top-k 顺序必须与官方一致，否则逐位对拍过不去。
- 从零训练默认 fp32；`--amp` 仅建议微调场景。

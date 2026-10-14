# YOLOv3-tiny 架构设计与实现说明

> 定位：本仓库的**第二架构**（2026-10 加入）——轻量对照基线与"廉价空间下采样"思路来源。
> 结构基准 = **darknet 官方 cfg**（cfg/yolov3-tiny.cfg，论文 arXiv:1804.02767 谱系，原生 416 输入）。
> 本文记录结构事实与本仓库的实现口径；训练/评估记录见 `yolov3-tiny-runs.md`。

## 1. 口径声明（读之前先看）

- **结构基准是 darknet，不是 ultralytics**。ultralytics 8.4.x 里的 "v3"（资产只有
  yolov3-tinyu.pt）是**无锚 DFL 重写**：SiLU 激活、reg_max=16、`Detect` 连 `anchors` 形参都没有。
  它既不能作为结构来源，也不能作为数值对齐对象。
- **算子替换版（2026-10-13 新增）**：`yolov3-tiny-silu`（`config/models/yolov3-tiny-silu.yaml`）
  ——与本文原版**逐行同构**，唯一差异 = LeakyReLU(0.1) → SiLU（激活无参数：两版 state_dict
  键集、参数量 8,852,366、@416 GFLOPs 均逐位一致）。官方 `.weights` 只配原版使用；两版由
  权重 metadata 的 `arch` 字段与文件名（`_ARCH_RE` 长名优先）区分。训练侧另有 IoU-aware objectness
  开关 `--obj-iou-target`（见第 6 节）。
- 本实现的验收对象 = 官方 darknet `.weights`（经 `scripts/convert_weights.py` 严格加载后
  与 cv2.dnn / 官方发布的 AP 对得上），见第 9 节。
- 与 yolo26 共享的只有**训练管线**（trainer/EMA/优化器/增强/数据加载/引擎）；模型、头、
  解码、损失、assigner 都是独立实现。

## 2. 逐层结构（`config/models/yolov3-tiny.yaml`）

416 输入下的 21 个模块（idx = `model.model` 下标，与权重转换的层序一一对应）：

| idx | from | 层 | 参数 | 输出尺寸 | 通道 |
|---|---|---|---|---|---|
| 0 | -1 | LeakyConv | [16, 3, 1] | 416 | 3 → 16 |
| 1 | -1 | MaxPool2d | [2, 2] | 208 | 16 |
| 2 | -1 | LeakyConv | [32, 3, 1] | 208 | 16 → 32 |
| 3 | -1 | MaxPool2d | [2, 2] | 104 | 32 |
| 4 | -1 | LeakyConv | [64, 3, 1] | 104 | 32 → 64 |
| 5 | -1 | MaxPool2d | [2, 2] | 52 | 64 |
| 6 | -1 | LeakyConv | [128, 3, 1] | 52 | 64 → 128 |
| 7 | -1 | MaxPool2d | [2, 2] | 26 | 128 |
| 8 | -1 | LeakyConv | [256, 3, 1] | 26 | 128 → 256 ← P4 融合来源 |
| 9 | -1 | MaxPool2d | [2, 2] | 13 | 256 |
| 10 | -1 | LeakyConv | [512, 3, 1] | 13 | 256 → 512 |
| 11 | -1 | ZeroPad2d | [[0,1,0,1]] | 13 | 512 |
| 12 | -1 | MaxPool2d | [2, 1] | 13 | 512 |
| 13 | -1 | LeakyConv | [1024, 3, 1] | 13 | 512 → 1024 |
| 14 | -1 | LeakyConv | [256, 1, 1] | 13 | 1024 → 256 |
| 15 | -1 | LeakyConv | [512, 3, 1] | 13 | 256 → 512 ← P5/32 头输入 |
| 16 | -2 | LeakyConv | [128, 1, 1] | 26 | 256 → 128（darknet `route -4`） |
| 17 | -1 | Upsample | [None, 2, nearest] | 26 | 128 |
| 18 | [17, 8] | Concat | [1] | 26 | 384 |
| 19 | -1 | LeakyConv | [256, 3, 1] | 26 | 384 → 256 ← P4/16 头输入 |
| 20 | [19, 15] | V3Detect | [nc] | 13 + 26 | 双尺度输出 |

- 规模：**21 层 / 8,852,366 参数（8.85M）/ 5.56 GFLOPs @416**（与 yolo26n@640 的
  5.48 GFLOPs 同档，是它的 GFLOPs 对齐点）。
- **保尺寸池化的两行表达**：darknet `[maxpool] size=2 stride=1` 隐含右下 padding，
  PyTorch 侧写作 `ZeroPad2d([0,1,0,1]) + MaxPool2d(2,1)`（idx 11+12）。不要用
  `MaxPool2d(2,1,pad)` 或居中 padding —— 与 darknet 的像素对齐不一致。
- **两颗头卷积折进 V3Detect**（idx 20 内部）：darknet 里它们是 cfg 末尾的独立
  `[convolutional]` 层（1×1 → 3·(5+nc)，bias、无 BN/激活），此处与 yolo26 的 `Detect`
  持有 `cv2/cv3` 的惯例一致地折进头里。
- **route 的相对索引在 parse 期归一化**：第 16 层 yaml 写 `-2`（darknet `route -4`），
  组装器在解析时折算为绝对下标 14 —— 与 darknet 的"-4 从 route 层自身往前数"语义等价。

## 3. 基础算子语义

- **LeakyConv**：`Conv2d(k, s, pad=k//2, bias=False)` → BN(**eps=0.001**、momentum=0.03)
  → `LeakyReLU(0.1)`。darknet 的 BN 口径与 yolo26 一致（eps 非 PyTorch 默认 1e-5）。
- **MaxPool2d**：核 2 stride 2（下采样）/ stride 1（保尺寸，配前置 ZeroPad2d）。
- **Upsample**：nearest ×2；**Concat**：通道维拼接（`model.build.Concat` 复用）。
- **V3Detect**：每级 `nn.Conv2d(c, 3·(5+nc), 1)`（bias、无 BN/act，darknet linear 卷积口径）；
  无 cv2/cv3 属性、无 `end2end` 开关、无 bias_init（darknet 无此先验）。

## 4. anchors（v3 最大的坑）

- **锚值是 yaml 数据**：每级 3 槽 × [w, h]，级序 = V3Detect 的 `from` 顺序（[P4/16, P5/32]）：

  | 级 | stride | 槽位（像素） |
  |---|---|---|
  | P4/16 | 16 | [23,27] · [37,58] · [81,82] |
  | P5/32 | 32 | [81,82] · [135,169] · [344,319] |

- **官方 mask 照原样保留**：P5 档 mask 3,4,5 / P4 档 mask 1,2,3 —— 原版历史坑
  （81,82 被两个尺度共用、10,14 从未启用），官方权重就是照此训练的。
- **固定像素单位，绝不随 imgsz 缩放**：官方权重在多尺度训练下由 tw 自补偿输入尺度，
  锚值必须原样配套。实测（500 图 FastMetrics，官方权重）：

  | 口径 | mAP50 |
  |---|---|
  | 416 原生 | 0.384 |
  | 640 + 原始锚值 | **0.412** |
  | 640 + 线性缩放锚值 | 0.154 |

  初版实现按 `imgsz/416` 线性缩放锚值 → 直接把 640 的精度打到 0.154。这是"看起来更合理
  但实际错"的典型：darknet 谱系没有训练启动时自动重算锚点的机制，锚点是权重的一部分。
- **persistent=False buffer**：`anchors`/`stride` 不进 state_dict —— 转换器的键集对账、
  EMA 键断言、strict load 三处都零特判；同时随 `.to(device)` 自动搬运。
- **从零训练换输入尺寸时才重聚类**（`scripts/compute_anchors.py`，见第 8 节）。
- **metadata 应用点**：重聚类后的锚点写到权重 metadata 随权重走；`PtEngine` 构建后
  `apply_meta` 覆盖 `head.anchors`（形状不符报错）。**训练**用 `--weights` 初始化时
  **不应用** metadata（架构以 yaml 为准）——自定义锚点微调需自带 yaml。

## 5. 解码契约（`model/head_v3.py::decode_level`）

**head 与 loss 共用的唯一实现**（darknet `get_yolo_box` 口径）：

```
raw (B, 3·(5+nc), H, W)  通道序 = 逐锚 [x, y, w, h, obj, cls...]
xy = (σ(t) + grid) · stride        # 网格偏移 0（darknet 口径，非 v5 的 0.5 偏移）
wh = anchor · exp(t)               # tw/th 取原始 logits，先 clamp(±8) 防早期 exp 溢出
```

- eval 前向输出：解码 + sigmoid 后拼接 **(B, NA, 5+nc) = [x1,y1,x2,y2, σ(obj), σ(cls)...]**，
  NA = 3·ΣH·W（416 → 3·(26²+13²) = 2535；640 → 3·(40²+20²) = 6000），
  扁平序 = 级 → 槽 → y → x。
- **不要换成 v5 式 `(2σ−0.5)/(2σ)²`**：与官方权重、cv2.dnn 输出、33.1 AP 验收全部对不上。
- 最终置信 = **obj × cls**（darknet 口径），在 `utils/postprocess.v3_detections` 里做；
  类感知 NMS（默认 conf 0.001 / iou 0.7 / max_det 300）。
- **无 E2E 路径**：`--nms` / `--raw` 是 yolo26 专属开关，对 v3 直接 `parser.error`。
- `postprocess_val` 与 `Detect.postprocess_val` 同契约（每图 (M,6)），validator 的逐图
  循环对两者无差别。

## 6. 训练：损失与匹配（`train/loss_v3.py::ComputeLossV3`）

- **匹配（darknet 语义）**：逐 GT、逐级，在 3 个槽位里取「形状 IoU」（中心重合的宽高 IoU）
  最大者为唯一正样本；责任格子 = GT 中心所在 cell（clamp 到网格内）。每 GT 每级至多
  一个正样本 —— 没有 TAL/STAL、没有 o2m/o2o 双头、没有 ProgLoss。
- **忽略区**：非正样本中「预测解码框与任一 GT 的 IoU ≥ 0.7（`ignore_thresh`）」的槽位
  obj 不罚（darknet 口径），其余为负样本。
- **损失三项**：`BCE(obj)`（按非忽略槽位数逐图归一）+ `BCE(cls)` 独热（仅正样本）+
  `CIoU`（仅正样本，box_gain 7.5 / cls_gain 0.5 / obj_gain 1.0）。
- 归一化沿用仓库口径：逐图均值 + Σ_级 Σ_图，与 trainer 的 `/batch` 及 nbs/accum 梯度口径
  衔接，梯度尺度与物理 batch 无关。
- 批量实现（`(B, N, M)` 掩码），无逐图循环 —— 与 yolo26 侧同一性能红线。
- 接口与 `ComputeLoss` 同构（`item_keys` / `__call__`），**无 `set_alpha`**（ProgLoss 不存在），
  trainer 以 `hasattr` 守卫。`item_keys = ("box", "obj", "cls")` → 训练日志/csv 的损失列名
  由此派生（yolo26 输出逐字节不变，测试断言其列）。

## 7. 权重转换（darknet `.weights`，`scripts/convert_weights.py`）

- **文件布局**：4 或 5 个 int32 头（版本 [+ seen]）+ float32 小端张量流；
  头部宽度按**文件长度自动判别**，并要求**文件恰好用尽**（多/少字节都报错）。
- **张量序 = darknet cfg 层序**：注意 **P5 头卷积（512→255）位于 P4 支路卷积
  （conv128/conv256）之前** —— 即 cfg 里第一个 `[yolo]` 的位置，不是 yaml 的书写顺序。
- **块内顺序**：BN 层 = `[bias(β), scale(γ), mean, var, conv 权重]`；输出层 = `[bias, 权重]`。
  conv 权重布局 `(out, in, kh, kw)` 与 torch 一致，**无需转置**。
- **映射以输入通道数对号**：512ch → P5 头 / 256ch → P4 头（配合层序硬编码表，
  `_darknet_specs`，改 yaml 需同步此表）。
- 非文件张量（`num_batches_tracked`）取零初值 —— 推理不受影响。
- **metadata**：`arch/scale/nc/anchors`（锚点取自构建好的模型 = yaml 官方原值，随权重走）；
  **names 不写**（.weights 文件里没有类别名信息，不如实写"未知"）。
- 实测对账：官方文件 **35,434,956 B = 8,858,734 floats + 20 B 头**，严格加载通过。

## 8. 工具：`scripts/compute_anchors.py`（通用，任何含 `anchors:` 段的模型）

- 算法 = darknet `calc_anchors` 口径的 k-means（距离 1 − IoU、n_init 次随机重启取最优、
  seed 固定可复现）+ YOLOv5 `check_anchors` 同款覆盖判定（形状比阈值 `--anchor-t`，默认 4.0）。
- 输出按面积升序均分为 `--levels` 组（细 → 粗），直接可粘贴进模型 yaml；`--out` 落盘。
- `--model`（默认 yolov3-tiny）决定"现锚组"从哪个 yaml 读取；`--anchors` 可直接评估候选锚集。
- **判定口径**：v5 的 4.0 / 98% 线对应其 `(2σ)²` 解码的硬边界；darknet 的 exp 解码没有硬
  边界（理论上任何锚可达任意尺寸），覆盖率只是先验质量的**软信号** —— 所以 verdict 同时看
  重聚类能把 bestIoU 提升多少（差 < 0.03 判"可选/不必"）。
- COCO 实测（2026-10）：@640 官方锚 coverage 97.0% / bestIoU 0.520 vs 重聚类 96.1% / 0.522
  —— **收益≈0，不必重算**；@416 官方 92.8% / 0.498 vs 重聚类 bestIoU 0.522（差 0.024，仍"可选"）。

## 9. 工程接入点

- **选择方式**：`--model yolov3-tiny`（train / eval / infer / export 全线支持）；
  结构固定 → **不要传 `--scale`**（传了告警且不生效）。
- **档位推断**：`model/weights.py::resolve_arch_scale` 按权重文件名推
  （`yolov3-tiny.safetensors → v3`）；文件名里的架构优先于误传的 `--scale`。
- **架构注册**：`model/build.py::ARCHS`（`build_yolov3_tiny`）；损失由 `build_loss` 按架构分派；
  Trainer 按 `cfg.model` 分派（非 yolo26 → yaml 默认档）。
- **训练配方**：默认吃内置 `default`；**推荐配置 = `--model yolov3-tiny-silu` +
  `--obj-iou-target`（+ default 核）**，100 轮正式 15.23 / 28.58（见 runs 文档 §4.5）。
  darknet 对齐配方（旧 `yolov3-tiny-darknet.yaml`）已经探针定案、随实验完成退役删除
  （见 runs 文档 §4.4）。
- **测试**（`pytest tests/`）：`test_v3_model.py`（结构/参数/前向形状/解码数值/锚点 imgsz
  不变性/未知档位拒绝/strict 往返）、`test_v3_loss.py`（分派/匹配/空 GT/批量口径/梯度回流）、
  `test_v3_io.py`（档位推断/darknet 转换往返与长度守卫/PtEngine/export 对拍/onnx imgsz）,
  以及 `test_export_parity` 等共享闸门。

## 10. 验收

- **官方权重**：`scripts/download_weights.py --model yolov3-tiny` →
  `scripts/convert_weights.py`（字节对账见第 7 节）→
  `scripts/eval.py --weights weights/yolov3-tiny.safetensors --data coco`：

  | 输入 | mAP@[.5:.95] | mAP@50 |
  |---|---|---|
  | 640 | **17.16** | **35.90** |
  | 416 | 16.94 | 34.37 |

  官方发布的 33.1 AP50 是 **COCO test-dev @416** 口径；同权重本管线 640 比 416 高（锚值不
  缩放 + 有效分辨率更高），两者不可直接相减。
- 从零训练数字、全部 run 记录与配方归因实验：见 `yolov3-tiny-runs.md`。

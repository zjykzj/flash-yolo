# 与官方实现的对齐与验证（Alignment & Verification）

> 本页收纳从 README 移出的官方对齐、数值验证与测试说明。项目运行时**不依赖 ultralytics**；
> 所有官方对照仅在开发期使用（`requirements-dev.txt`），对照结果不进入交付链路。

## 1. 官方权重对齐（YOLO26n 检测）

口径：COCO val2017、官方 pycocotools（`scripts/eval.py`）、640 输入。E2E = NMS-free 默认路径；
NMS = one-to-many 头 + NMS。`ultralytics` 行使用**完全相同**的预处理、后处理与评估管线，
只有 forward 的实现不同。

| 模型 | 实现 | 训练 | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 | 参数 | GFLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n 官方权重 | 本仓库 | O365 预训练 150 ep + COCO 微调 245 ep | **40.27** | 55.80 | **40.89** | 56.87 | 2.57M | 5.48 |
| YOLO26n 官方权重 | ultralytics | 同一 checkpoint | 40.27 | 55.80 | 40.89 | 56.87 | 2.57M | 5.48 |
| YOLO26n 从零 | 本仓库 | 100 ep，baseline 配方，fp32，单卡 RTX 5090 | 36.22 | 51.32 | 37.15 | 52.67 | 2.57M | 5.48 |
| Flash-YOLO 从零 | 本仓库 | 100 ep，同预算 | 33.77 | 48.63 | 34.72 | 50.16 | 1.97M | 3.80 |

- **逐位复现**：官方 checkpoint（260 层 · 2,572,280 参数）以 `strict=True` 零 key 映射加载；
  同一输入下 forward 输出与官方实现逐位一致。
- 官方发布值 40.1 / 40.9 与本管线 40.27 / 40.89 的 ~0.1 差值来自指标实现噪声，不是模型差异。
- 从零行与官方行**不是同预算**：官方 checkpoint 均从 Objects365 预训练开始。

```bash
# 官方权重：下载 + 转换 + 评估（~10 min，无需训练）
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco        # 40.27 / 55.80
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms  # 40.89 / 56.87

# 可选交叉验证：同一权重用 ultralytics 实现跑本仓库管线（仅 dev-time）
python scripts/compare_official.py --data coco
python scripts/compare_official.py --data coco --nms
```

## 2. 官方权重对齐（YOLO26n-seg 分割）

口径：COCO val2017 全量 5000 图，`scripts/eval.py`（pycocotools，Box + Mask 双任务轴）。

| 口径 | mAP50 | mAP50-95 |
|---|---:|---:|
| Box | 0.5548 | **0.3967** |
| Mask | 0.5276 | **0.3367** |

官方训练期 validator 记录为 box **0.3975** / mask **0.3386**，与本管线差 ~0.002——与检测侧
40.1 vs 40.27 同类的指标实现噪声。掩码解码（`process_mask`）与官方 val 逐像素一致；
导出侧 pt(CPU) vs onnx(CPU) 原始输出相对差 ~1e-6，跨设备（CUDA vs CPU oneDNN）为
1e-3~4e-3、掩码边界像素有翻转（最差实例 IoU 0.993）——跨设备对齐应在同运行时比较或走
TensorRT，详见 [flash-yolo-seg-spec.md](flash-yolo-seg-spec.md)。

## 3. 训练内部对齐

同一权重、同一真实 batch 下与官方实现逐项对照：

- **损失 / 梯度**：总损失差 <1%，梯度范数差 <2%（本仓库 2091 vs 官方 2130）。
- **o2o 分支**：detach 特征输入而非输出——梯度只经 o2m 回流 backbone/neck，o2o 头自身仍训练。
- **优化器**：MuSGD 分组、NS 迭代与系数与官方一致；梯度裁剪 `max_norm=10` 每步无条件执行；
  EMA 公式 `decay*(1-exp(-steps/tau))` 与更新时机一致；ProgLoss 每 epoch 更新。
- **增强**：mosaic 画布 1,638,400 px 0 差异、仿射输出 409,600 px 0 差异、标签 0 差异。
- **数据格式**：COCO-json 与 YOLO-txt 加载器在同一批图上框完全一致（0 差异），训练侧统计
  逐项相等（14205 / 14205）。

对齐口径与历史坑的完整记录见 [train.md](train.md)、[val.md](val.md) 与各架构 spec。

## 4. 运行时一致性

- **pt ↔ onnx**：raw 输出相对差 <5e-4，E2E 检出 <1e-3。
- **TensorRT**：fp32 引擎与 PyTorch 差 <2e-3；fp16 引擎与 PyTorch half 模型的 raw 输出逐位一致
  （两边都是 fp16 乘 + fp32 累加 + 终值舍入）。
- **分割双输出**：`output0`（检测 + 掩码系数）与 `output1`（原型）在 pt / onnx / TensorRT 三后端
  契约一致；掩码后处理与官方 val 逐像素一致。

## 5. 可复现性

- 默认 `--deterministic`：关闭 cuDNN autotune、使用确定性 kernel、每个 epoch 重播增强 RNG；
  同 seed 两次训练权重**逐位相同**，resume 与不中断训练**逐位相同**。
- `FLASH_YOLO_REPRO=1` 门控 CPU/GPU 双跑与 resume 等价测试（实测 0 位差）。
- 训练产物（best/last/epochNNN）的 metadata 记录 arch / scale / nc / imgsz / names / anchors
  与出处 commit，权重自包含、可审计。

## 6. 测试

```bash
pytest tests/          # 286 collected: 268 passed, 18 skipped (env-gated / optional deps)
pytest tests/test_weight_alignment.py -v   # 需要 dev deps + 转换后的官方权重
```

- **环境门控**：`FLASH_YOLO_SMOKE`（CPU 冒烟 / 训练小闭环）、`FLASH_YOLO_REPRO`
  （CPU/GPU 双跑 + resume 等价）。
- **覆盖范围**：权重逐位对齐、导出 parity、评估器与指标正确性、训练组件（assigner / loss
  parity / MuSGD / EMA / checkpoint / augment / dataset / config / smoke / head-train）、
  v3（model / loss / darknet converter / engine / export 对拍）、分割（model / loss /
  dataset / metrics / validator / smoke / repro）。

## 7. 相关文档

- [权重与延迟档案](weights.md) — 发布 checkpoint 的完整精度 / 速度 / 部署产物
- [评估口径](val.md) — pycocotools、FastMetrics、NMS 与评估坑
- [训练](train.md) — 配方、增强、优化器、复现
- 架构说明：[yolo26-spec.md](yolo26-spec.md) · [flash-yolo-spec.md](flash-yolo-spec.md) ·
  [flash-yolo-seg-spec.md](flash-yolo-seg-spec.md) · [yolov3-tiny-spec.md](yolov3-tiny-spec.md)

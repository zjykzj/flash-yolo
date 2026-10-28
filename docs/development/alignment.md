# 与官方实现的对齐与验证

> 本页收纳官方对齐、数值验证与测试说明。项目运行时**不依赖 ultralytics**；所有官方对照
> 仅在开发期使用（`requirements-dev.txt`），不进入交付链路。

## 1. 官方权重对齐（YOLO26n 检测）

口径：COCO val2017、pycocotools（`scripts/eval.py`）、640 输入。E2E = NMS-free 默认路径；
NMS = one-to-many 头 + NMS。`ultralytics` 行使用**完全相同**的预处理、后处理与评估管线，
只有 forward 实现不同。

| 模型 | 实现 | 训练 | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 |
|---|---|---|---:|---:|---:|---:|
| YOLO26n 官方权重 | 本仓库 | O365 150 ep + COCO 245 ep | **40.27** | 55.80 | **40.89** | 56.87 |
| YOLO26n 官方权重 | ultralytics | 同一 checkpoint | 40.27 | 55.80 | 40.89 | 56.87 |
| YOLO26n 从零 | 本仓库 | 100 ep | 36.22 | 51.32 | 37.15 | 52.67 |
| YOLO26n 从零 | 本仓库 | 300 ep | **37.82** | **53.24** | **38.65** | **54.46** |
| Flash-YOLO 从零 | 本仓库 | 100 ep | 33.77 | 48.63 | 34.72 | 50.16 |
| Flash-YOLO 从零 | 本仓库 | 300 ep | **35.14** | **50.15** | **36.09** | **51.56** |

- **逐位复现**：官方 checkpoint（260 层 · 2,572,280 参数）以 `strict=True` 零 key 映射加载；
  同一输入下 forward 输出与官方实现逐位一致。
- 官方发布值 40.1 / 40.9 与本管线 40.27 / 40.89 的 ~0.1 差来自指标实现噪声。
- 从零行与官方行**不是同预算**：官方 checkpoint 均从 Objects365 预训练开始。

```bash
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms

# 可选交叉验证：同一权重用 ultralytics 实现跑本仓库管线（仅 dev-time）
python scripts/compare_official.py --data coco
python scripts/compare_official.py --data coco --nms
```

## 2. 官方权重对齐（YOLO26n-seg）

口径：COCO val2017 全量 5000 图，`scripts/eval.py`（pycocotools，Box + Mask）。

| 口径 | mAP50 | mAP50-95 |
|---|---:|---:|
| Box | 0.5548 | **0.3967** |
| Mask | 0.5276 | **0.3367** |

官方训练期 validator 记录 box **0.3975** / mask **0.3386**，与本管线差 ~0.002——与检测侧
40.1 vs 40.27 同类的指标实现噪声。掩码解码与官方 val 逐像素一致；导出侧 pt(CPU) vs
onnx(CPU) 原始输出相对差 ~1e-6，跨设备（CUDA vs CPU oneDNN）为 1e-3~4e-3、掩码边界有
像素翻转（最差实例 IoU 0.993）——跨设备对齐应在同运行时比较或走 TensorRT。

## 3. 训练内部对齐

同一权重、同一真实 batch 下与官方实现逐项对照：

- **损失 / 梯度**：总损失差 <1%，梯度范数差 <2%（本仓库 2091 vs 官方 2130）。
- **o2o 分支**：detach 特征输入而非输出——梯度只经 o2m 回流 backbone/neck，o2o 头自身仍训练。
- **优化器**：MuSGD 分组、NS 迭代与系数与官方一致；梯度裁剪 `max_norm=10` 每步无条件执行；
  EMA 公式 `decay*(1-exp(-steps/tau))` 与更新时机一致；ProgLoss 每 epoch 更新。
- **增强**：mosaic 画布 1,638,400 px 0 差异、仿射输出 409,600 px 0 差异、标签 0 差异。
- **数据格式**：COCO-json 与 YOLO-txt 加载器在同一批图上框完全一致（0 差异），训练侧统计
  逐项相等（14205 / 14205）。

完整历史坑见 [../pipeline/train.md](../pipeline/train.md)、
[../pipeline/eval.md](../pipeline/eval.md) 与各模型页。

## 4. 运行时一致性

- **pt ↔ onnx**：raw 输出相对差 <5e-4，E2E 检出 <1e-3。
- **TensorRT**：fp32 引擎与 PyTorch 差 <2e-3；fp16 引擎与 PyTorch half 模型的 raw 输出逐位
  一致（两边都是 fp16 乘 + fp32 累加 + 终值舍入）。
- **分割双输出**：`output0`（检测 + 系数）与 `output1`（原型）在 pt / onnx / TRT 三后端契约
  一致；掩码后处理与官方 val 逐像素一致。

## 5. 可复现性

- 默认 `--deterministic`：关闭 cuDNN autotune、使用确定性 kernel、每个 epoch 重播增强 RNG；
  同 seed 两次训练权重**逐位相同**，resume 与不中断训练**逐位相同**。
- `FLASH_YOLO_REPRO=1` 门控 CPU/GPU 双跑与 resume 等价测试（实测 0 位差）。
- 训练产物 metadata 记录 arch/scale/nc/imgsz/names/anchors 与出处 commit，权重自包含、可审计。

## 6. 测试

```bash
pytest tests/ -q          # 核心套件；env-gated 套件默认跳过
pytest tests/test_weight_alignment.py -v   # 需要 dev deps + 转换后的官方权重
```

- **环境门控**：`FLASH_YOLO_SMOKE`（CPU 冒烟 / 训练小闭环）、`FLASH_YOLO_REPRO`
  （CPU/GPU 双跑 + resume 等价）、`FLASH_YOLO_DDP`（2 进程 CPU gloo）。
- **覆盖范围**：权重逐位对齐、导出 parity、评估器与指标正确性、训练组件（assigner /
  loss parity / MuSGD / EMA / checkpoint / augment / dataset / config / smoke / head-train）、
  v3（model / loss / darknet converter / engine / export 对拍）、分割（model / loss /
  dataset / metrics / validator / smoke / repro）。
- CI：`.github/workflows/tests.yml` 在 push/PR 跑 ungated CPU 核心套件（不装 ultralytics）。

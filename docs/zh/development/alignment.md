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
| YOLO26n 从零 | 本仓库 | 300 ep | **37.82** | **53.24** | **38.65** | **54.46** |
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
| 官方权重 Box | 0.5548 | **0.3967** |
| 官方权重 Mask | 0.5276 | **0.3367** |
| 本仓库从零 300 ep Box | 0.5369 | 0.3790 |
| 本仓库从零 300 ep Mask | 0.5116 | 0.3237 |

从零 300 轮（default 配方）与官方（O365 + COCO）的差距为 **1.77 / 1.30**（Box / Mask
mAP50-95），与检测侧同量级。

官方训练期 validator 记录 box **0.3975** / mask **0.3386**，与本管线差 ~0.002——与检测侧
40.1 vs 40.27 同类的指标实现噪声。掩码解码与官方 val 逐像素一致；导出侧 pt(CPU) vs
onnx(CPU) 原始输出相对差 ~1e-6，跨设备（CUDA vs CPU oneDNN）为 1e-3~4e-3、掩码边界有
像素翻转（最差实例 IoU 0.993）——跨设备对齐应在同运行时比较或走 TensorRT。

## 3. 官方权重对齐（YOLO26n-pose）

口径：COCO-pose val2017 的 **2,346 张有有效关键点的图**，`scripts/eval.py`（pycocotools
keypoints，OKS，maxDets=20）；`ultralytics` 行使用同一批图、同一预处理/后处理，仅 forward
实现不同。

| 模型 | 实现 | E2E AP | E2E AP50 | NMS AP | NMS AP50 |
|---|---|---|---:|---:|---:|---:|
| YOLO26n-pose 官方权重 | 本仓库 | **0.5687** | 0.8250 | **0.5670** | 0.8280 |
| YOLO26n-pose 官方权重 | ultralytics（同 2,346 图） | 0.572 | 0.833 | 0.570 | 0.831 |

- **逐位复现**：官方 `yolo26n-pose`（363 层 · **3,679,464 参数**）以 `strict=True` 全 key
  加载（含 `cv4_kpts` / `cv4_sigma` / `flow_model`）；同一输入下 E2E `(1,300,57)` 与
  raw `(1,56,8400)` 输出与官方 forward **max diff 0.0**。
- 与 ultralytics 的 ~0.003 AP 差来自 square letterbox（官方 val 是 auto/rect）与指标实现
  噪声；训练期 validator 记录 pose 0.5717 / AP50 0.8332，同量级。
- 导出/运行时：ONNX E2E/raw 与 pt 在 50 图子集上的指标差 ≤0.001；raw 输出逐元素差
  ≤1e-2 px（均值 ~3e-5）；TRT fp32/fp16 与 pt 的 50 图 E2E AP 为 0.6442 / 0.6448 vs
  0.6451（conf>0.25 检出无翻转）。
- **训练内部**：`ComputeLossPose` 与官方 `PoseLoss26` 在真实 COCO-pose batch 上总损失
  **40.0884819 vs 40.0884819（rel 0）**；o2o 的 pose/kobj/rle 三项逐项 parity
  （`tests/test_loss_pose.py`）。
- 相关测试：`tests/test_pose.py`（strict load / forward parity / 导出分支回归）、
  `tests/test_dataset_pose.py`（关键点对齐 / flip_idx / mosaic）、
  `tests/test_metrics_pose.py`（OKS AP）、`tests/test_train_smoke.py -k pose`（门控）。

### 3.1 10 轮微调管线对齐（2026-11）

口径：官方 `yolo26n-pose` 权重初始化；本仓库与 ultralytics 8.4.173 使用同一批
**56,599 张训练图 / 2,346 张 val 图**，10 轮、batch 64、nbs 64、640、MuSGD、AMP、seed 0。
两边关闭 `close_mosaic`，并按 released ultralytics 代码的有效值覆盖
`cls_w=1.0` / `prog_alpha_init=0.8` / `muon_w=0.2` / `sgd_w=1.0`（该版本不消费 `cls_w`，
后三项硬编码；`yolo26-pose-coco` 配方记录的是官方 ckpt 的 `train_args`）。训练完成后
两边用同一个 `scripts/eval.py --data coco-pose` 复评。

| 实现 | E2E AP | E2E AP50 | NMS AP | NMS AP50 |
|---|---:|---:|---:|---:|
| 本仓库 10 轮 | **0.5476** | 0.8089 | **0.5528** | 0.8170 |
| ultralytics 10 轮 | **0.5477** | 0.8164 | **0.5508** | 0.8203 |

- E2E AP 差 **0.0001**、NMS AP 差 **0.0020**；AP75 / AP(M) / AP(L) / AR 分项最大差约
  **0.004**，在单 seed、10 轮预算的噪声内——数据/增强/损失/优化器/EMA/验证链路完成一次
  端到端对齐验证。
- 这是**管线 parity 检查，不是收敛精度**：从已收敛的官方权重以 lr0 0.005 再训 10 轮，
  两边都回落到 ~0.548（官方起点 0.5687）；不作为模型能力上界。
- 训练中 validator 的 pose AP 与 pycocotools 差距明显（本仓库 FastPose 0.4367、
  ultralytics PoseMetrics 0.3676 vs 正式 0.5476 / 0.5477）；训练日志只用于趋势与 best，
  跨实现不可直接比较。

```bash
# 本仓库侧；ultralytics 侧用同一数据与有效超参（dev-time）
python scripts/train.py --data coco-pose --model yolo26-pose \
    --weights weights/yolo26n-pose.safetensors \
    --recipe yolo26-pose-coco --epochs 10 --batch 64 \
    --close-mosaic 0 --cls-w 1.0 --prog-alpha-init 0.8 \
    --muon-w 0.2 --sgd-w 1.0 --amp
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose --nms
```

- **尚无完整 400 轮 / 从零全长 pose 结果**；上表覆盖官方权重复现与 10 轮微调对齐。

## 4. 训练内部对齐

同一权重、同一真实 batch 下与官方实现逐项对照：

- **损失 / 梯度**：总损失差 <1%，梯度范数差 <2%（本仓库 2091 vs 官方 2130）。
- **o2o 分支**：detach 特征输入而非输出——梯度只经 o2m 回流 backbone/neck，o2o 头自身仍训练。
- **优化器**：MuSGD 分组、NS 迭代与系数与官方一致；梯度裁剪 `max_norm=10` 每步无条件执行；
  EMA 公式 `decay*(1-exp(-steps/tau))` 与更新时机一致；ProgLoss 每 epoch 更新。
- **增强**：mosaic 画布 1,638,400 px 0 差异、仿射输出 409,600 px 0 差异、标签 0 差异。
- **数据格式**：COCO-json 与 YOLO-txt 加载器在同一批图上框完全一致（0 差异），训练侧统计
  逐项相等（14205 / 14205）。

相关规则见 [../pipeline/train.md](../pipeline/train.md)、
[../pipeline/eval.md](../pipeline/eval.md) 与各模型页。

## 5. 运行时一致性

- **pt ↔ onnx**：raw 输出相对差 <5e-4，E2E 检出 <1e-3。
- **TensorRT**：fp32 引擎与 PyTorch 差 <2e-3；fp16 引擎与 PyTorch half 模型的 raw 输出逐位
  一致（两边都是 fp16 乘 + fp32 累加 + 终值舍入）。
- **分割双输出**：`output0`（检测 + 系数）与 `output1`（原型）在 pt / onnx / TRT 三后端契约
  一致；掩码后处理与官方 val 逐像素一致。

## 6. 可复现性

- 默认 `--deterministic`：关闭 cuDNN autotune、使用确定性 kernel、每个 epoch 重播增强 RNG；
  同 seed 两次训练权重**逐位相同**，resume 与不中断训练**逐位相同**。
- `FLASH_YOLO_REPRO=1` 门控 CPU/GPU 双跑与 resume 等价测试（实测 0 位差）。
- 训练产物 metadata 记录 arch/scale/nc/imgsz/names/anchors 与出处 commit，权重自包含、可审计。

## 7. 测试

```bash
pytest tests/ -q          # 核心套件；env-gated 套件默认跳过
pytest tests/test_weight_alignment.py -v   # 需要 dev deps + 转换后的官方权重
```

- **环境门控**：`FLASH_YOLO_SMOKE`（CPU 冒烟 / 训练小闭环）、`FLASH_YOLO_REPRO`
  （CPU/GPU 双跑 + resume 等价）、`FLASH_YOLO_DDP`（2 进程 CPU gloo）。
- **覆盖范围**：权重逐位对齐、导出 parity、评估器与指标正确性、训练组件（assigner /
  loss parity / MuSGD / EMA / checkpoint / augment / dataset / config / smoke / head-train）、
  v3（model / loss / darknet converter / engine / export 对拍）、分割（model / loss /
  dataset / metrics / validator / smoke / repro）、pose（model / loss / dataset / metrics /
  smoke / 导出分支回归）。
- CI：`.github/workflows/tests.yml` 在 push/PR 跑 ungated CPU 核心套件（不装 ultralytics）。

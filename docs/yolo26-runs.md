# YOLO26 训练与评估记录

> 数字口径：**正式** = `scripts/eval.py`（pycocotools，COCO val2017）；**训练中** = FastMetrics
> （纯 numpy 快速近似，只用于趋势与选 `best.safetensors`）。
> 结构事实与实现口径见 `yolo26-spec.md`。

## 1. 口径说明

- 官方 40.1（E2E）/ 40.9（NMS）= **Objects365v1 预训练 150 轮 + COCO 微调 245 轮**（n 档
  batch 128）的结果，**不是从零训练**——本页的从零数字与它不构成 like-for-like 对比。
- 本仓库自己那条"从零 100 轮"基线是给"100 轮预算能买到什么"这个问题一个诚实回答。

## 2. 官方权重验收（复现基线）

```bash
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco          # E2E
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms    # o2m + NMS
```

| 路径 | 输入 | mAP@[.5:.95] | mAP@50 |
|---|---|---|---|
| 官方权重 · 本管线 · E2E（NMS-free） | 640 | **40.27** | 55.80 |
| 官方权重 · 本管线 · NMS（o2m） | 640 | **40.89** | 56.87 |
| 官方权重 · 本管线（换输入尺寸参考） | 416 | 34.31 | 48.61 |
| 官方权重 · 本管线（换输入尺寸参考） | 512 | 37.49 | 52.59 |

- 官方发布 40.1 / 40.9 来自官方指标实现；~0.1 的差是**指标实现噪声**，不是模型差异。
- 这四行同时也是"逐位复现"的对外证据：同一份权重在本仓库管线里跑出与官方一致的数。
- **双实现交叉验证（2026-10-11 实测）**：同一 checkpoint 分别由本工程实现与 **ultralytics 实现**
  （`scripts/compare_official.py`，dev-time；模型来自 ultralytics，前后处理/指标全用本仓库）
  跑同一管线 —— E2E **40.27 / 55.80**（两者 P 0.6708 · R 0.5221）、NMS **40.89 / 56.87**
  （两者 P 0.6769 · R 0.5357），**逐项一致**。"实现可互换"由此有两层证据：输出逐位一致
  （`test_weight_alignment`）+ 端到端指标逐项一致（本行）。
  （历史注：旧文档记的 NMS mAP50 = 56.88 是当时 NMS 实现版本的测量值；早停版等同实现下今天两条
  实现都实测 56.87。）
- **早期延迟基准（2026-09，RTX 4060 Laptop / WSL2 CPU，推理段 20 次平均，已过时仅供溯源）**：
  本管线 E2E 20.4 ms GPU · 57.2 ms ONNX-CPU；NMS 16.5 ms GPU；官方实现同条件 18.6 ms（E2E）/
  14.6 ms（NMS）。当前口径见 README 的 TensorRT 表。

## 3. 从零训练记录

| run | 时间 | 配置（640 输入） | 状态 | 训练中 best | 正式 mAP@[.5:.95] / mAP@50 |
|---|---|---|---|---|---|
| `yolo26n-from-scratch` | 10-07 20:44 → 10-08 11:03 | 100 ep · batch 64 · default 配方 · 14 h 18 m | 完成 | 0.3581 / 0.5065 | **36.22 / 51.32**（E2E）· **37.15 / 52.67**（NMS） |
| （早期）误用 `coco-ft` 微调档的 100 轮 | 2026-10 早期 | 100 ep · batch 64 | 完成（**产物未留存**） | — | 32.33 / —（E2E）· 33.29 / —（NMS） |
| `yolo26n` | 10-08 14:36 | 100 ep 启动 | 第 1 轮内中断（无产物） | — | — |
| `abcur` / `abcurb` | 10-08 16:31 | coco-tiny · 1 ep · `--val-limit 100` | 开发冒烟 | — | — |
| `fy-s0-yolo26n` | 10-09 | 20 ep · batch 32（flash-yolo 筛选对照） | 完成 | 0.2922 | 见 `flash-yolo-spec.md` |

```bash
# 从零 100 轮（基线配方；约 14 h / RTX 5090）
python scripts/train.py --data coco --name yolo26n-from-scratch --batch 64 --workers 16
python scripts/eval.py --weights yolo26n-from-scratch-e100-640.safetensors --data coco
python scripts/eval.py --weights yolo26n-from-scratch-e100-640.safetensors --data coco --nms
```

**收敛观察**：100 轮最后 10 轮 FastMetrics 仍 **+0.0038 mAP / +0.0046 mAP50**——放缓但没平，
说明 100 轮还没吃干净基线配方。这也是"官方 245 轮 + O365 预训练"不可直接对比的直观证据。

## 4. 训练管线与官方的对齐

改训练代码前先跑 `tests/test_loss_parity.py`（以固定输入对官方 `E2ELoss` 逐项比对，rel<1e-4）。

- **配方**：内置 `default` = ultralytics `default.yaml`（lr0 0.01 / lrf 0.01 / mosaic 1.0 /
  scale 0.5 / hsv 0.015,0.7,0.4 / bgr 0），`config/train.yaml` 逐项一致。
- **损失**：ProgLoss（每 epoch 更新 alpha）+ TAL/STAL 分配；归一化 = 官方三件套
  （三项共用分母 Σtarget_scores，`clamp_(min=1)`）；**o2o 分支 detach 特征输入**（不是输出）。
- **优化器**：MuSGD（Muon 分组 + NS5 松弛正交化 + SGD 组），**梯度裁剪 max_norm=10 每步无条件**
  （AMP 顺序 unscale → clip → step）。EMA：`decay·(1-exp(-steps/tau))`，resume 必须续上 steps。
- **双实现对照（dev-time）**：同权重、同真实 batch，本实现 vs 官方实现——**总损失差 <1%、
  梯度范数差 <2%**（2091 vs 2130）；即 loss/assigner/归一化/梯度管线与官方一致。
- **增强管线**：逐像素对照过（mosaic 画布 1,638,400 px / 仿射 409,600 px / 标签均 0 差异）。

## 5. 经验教训（踩过的坑，别重复）

1. **配方档位不能混用**：`yolo26-coco-ft` 是"面向 O365 权重"的微调配方（lr0 0.0054、陡衰减），
   从零跑它 = 把模型按在慢速档上——实测从零跑它 20 轮才 0.2896，而同增强同调度的 default 档
   7 轮就有 0.2379（同轮次相对 +43%）。**从零训练只有 default 一档**。
2. **增强管线的 5 处修复（2026-10）**：从零 100 轮 mAP 明显低于官方时的复盘中找到并修复，
   修复前后同 seed 分布实测：每图保留框 9.5 → 15.4（旧实现丢掉 ~40% 标注框）、框长宽比
   p95 8.9 → 5.0、max 223 → 27.8（旧实现把碎片框当正样本）。修复后 20 轮筛选逐轮相对基线：
   ep1 +21% / ep2 +61% / ep3 +49%。
3. **梯度裁剪长期饱和**：`diag/train_diag.csv` 的 gnorm：随机初始化 ‖g‖≈2000-6000、100 轮
   收敛态仍有 ~70，而阈值是 10 → **实际步长 = lr×10×单位方向，即 ∝ lr**。所以任何调 lr 的
   实验（换档、warmup 改动）实际改的是总步长，别按"loss 增益"直觉推。
4. **不要给 o2o 分支额外加权重**：曾按配方文档给 o2o cls 乘 cls_w=2.74，ProgLoss 后期
   o2o cls P3 头出现 BN 掩蔽的复利增长（loss 全程正常而权重 1.8→1e10，~110 batch 内发散）。
   排查这类"loss 正常但权重爆"要盯**非 bias 权重的 |w|max**。

## 6. 为什么还差 4 个点

- 36.22（本仓库从零 100 轮）vs 40.27（官方权重本管线）：差距全部来自**训练侧**，不含实现
  差异——同一套代码跑官方权重就是 40.27，且 loss/梯度与官方双实现对照 <1%/<2%。
- 两个已知来源：**O365 预训练**（官方全部 checkpoint 都有）+ **轮数预算**（100 vs 245）。
- 100 轮未收敛（见第 3 节末 10 轮数据）支持"预算不足"这一半；O365 那一半本仓库无法提供
  （缺数据集与算力），因此不宣称"从零可复现 40.1"。

## 7. 备注

- `runs/train/yolo26n-from-scratch/` 是唯一保留完整产物的从零 run（diag / samples / 周期
  checkpoint / resume.pt 齐全），复盘与对照都用它。
- 早期"误用 coco-ft 档"的 100 轮 run 与增强修复前的基线 run 的产物未留存，本页记录的是
  当时写入 CLAUDE.md 的正式数字。

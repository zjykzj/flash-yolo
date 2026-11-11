# Flash-YOLO-pose（姿态估计）

> `flash-yolo` 的轻量 backbone/neck + `Pose26` 头；单档 `flash`，@640 **3.08M 参数 /
> 5.9 GFLOPs**。姿态头与 [yolo26-pose.md](yolo26-pose.md) 完全同构（关键点 + 可见性 +
> 训练期 RLE），检测/分割结构见 [flash-yolo.md](flash-yolo.md) /
> [flash-yolo-seg.md](flash-yolo-seg.md)。
> 当前状态：结构 / 训练 / 评估 / 导出 / 推理链路已接入；**尚无公开权重与正式指标**。

## 1. 结构

- backbone/neck 与 `flash-yolo` 逐行相同（25 行 yaml 的前 24 行），只把最后的 `Detect`
  换成 `Pose26`：`config/models/flash-yolo-pose.yaml`。
- `Pose26` = `Detect` + `cv4`（每级两个 3×3 Conv，`c4 = max(ch[0]//4, K·(D+2)) = 85`）
  + `cv4_kpts`（1×1 → 51，x/y/v 交错）+ `cv4_sigma`（训练期 RLE，2K=34）
  + `flow_model`（RealNVP，训练期 RLE）；`end2end=True` 时另有 `one2one_*` 深拷贝。
- 头输入通道 = `[64, 128, 256]`（P3/P4/P5）；模块树 **365 层**，其中头 1,416,840 参数。
- `cv4_sigma` / `flow_model` 只服务训练，不进 ONNX/TRT 的推理图。
- `kpt_shape = (17, 3)` 是结构参数，随权重 metadata 的 `kpt_shape` 走；自定义关键点数据集
  必须同时改描述符/权重 metadata，不能只改 `nc`。

## 2. 契约

- **E2E（默认）**：`(B, 300, 6 + nk)` = `[x1,y1,x2,y2,score,cls, kpts...]`；关键点与框共享
  同一次两阶段 top-k。COCO 17 点下 `nk=51`，即 `(B, 300, 57)`。
- **raw / NMS**：`(B, 4 + nc + nk, N)`，内容是**已解码**像素 xywh + sigmoid 分数 + 解码
  关键点；COCO 下 `(B, 56, 8400)`，NMS 在外。与检测头 raw 的 ltrb+logits 不同。
- 关键点解码 `(t + anchor) * stride`，v 通道 sigmoid；没有 v8-Pose 的 `2t - 0.5` 口径。
- 训练中验证同时算 Box + Pose；best fitness = box mAP50-95 + pose mAP50-95；正式数字仍以
  `scripts/eval.py` 为准。

## 3. 训练

从零训练用内置 `default` 配方（100 轮）——`yolo26-pose-coco` 是官方 pose 权重的 400 轮
微调配方，依赖 O365 预训练起点，不适合直接拿来从零训 flash 结构：

```bash
# 结构自检（无需数据）
python model/summary.py --model flash-yolo-pose

# 从零训练（COCO-pose；从零默认 fp32）
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --name flash-pose --batch 32 --workers 8 --no-amp

# 可选：从 flash-yolo 检测权重初始化 backbone/neck；pose 头随机初始化
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --weights flash-yolo-e300-640.safetensors \
    --name flash-pose-ft --batch 32 --workers 8 --amp
```

- 从检测权重初始化时，`cv4/cv4_kpts/cv4_sigma/flow_model` 在检测 ckpt 中不存在，
  `load_weights_finetune(..., allow_missing_head=True)` 会随机初始化它们，backbone/neck
  逐位加载（见 `tests/test_finetune_load.py`）。
- pose 下 `copy_paste > 0` 会直接报错；`fliplr` 使用描述符里的 `flip_idx`。
- 训练中 FastPoseMetrics 与 pycocotools 可差约 0.1 AP；训练日志只用于趋势与 best 选取，
  正式数字用 `scripts/eval.py`（与 [yolo26-pose.md](yolo26-pose.md) 同口径）。

## 4. 评估 / 导出 / 推理

```bash
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose --nms

python scripts/export.py --weights runs/train/<run>/weights/best.safetensors
python scripts/export.py --weights runs/train/<run>/weights/best.safetensors --raw
python scripts/infer.py --weights runs/train/<run>/weights/best.safetensors \
    --image assets/traffic.jpg --data coco-pose
```

- COCO-pose val 只评估 **2,346 张有有效关键点的图**（不是 5,000 张检测 val）；pycocotools
  OKS，maxDets=20。
- 导出 metadata 带 `arch=flash-yolo-pose / scale / nc / imgsz / names / kpt_shape / end2end`；
  infer/eval 不再需要原 `.safetensors`。
- `--nms` 支持（pose 的 o2m 分支）；`--single-cls` 对 pose 不适用。

## 5. 注意

- 只有 `flash` 单档，不要传 `--scale`。
- 官方 `yolo26n-pose` 的参考指标（0.5687 / 0.8250 E2E）**不适用于**本结构；本页不发布
  未完成的 flash-yolo-pose 训练数字，等有完整 run 后再补正式指标。
- 结构参数：3,081,432 参数、5.9 GFLOPs @640；与 `flash-yolo` 检测版共用前 24 行结构。
- 训练中 `val_rle` 恒 0（验证前向不返回 sigma），不代表 RLE 未训练。

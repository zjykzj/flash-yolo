# YOLO26-seg（实例分割）

> 检测版的结构、损失与训练管线见 [yolo26.md](yolo26.md)；本文只写分割差异。
> 结构定义：`config/models/yolo26-seg.yaml`。

## 1. 结构

backbone/neck 与检测版**逐行相同**（官方 yaml diff 的唯一一行是最后一行），头换成
`Segment26` = `Detect` + 两个分割专属件：

| 件 | 内容 |
|---|---|
| `cv4` / `one2one_cv4` | 每级 3×3 conv stack → `nm=32` mask 系数（`c4 = max(ch[0]//4, 32)`） |
| `proto`（Proto26） | P4/P5 → 1×1 投到 P3 通道 → nearest 上采样 ×2/×4 相加 → 3×3 → ConvTranspose2d ×2 → 3×3 → 1×1，输出 `(B, 32, H/4, W/4)` |
| `proto.semseg` | 训练期辅助语义头：BCE + 多通道 Dice，`loss[4] *= box_gain`；验证/导出不计算（零成本） |

- `npr`（yaml 256）随 width 缩放：n 档 64；`nm=32` 是任务常数，不缩放。
- n 档合计 **3,126,280 参数 / 309 层 / 9.3 GFLOPs @640**。
- o2o 分支的 proto 走 `detach()`（梯度只经 o2m 回流）。

## 2. 推理契约

- E2E：`(B, 300, 6+nm)` 检测+系数 + `(B, nm, H/4, W/4)` 原型。
- raw：`(B, 4+nc+nm, N)` + 原型。
- E2E 的两阶段 top-k 与检测共用同一实现（系数按同一组 anchor 索引取回）。
- 掩码解码 `utils/postprocess.process_mask`：系数 @ 原型 → bilinear 上采样到网络输入尺寸 →
  `>0` 二值化 → 按框裁剪 → 反 letterbox 重采样到原图 → `>0.5` 再二值化（与官方
  `process_mask(upsample=True)` + `scale_masks` 两段口径一致）。

## 3. 训练

- `ComputeLossSeg` 复用检测双分支损失（box/cls/l1 + ProgLoss），每个分支额外计算掩码项：
  系数 @ 原型 → BCE（按责任 GT 框裁剪、按归一化框面积缩放，全 batch 求和 / 正样本数），
  掩码增益 = `box_gain`。
- 数据侧接受 COCO 多边形/RLE 与 YOLO-seg 多边形行（见
  [../datasets/coco.md](../datasets/coco.md) / [../datasets/yolo.md](../datasets/yolo.md)）。
- `mask_ratio`（默认 4）把 GT 掩码降采样到 `imgsz/mask_ratio`（640 → 160，与 Proto26 同网格）；
  `overlap_mask=True` 生成面积降序实例索引图。
- 官方 seg 配方：`yolo26-seg-coco-ft`（`mask_ratio=1` / `overlap_mask=true` /
  `copy_paste_mode=flip`），需要 `yolo26*-objv1-seg` 365 类初始化。
- 验证同时算 Box + Mask；best fitness = box mAP50-95 + mask mAP50-95；正式数字仍以
  `scripts/eval.py` 为准。

## 4. 权重下载与初始化

下载链接固定到 **v1.0.0**（下一个发布版本；发布前会 404，可先用 Releases 页面的当前版本）。
资产名固定，发新版时统一替换 tag 即可。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `yolo26n-seg-from-scratch-e100-640` | 100 ep 从零 | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolo26n-seg-from-scratch-e100-640.safetensors) |

本轮 300 轮从零产物只在本地（`runs/train/train-v1-yolo26n-seg-e3002/weights/best.safetensors`），
尚未随 Release 分发。

官方权重不随 Release 分发，需自行下载：

```bash
# COCO 80 类官方权重（直接评估/推理）
python scripts/download_weights.py --model yolo26n-seg
python scripts/convert_weights.py --src weights/yolo26n-seg.pt --dst weights/yolo26n-seg.safetensors

# O365 365 类分割初始化（官方 COCO 阶段微调用）
python scripts/download_weights.py --model yolo26n-objv1-seg
python scripts/convert_weights.py --src weights/yolo26n-objv1-seg.pt \
    --dst weights/yolo26n-objv1-seg.safetensors
```

官方 O365 seg 初始化是 **365 类**，COCO 训练建 80 类头。`--weights` 走
`load_weights_finetune` 的 key+shape 交集加载：

- backbone/neck/proto/cv2 逐位加载；
- `cv3/one2one_cv3/semseg` 等类别相关头按目标 nc 重建；
- n 档实测 740/844 张量加载、104 个类别头重初始化；
- key 不匹配 / 换错架构 / 头外 shape 不匹配直接报错。

## 5. 正式指标（COCO val2017 @640，Box + Mask）

| 训练 | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 |
|---|---:|---:|---:|---:|
| 官方权重（O365 + COCO） | **39.67** | 55.48 | **33.67** | 52.76 |
| 从零 100 ep | 35.69 | 51.03 | 30.52 | 48.61 |
| 从零 300 ep | **37.90** | **53.69** | **32.37** | **51.16** |

- 官方训练期 validator 记录 box 39.75 / mask 33.86；本管线 39.67 / 33.67，差 ~0.002
  属指标实现噪声。
- 100/300 轮都是预算点（同一 default 配方，batch 16 / 32、nbs 64）；300 轮末段斜率已明显
  放缓（最后 10 轮 +0.2 box / +0.3 mask），best 在 ep295。与 Flash-YOLO-seg 的同预算对照见
  [flash-yolo-seg.md](flash-yolo-seg.md)。
- 完整官方 245 轮尚未跑：单卡 `mask_ratio=1` 约 5.2 img/s，DDP 可扩但数据管线先饱和。

## 6. 推理延迟

> 口径：**model-only**（不含掩码解码/前后处理），输入 640；b1 = 单图，b8 = batch 8 每图平均。
> 引擎 init 已做 warmup。分割模型尚未做完整的双平台基准，下表待补测。

### 云服务器 / 工作站

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 待测 | 待测 |
| RTX 5090 | TensorRT 11.4 · fp16 | 待测 | 待测 |
| Xeon 8470Q（25 核配额） | ONNX Runtime 1.30 · fp32 | 待测 | 待测 |

### 本地笔记本

| 平台 | 运行时 / 精度 | b1 | b8（每图） |
|---|---|---:|---:|
| （CPU 型号待填） | ONNX Runtime · fp32 | 待测 | 待测 |

> 掩码解码与 IoU 目前主要在 CPU；部署时的端到端延迟显著高于 model-only，需单独测量。

## 7. 用法

```bash
python scripts/train.py --data coco --model yolo26-seg --name seg-coco --batch 32 --workers 8
python scripts/train.py --data coco --recipe yolo26-seg-coco-ft \
    --weights weights/yolo26n-objv1-seg.safetensors --batch 32 --workers 8

python scripts/eval.py --weights runs/train/train-seg-coco/weights/best.safetensors --data coco
python scripts/export.py --weights runs/train/train-seg-coco/weights/best.safetensors --trt
python scripts/infer.py --weights runs/train/train-seg-coco/weights/best.safetensors \
    --image assets/traffic.jpg
```

## 8. 注意

- 掩码解码 + IoU 目前在 CPU（训练验证的 `process_mask_lowres` 自动分块）；GPU 上 `process_mask`
  会使用引擎设备，逐位一致。
- `mask_ratio=1 @640` 的掩码损失走按 GT 框切片实现，显存显著低于整图 gather。
- `val_sem` 在验证模式恒 0（官方同口径），不代表 semseg 未训练。

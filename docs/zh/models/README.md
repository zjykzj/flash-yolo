# 模型（Models）

本文给出模型矩阵与所有架构共享的约定；各架构的细节见对应页面。

## 1. 模型矩阵

| 架构 | 任务 | 档位 | Params | GFLOPs | 文档 |
|---|---|---|---:|---:|---|
| YOLO26 | 检测 | n / s / m / l / x | 2.57M (n) | 5.48 (n) | [yolo26.md](yolo26.md) |
| YOLO26-seg | 实例分割 | n / s / m / l / x | 3.13M (n) | 9.3 (n) | [yolo26-seg.md](yolo26-seg.md) |
| Flash-YOLO | 检测 | flash（单档） | 1.97M | 3.80 | [flash-yolo.md](flash-yolo.md) |
| Flash-YOLO-seg | 实例分割 | flash（单档） | 2.53M | 7.6 | [flash-yolo-seg.md](flash-yolo-seg.md) |
| YOLOv3-tiny / -SiLU | 检测 | tiny（单档） | 8.85M | 5.56 @416 | [yolov3-tiny.md](yolov3-tiny.md) |
| YOLOE-26 | 开放词汇分割（pf / 文本 / 视觉提示词） | n / s / m / l / x | 7.84M (n) | 16.0 (n) | [yoloe.md](yoloe.md) |

选择方式：

```bash
python scripts/train.py --data coco --model yolo26 --scale n
python scripts/train.py --data coco --model yolo26-seg --scale n
python scripts/train.py --data coco --model flash-yolo
python scripts/train.py --data coco --model flash-yolo-seg
python scripts/train.py --data coco --model yolov3-tiny --imgsz 416
```

- `yolo26` / `yolo26-seg` / `yoloe-26` 有 `n/s/m/l/x` 档位；`flash-yolo` / `flash-yolo-seg` 只有
  `flash`，`yolov3-tiny` 只有 `tiny`——对这些架构传 `--scale` 会告警且不生效。
- `yoloe-26` 支持 prompt-free（-pf）与文本 prompt 权重；`--raw` 不适用，`--nms` 仅支持
  文本模式（o2m + NMS；pf 仅 E2E，见 yoloe.md）。文本模式另需 `mobileclip2_b.safetensors`
  （Apple MobileCLIP2-B 文本编码器的转换产物）与模型权重同目录。
- `yoloe-26` 训练支持 PE 线性探测（`--pe --weights <文本权重>`：只训练 cv3 尾卷积，
  backbone/BN/掩码分支冻结；产物类锁定，可直接评估/导出）与全量微调（`--full-finetune`：
  全部参数训练，reprta 在图内；产物未 fused，评估需同一数据集描述符）。从零训练尚未接入。
- 结构定义在 `config/models/<model>.yaml`；`--cfg` 可指向自定义 yaml（自定义变体）。
- 自训权重的下载链接指向**最新发布**资产（v1.0.0 的 checkpoint）；YOLOE 官方权重见
  yoloe.md（Ultralytics assets），文本编码器的转换产物 `mobileclip2_b.safetensors` 随
  Releases 提供（与模型权重同目录，见 yoloe.md）。

## 2. 公共结构约定

### state_dict 与 strict load

- key 是**属性路径**（如 `model.23.one2one_cv2.0.0.conv.weight`），不是类名。
- 属性树必须与官方布局逐位对齐；重命名模块属性会破坏零映射 strict load，对齐测试会拦截。
- 官方 YOLO26 ckpt 的 `ckpt["model"]` 是完整模块对象（`ema` 为 None），
  `scripts/convert_weights.py` 解包一次写成纯 safetensors；运行时不依赖 ultralytics。

### BN 与数值口径

- BN **eps = 0.001**（官方非默认值；PyTorch 默认 1e-5 会让第 0 层起激活分叉）。
- BN momentum = 0.03（只影响 running stats，不影响 train 前向）。
- 推理/导出链路的 letterbox、BGR→RGB、/255 与官方 val 一致。

### scale 规则（YOLO26 系）

| scale | depth | width | max_channels |
|---|---|---|---|
| n | 0.50 | 0.25 | 1024 |
| s | 0.50 | 0.50 | 1024 |
| m | 0.50 | 1.00 | 512 |
| l | 1.00 | 1.00 | 512 |
| x | 1.00 | 1.50 | 512 |

- `repeats = max(round(repeats * depth), 1)`；
- `c2 = make_divisible(min(c2, max_ch) * width, 8)`；
- m/l/x 档所有 C3k2 强制 `c3k=True`（yaml 里的 False 只对 n/s 生效）。

### 权重 metadata

训练保存与转换器都会往 safetensors header 写可选 JSON：

- 模型参数：`arch` / `scale` / `nc` / `imgsz` / `names`，含锚模型另有 `anchors`；
- 出处：`version` / `repo` / `commit` / `describe`（best-effort，读不到不写该键）。

读取顺序：**CLI 显式 > metadata > 文件名/默认**。所以 `best.safetensors` 这类文件名推不出
档位的产物免 `--model/--scale`；自定义数据集产物免 `--data`（nc/names 随盘）。
例外：onnx 输入尺寸以图内 shape 为准（显式 `--imgsz` 冲突会报错）。

### E2E 与 NMS

- **E2E（默认）**：o2o 头、无 NMS，输出 `(B, 300, 6)`；官方 40.1 走这条路径。
- **NMS**：o2m 头 + 类感知 NMS，通常高 ~0.9 点、略慢；`--nms` 启用。
- YOLOv3-tiny 只有一条“解码 + obj×cls + NMS”路径，`--nms` / `--raw` 对它直接报错。

### anchors（仅 YOLOv3-tiny）

- 锚值是 **yaml 数据**（darknet 固定像素值），绝不随 `imgsz` 缩放；官方权重在多尺度训练下
  由网络自补偿输入尺度。
- `anchors` / `stride` 是 `persistent=False` buffer：不进 state_dict，随 `.to(device)` 搬运。
- 从零训练换输入尺寸才需要重聚类：`python scripts/compute_anchors.py --data <desc>`。

## 3. 官方 vs 从零

- 官方 YOLO26 / YOLO26-seg 权重是 **Objects365v1 预训练 + COCO 微调**，不是从零训练；
  把从零数字和官方数字放在一起比较不是 like-for-like。
- 各模型页给出最终结果（300 轮从零）、权重下载与延迟；跨模型的同预算对照见根 README 的 Results。
- 正式指标一律以 `scripts/eval.py`（pycocotools）为准；训练中的 FastMetrics 只用于趋势与选 best。

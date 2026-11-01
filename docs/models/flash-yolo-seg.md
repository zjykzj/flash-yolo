# Flash-YOLO-seg（实例分割）

> `flash-yolo` 的实例分割变体（`arch_task = segment`）。backbone/neck 与检测版**逐行相同**，
> 只把最后一行 `Detect` 换成 `Segment26`——与 `yolo26 → yolo26-seg` 是同一种关系。
> 结构定义：`config/models/flash-yolo-seg.yaml`；共享结构见
> [flash-yolo.md](flash-yolo.md)。

## 1. 命名与档位

- 家族名在前、任务后缀在后：`flash-yolo-seg`（不是 `flash-yolo-seg-n`）。
- 单档 `flash`：通道已材料化，没有 n/s/m/l/x 维度；`--scale` 不适用。
- 权重文件名可反推架构（`flash-yolo-seg.safetensors`），长名正则优先，不会被
  `flash-yolo` 前缀误吸。

## 2. 结构

25 行 yaml，与 `flash-yolo.yaml` 的差异**只有最后一行**：

```yaml
- [[17, 20, 23], 1, Segment26, [nc, 32, 64]]
```

`Segment26` = `Detect` + `cv4` 掩码系数分支 + `Proto26` 原型分支 + 训练期 semseg 辅助头
（与 yolo26-seg 共用一个实现）。三个检测层仍是 P3/P4/P5（idx 17/20/23）。

- `npr = 64`：`yolo26-seg.yaml` 写 256，n 档 `width=0.25` 经 `make_divisible(256*0.25,8)`
  得 64；flash 档 `width=1.0`，所以直接写生效值 64（写 256 会得到 4.68M / 47.8 GFLOPs 的错误结构）。
- `nm = 32` 是任务常数，不缩放。

| 指标 | flash-yolo | flash-yolo-seg | yolo26n-seg |
|---|---:|---:|---:|
| 参数 | 1,974,248 | **2,528,248** | 3,126,280 |
| GFLOPs @640 | 3.80 | **7.6** | 9.3 |
| 头参数 | 309,656 | 863,656 | 863,656 |

- backbone/neck 逐键同构（测试 `test_backbone_neck_identical_to_detect`）。
- 与 yolo26n-seg 的差异只省在 backbone/neck（−598,032 参数、−1.7 GFLOPs）；分割头逐键相同。
- GFLOPs 相对检测版约翻倍来自 Proto26 原型分支（官方 seg 同样如此）。

## 3. 复用的链路

| 环节 | 复用方式 |
|---|---|
| 前向契约 | E2E `(B,300,6+nm)` + proto `(B,32,160,160)`；raw `(B,4+nc+nm,N)` + proto |
| 训练模式 | 双分支 dict（`one2many`/`one2one`，o2o 的 proto/semantic 均 detach） |
| 损失 | `build_loss` 分派到 `ComputeLossSeg`（检测三项/归一化/ProgLoss 复用 + 掩码 BCE） |
| 验证 | 训练中 Box+Mask，best fitness = box + mask mAP50-95；正式口径走 `scripts/eval.py` |
| 权重 | metadata 写 `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz` |
| 引擎 | pt / onnx / trt 双输出（output0 检测+系数、output1 原型） |

除 `ARCHS`、损失分派表与权重名正则外，没有新增分支。

## 4. 权重下载

下载链接固定到 **v1.0.0**（下一个发布版本；发布前会 404，可先用 Releases 页面的当前版本）。
资产名固定，发新版时统一替换 tag 即可。

| 权重 | 训练 | 输入 | 下载 |
|---|---|---|---|
| `flash-yolo-seg-e100-640` | 100 ep 从零 | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/flash-yolo-seg-e100-640.safetensors) |

- 本地 100 轮产物：`runs/train/train-flash-seg-100ep2/weights/best.safetensors`。
- 本地 300 轮产物：`runs/train/train-v1-flash-yolo-seg-e300/weights/best.safetensors`
  （未随 Release 分发）。
- 权重 metadata 自带 `arch=flash-yolo-seg` / `scale=flash` / `nc` / `names` / `imgsz`，
  eval/infer/export 可自动解析。

## 5. 正式指标（COCO val2017 @640）

两个从零预算点：100 轮（default 配方、batch 16、nbs 64、fp32、mask_ratio 4、seed 0）与
300 轮（同配方、batch 32、nbs 64）：

| 训练 | 模型 | Params | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 100 ep | Flash-YOLO-seg | 2,528,248 | 7.6 | 47.96 | **33.11** | 45.34 | **28.24** |
| 100 ep | YOLO26n-seg（同预算） | 3,126,280 | 9.3 | 51.03 | 35.69 | 48.61 | 30.52 |
| 100 ep | Δ（相对） | −19.1% | −18.3% | −6.0% | **−7.2%** | −6.7% | **−7.5%** |
| 300 ep | Flash-YOLO-seg | 2,528,248 | 7.6 | 50.74 | **35.05** | 48.14 | **29.98** |
| 300 ep | YOLO26n-seg（同预算） | 3,126,280 | 9.3 | 53.69 | 37.90 | 51.16 | 32.37 |
| 300 ep | Δ（相对） | −19.1% | −18.3% | −5.5% | **−7.5%** | −5.9% | **−7.4%** |

- 架构代价与检测版同量级（检测 300 轮 −7.1%），说明 PoolConv/DW stem 的取舍不随任务改变。
- 100 轮 best 在末轮（ep100），曲线仍在上升；300 轮 best 也在末轮（ep300），但末段已放缓
  （最后 10 轮 +0.2 box / +0.1 mask）——两者都是预算点，不是收敛上限。
- 不要与官方 O365 初始化权重（Box 39.67 / Mask 33.67）直接横比；同起点对照是上面同预算行。

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
python model/summary.py --model flash-yolo-seg
# -> 311 layers, 2,528,248 parameters, 7.6 GFLOPs

python scripts/train.py --data coco --model flash-yolo-seg --name flash-seg \
    --batch 32 --workers 8 --mask-ratio 4
python scripts/eval.py --weights runs/train/train-flash-seg/weights/best.safetensors --data coco
python scripts/export.py --weights runs/train/train-flash-seg/weights/best.safetensors --trt
python scripts/infer.py --weights runs/train/train-flash-seg/weights/best.safetensors \
    --image assets/traffic.jpg
```

## 8. 注意

- 单档 `flash`，不要传 `--scale`。
- 分割训练由 loader 与验证主导，架构差异在吞吐上不可辨。
- 跨设备（CUDA vs CPU oneDNN）对齐会有 1e-3~4e-3 的原始输出差，掩码边界可能翻转；
  同运行时比较或走 TensorRT。

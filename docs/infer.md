# 推理（Infer）

> 入口：`python scripts/infer.py --weights <w> --image <文件|目录>`。
> 三种运行时（pt / onnx / trt）同一套前后处理，输出可视化图 + YOLO txt 标签。

## 1. 用法

```bash
# 单图（E2E，默认 NMS-free）
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/traffic.jpg

# 目录批量
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/

# 换路径 / 换引擎 / 换解码路径
python scripts/infer.py --weights runs/train/my-run/weights/best.safetensors --image img.jpg --nms
python scripts/infer.py --weights weights/yolo26n.onnx --engine onnx --image img.jpg
python scripts/infer.py --weights weights/yolo26n.engine --engine trt --image img.jpg

# 非 COCO 数据训练的模型：用描述符带出 nc 与标签名
python scripts/infer.py --weights runs/train/my-run/weights/best.safetensors \
    --data mydataset --image img.jpg
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--image` | — | 图片文件或目录 |
| `--engine` | `pt` | `pt` / `onnx` / `trt`（trt 需已构建的 `.engine`） |
| `--conf` | **0.25** | 置信度阈值（注意：**eval 的默认是 0.001**，两者用途不同） |
| `--nms` | off | o2m + NMS 路径（yolo26 专属；v3 会报错） |
| `--model` / `--scale` | 从文件名推断 | 架构/档位（推不出时报错要求显式给） |
| `--output` | `runs/predict/predictN/` | 单图：输出文件路径；目录模式：输出目录 |
| `--data` | COCO 标签 | 只用 `names` + `nc`（不读图片） |
| `--imgsz` | 权重 metadata > 640 | 覆盖输入尺寸 |
| `--device` | auto | pt 引擎设备 |

## 2. 输出

```text
image 1/1 assets/traffic.jpg: 640x427 4 persons, 17 cars, 1 motorcycle, 1 bus, 1 truck, 15.0ms
Speed: 5.9ms preprocess, 8.9ms inference, 0.2ms postprocess per image at shape (1, 3, 640, 640)
Total: 25.8ms end-to-end for 1 image(s) (incl. image read + save)
Visualization -> runs/predict/predict21
Labels        -> runs/predict/predict21/labels
```

- 每图一行：原图尺寸 + 类别计数 + 纯推理时间；`Speed` 行是 preprocess / inference / postprocess
  三段（`predict_timed` 口径，引擎 init 时已做 3 次 warmup，数字是稳态值）；`Total` 含读图与存图。
- 产物目录（`runs/predict/` 自动递增）：`<图名>.jpg` 可视化 + `labels/<图名>.txt`
  （YOLO 格式：`cls xc yc w h`，归一化到原图）。
- 模型行：`YOLO26n · 260 layers · 2,572,280 params · E2E (NMS-free) · engine pt`
  （onnx/trt 显示的是部署口径：文件大小 + 输入/输出 shape）。

## 3. 引擎选择

| 引擎 | 何时用 | 注意 |
|---|---|---|
| `pt` | 默认、对齐验收 | GPU/CPU 均可；模型行给模块树口径 |
| `onnx` | CPU 部署（ORT） | 输入尺寸以图内 shape 为准 |
| `trt` | GPU 部署 | 引擎绑定构建机 GPU + TRT 版本；换机器要重建（见 `export.md`） |

## 4. 坑

- **conf 默认值与 eval 不同**（0.25 vs 0.001）：推理图省事，评估要对齐官方口径。
- v3 只有一条"解码 + obj×cls + NMS"路径，`--nms` / `--raw` 对它直接报错。
- 目录模式**只扫一层**（不递归，按文件名排序）；读不了的图会告警跳过，不中断批量。
- 类别名优先级：`--data`/权重 metadata 的 names > COCO 默认；名字对不上时可视化标的是序号。

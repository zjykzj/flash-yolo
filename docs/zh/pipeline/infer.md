# 推理（Infer）

> 入口：`python scripts/infer.py --weights <w> --image <文件|目录>`，或
> `--video <文件|URL|相机索引>`。三种运行时（pt / onnx / trt）同一套前后处理；图片模式
> 输出可视化图 + YOLO txt 标签，视频模式输出带标注的视频文件。

## 1. 用法

```bash
# 单图（E2E，默认 NMS-free）
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/traffic.jpg

# 目录批量（只扫一层，按文件名排序）
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/

# 换路径 / 换引擎 / 换解码路径
python scripts/infer.py --weights yolo26n.safetensors --image img.jpg --nms
python scripts/infer.py --weights weights/yolo26n.onnx --engine onnx --image img.jpg
python scripts/infer.py --weights weights/yolo26n.engine --engine trt --image img.jpg

# 非 COCO 数据训练的模型：用描述符带出 nc 与标签名
python scripts/infer.py --weights <自训权重>.safetensors --data mydataset --image img.jpg

# 分割权重：自动识别 arch，可视化叠加实例掩码
python scripts/infer.py --weights weights/yolo26n-seg.safetensors --image assets/traffic.jpg

# yoloe prompt-free：标签名 = 权重 metadata 里烘焙的 4,585 名词汇
python scripts/infer.py --weights weights/yoloe-26n-seg-pf.safetensors --image assets/traffic.jpg
# yoloe 文本 prompt：pt 经 MobileCLIP2-B 编码；onnx/trt 消费类锁定图
python scripts/infer.py --weights weights/yoloe-26n-seg.safetensors --image assets/traffic.jpg \
    --classes "person, bicycle, car, bus, truck"

# 视频文件（输出 <run_dir>/<源名>.mp4，默认 runs/predict/predictN/）
python scripts/infer.py --weights weights/yolo26n.safetensors --video /path/to/video.mp4
# 相机 / RTSP 流：用 --max-frames 限流（0 = 直到 EOF；Ctrl-C 干净退出）
python scripts/infer.py --weights weights/yolo26n.safetensors --video 0 --max-frames 300
python scripts/infer.py --weights weights/yolo26n.safetensors --video rtsp://host/stream --max-frames 600
# 窗口预览（需要 GUI；按 q 退出）+ 指定输出文件
python scripts/infer.py --weights weights/yolo26n.safetensors --video /path/to/video.mp4 \
    --show --output out.avi
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--image` | — | 图片文件或目录（与 `--video` 二选一） |
| `--video` | — | 视频文件 / 流 URL / 相机索引（`0`） |
| `--max-frames` | 0 | 视频模式最多处理 N 帧（流式源建议显式给） |
| `--show` | off | 视频实时预览（需要 GUI） |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--conf` | **0.25** | 置信度阈值（eval 默认 0.001，两者用途不同） |
| `--nms` | off | o2m + NMS（yolo26 系 + yoloe-26 文本模式；v3 报错，pf 仅 E2E） |
| `--agnostic-nms` | off | 跨类 NMS（需同时给 `--nms`） |
| `--classes` | — | yoloe-26 文本模式：逗号分隔的类别名（pt 必填；onnx/trt 可省略，需与锁定列表一致） |
| `--model` / `--scale` | 从文件名/metadata 推断 | 推不出时报错 |
| `--output` | `runs/predict/predictN/` | 图片：文件或目录；视频：输出视频文件 |
| `--data` | COCO 标签 | 只用 `names` + `nc`；不读图片（yoloe 忽略） |
| `--imgsz` | metadata > 640 | 覆盖输入尺寸 |
| `--device` | auto | pt 引擎设备 |

## 2. 输出

```text
image 1/1 assets/traffic.jpg: 640x427 4 persons, 17 cars, 1 motorcycle, 1 bus, 1 truck, 15.0ms
Speed: 5.9ms preprocess, 8.9ms inference, 0.2ms postprocess per image at shape (1, 3, 640, 640)
Total: 25.8ms end-to-end for 1 image(s) (incl. image read + save)
Visualization -> runs/predict/predict21
Labels        -> runs/predict/predict21/labels
```

- 每图一行：原图尺寸 + 类别计数 + 纯推理时间；`Speed` 是 preprocess / inference /
  postprocess 三段（引擎 init 已做 3 次 warmup，数字是稳态值）；`Total` 含读图与存图。
- 图片产物：`<图名>.jpg` 可视化 + `labels/<图名>.txt`（YOLO 格式 `cls xc yc w h`，归一化）。
- 分割权重：可视化叠加半透明实例掩码；`labels/*.txt` 写**归一化多边形行**
  `cls x1 y1 x2 y2 ...`（外轮廓、多连通域按最近点合并、少于 3 点跳过，与官方 `save_txt`
  同口径）。
- 视频模式**不写 `labels/`**（逐帧 txt 没有稳定命名语义）；编码按后缀选：`.avi`/`.mjpg`
  用 MJPG，其余用 mp4v；写不了时报错并提示换 `.avi`。
- 相机/流的输出尺寸在**首帧**确定；`--show` 在无显示服务器上会告警并自动关闭，不影响写入。

## 3. 引擎选择

| 引擎 | 何时用 | 注意 |
|---|---|---|
| `pt` | 默认、对齐验收 | GPU/CPU 均可；模型行给模块树口径 |
| `onnx` | CPU 部署（ORT） | 输入尺寸以图内 shape 为准 |
| `trt` | GPU 部署 | 引擎绑定构建机 GPU + TRT 版本；换机要重建 |

## 4. YOLOE（开放词汇）

- **prompt-free（`-pf` 权重）**：标签名 = 权重 metadata 里烘焙的词汇名，不需要
  `--data` / `--classes`；三种引擎都可跑。
- **文本 prompt（`-seg` 权重）**：
  - `--engine pt`：`--classes "a, b, c"` 把每个名字经 MobileCLIP2-B 文本编码器编码
    （`mobileclip2_b.safetensors` 需与模型权重同目录——只在 `set_classes` 时加载一次，
    不在模型图里）。
  - `--engine onnx|trt`：消费 `export --classes` 产出的类锁定图，类别名随部署 metadata
    走，无需文本编码器；`--classes` 可省略，给出时必须与锁定列表逐项一致。
- `--raw` 对 yoloe 直接报错；`--nms` 支持 yoloe-26 文本模式（o2m + 类感知 NMS），
  pf（lrpc）仍只有 E2E（原始输出列数随提案过滤变化）。
- 模型细节：[../models/yoloe.md](../models/yoloe.md)。

## 5. 坑

- **conf 默认值与 eval 不同**（0.25 vs 0.001）。
- v3 只有一条“解码 + obj×cls + NMS”路径，`--nms` / `--raw` 直接报错。
- 目录模式只扫一层（不递归）；读不了的图告警跳过，不中断批量。
- **`--output` 的图片语义**：单图给图片后缀（或已存在文件）按文件处理，文件名生效；
  给目录 / 无后缀 / 非图片后缀按目录处理，文件名沿用源图名。目录输入只接受目录输出。
- 类别名优先级：`--data` / 权重 metadata 的 names > COCO 默认；名字对不上时标序号。
- 视频输出尺寸在首帧确定；流式源建议 `--max-frames`，Ctrl-C 在 finally 里释放资源。

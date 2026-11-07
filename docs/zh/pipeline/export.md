# 导出与部署（Export）

> 入口：`python scripts/export.py --weights <w.safetensors> [...]`。
> 一条链：safetensors → onnx →（可选）TensorRT engine。三个运行时共用同一个 `predict()` 契约。

## 1. pt → onnx

```bash
python scripts/export.py --weights weights/yolo26n.safetensors              # 默认 E2E 融图
python scripts/export.py --weights weights/yolo26n.safetensors --raw        # 原始头输出
python scripts/export.py --weights weights/yolo26n.safetensors --dynamic    # 动态 batch
python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny

# yoloe prompt-free：E2E 图，conf=0 静态提案口径（官方同款）
python scripts/export.py --weights weights/yoloe-26n-seg-pf.safetensors
# yoloe 文本模式：prompt 融合进分类卷积——单输入、类锁定 ONNX
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car"
# yoloe 文本模式 Non-E2E：导出 o2m 密集输出，推理/评估配 --nms
python scripts/export.py --weights weights/yoloe-26n-seg.safetensors \
    --classes "person, bicycle, car" --nms
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` |
| `--out` | `runs/export/<stem>.onnx` | 输出路径 |
| `--raw` | off | 导出原始头 `(B, 4+nc, N)`（yolo26 专属；v3 报错） |
| `--nms` | off | 导出 o2m 密集输出（yoloe-26 文本专属；推理/评估需配 `--nms`） |
| `--dynamic` | off | 动态 batch 轴；默认固定 batch 1 |
| `--classes` | — | yoloe-26 文本模式：锁定进导出模型的类别名（必填） |
| `--model` / `--scale` | 从文件名推断 | 架构/档位 |
| `--nc` | metadata > 模型 yaml | 非 COCO 模型改头 |
| `--imgsz` | metadata > 640 | 导出输入尺寸 |
| `--opset` | 脚本默认 | ONNX opset |
| `--trt` | off | 同时构建 TensorRT 引擎 |
| `--fp16` | off | 半精度 onnx + strongly-typed engine（需 `--trt`） |

**导出图口径**

| 路径 | 输出 | 说明 |
|---|---|---|
| yolo26 默认（E2E） | `(B, 300, 6)` = [x1,y1,x2,y2,conf,cls] | 解码 + 两阶段 top-k 在图内 |
| yolo26 `--raw` | `(B, 4+nc, N)` | 解码/后处理在外 |
| yolo26-seg / flash-yolo-seg | `output0 (B, 300, 6+nm)` + `output1 (B, nm, H/4, W/4)` | 系数接末维；原型单独一路 |
| seg `--raw` | `output0 (B, 4+nc+nm, N)` + `output1` | 掩码由 `process_mask` 解码 |
| yolov3-tiny | `(B, NA, 5+nc)` | NA = 3·ΣH·W；obj×cls 与 NMS 在外 |
| yoloe pf | `output0 (B, 1000, 6+nm)` + `output1 (B, nm, H/4, W/4)` | E2E，**max_det=1000**，conf=0 静态提案口径 |
| yoloe 文本（`--classes`） | `output0 (B, 300, 6+nm)` + `output1` | 单输入：文本嵌入已融合进分类卷积（BN 折叠、reprta 移除）；导出模型拒收新 prompt |
| yoloe 文本 `--nms` | `output0 (B, 4+V+nm, N)` + `output1` | 密集输出已是像素 xyxy + sigmoid；NMS 在外（`nms_per_image`） |

- 训练产物自带 metadata，`--scale` / `--nc` 可省略。导出产物也带部署 metadata：ONNX 写入
  `metadata_props`（arch/scale/nc/imgsz/names/prompt_mode/end2end），TRT 写旁车
  `<engine>.json`——类锁定的 yoloe text 图因此无需原 `.safetensors` 即可
  `infer/eval --engine onnx|trt`；旧的无 metadata 产物仍需 `--model` / `--scale`。
- yoloe 文本模式导出会先跑完整 prompt 链（tokenize → MobileCLIP2-B → reprta+L2——
  `mobileclip2_b.safetensors` 需与模型权重同目录），再把结果融合进分类卷积，导出的 onnx
  无文本编码器依赖。fuse 等价性已验证：fused pt vs 未 fuse 1.8e-6、onnx vs fused 6e-5、
  conf>0.25 零类别翻转。
- **TorchScript 导出器**：`torch.onnx.export` 在 torch 2.9 起默认走 dynamo，其
  `external_data=True` 会把 ~10MB 模型也拆成 `.onnx.data`。本脚本固定 `dynamo=False`
  （权重内嵌、单文件、`dynamic_axes` 原生可用）。退路：dynamo + `external_data=False`。

## 2. onnx → TensorRT

```bash
python scripts/export.py --weights weights/yolo26n.safetensors --trt           # fp32
python scripts/export.py --weights weights/yolo26n.safetensors --trt --fp16    # + fp16
```

- `--fp16` 必须配 `--trt`：先 `model.half()` 另导 `<stem>.fp16.onnx`，再用
  **strongly-typed** 网络构建 `<stem>.fp16.engine`（fp32/fp16 产物共存）。
- TRT 11 移除了 `BuilderFlag.FP16`，也没有 precision-constraint 标志；精度完全由张量类型决定。
  把 fp32 图交给 fp16 构建会直接 ValueError。
- **引擎绑定构建机 GPU 架构 + TRT 版本**：换机器/换版本必须重建。
- TensorRT 是可选依赖，不进 `requirements.txt`；本仓库按 TRT 11.4 / onnxruntime 1.30 构建与实测。

### fp16 实测（RTX 5090，E2E 推理段）

| Model | b1 fp32 | b1 fp16 | b8 fp32 | b8 fp16 |
|---|---|---|---|---|
| YOLO26n @640 | 1.38 ms | 1.22 ms | 0.248 ms/img | **0.139 ms/img** |
| Flash-YOLO @640 | 1.32 ms | 1.23 ms | 0.213 ms/img | **0.126 ms/img** |

- b1 贴着启动开销地板；fp16 收益在 batch（b8 −41~44%）。
- 精度无可测损失；逐框对照要按 IoU 匹配，不要按分数序逐位比（近分框会换位）。

## 3. 三个运行时

| 引擎 | 设备 | 说明 |
|---|---|---|
| `pt` | GPU / CPU | 基准实现 |
| `onnx` | CPU（ORT） | 输入尺寸以图内 shape 为准；显式冲突报错 |
| `trt` | GPU | 需已构建引擎；half 输出由引擎层统一转回 fp32 |

`scripts/infer.py` 与 `scripts/eval.py` 都吃 `--engine pt|onnx|trt`，同一套前后处理。

## 4. 验收阈值

| 对拍 | 阈值 | 测试 |
|---|---|---|
| pt ↔ onnx，raw | < 5e-4 | `tests/test_export_parity.py` |
| pt ↔ onnx，E2E 检测 | < 1e-3 | 同上 |
| pt ↔ TensorRT fp32 | ≤ 2e-3 | `tests/test_trt.py` |
| torch-half ↔ TensorRT fp16（raw） | 逐位一致 | `tests/test_trt.py` |

E2E 图在 head 里对 cls 做 `.float()` 类型提升，所以半精度引擎的 E2E 输出仍是 fp32；
raw / v3 输出才是半精度（`TRTEngine._forward` 统一转回 float32）。

## 5. 发布产物约定

Release 里每份权重通常配套 9 个文件（本仓库 release 档案的口径）：

| 文件 | 口径 |
|---|---|
| `<stem>.onnx` | fp32 · batch 1（yolo26 系 = E2E 图内 top-k；v3 = 解码输出） |
| `<stem>.fp16.onnx` / `<stem>.fp16.engine` | 半精度图 + strongly-typed 构建 |
| `<stem>.engine` | fp32 · b1 |
| `<stem>.dyn.onnx` | 动态 batch 图（ORT b8 用） |
| `<stem>.b8.fp32.onnx` / `.b8.fp32.engine` | raw batch-8 图 + 引擎 |
| `<stem>.b8.fp16.onnx` / `.b8.fp16.engine` | 同上 · fp16 |

- **engine 绑定构建机**（GPU 架构 + TRT 版本）——换机需重建：
  `scripts/export.py --weights <w> --trt [--fp16]`。
- b8 的 raw 构造需要自行用 `--dynamic` 导出并构建；仓库导出路径固定 b1。
- 权重文件名与下载链接按各模型页的约定；metadata 随 safetensors header 走，见
  [../models/README.md](../models/README.md)。

## 6. 坑

- `utils/anchors.py::make_anchors` 必须 **fp32 建网格再 cast**——half 的 `torch.arange`
  会被 TorchScript ONNX 导出器拒绝。回归闸门
  `test_export_parity.py::test_make_anchors_half_dtype_is_export_safe`，不要改回去。
- batch > 1 的引擎需要 `--dynamic` 导出后自行构建；`TRTEngine.predict()` 契约是 batch 1。
- 边缘工具链优先 `--dynamic` 或固定 shape；**不要手工改 onnx 输入名**（I/O 绑定按名称匹配）。
- TensorRT 安装不要走 PyPI 同名壳包；按 Dockerfile 的 NVIDIA 索引路线，或直接用 Docker 镜像。

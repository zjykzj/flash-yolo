# 导出与部署（Export & Deployment）

> 入口：`python scripts/export.py --weights <w.safetensors> [...]`。
> 一条链：safetensors → onnx →（可选）TensorRT engine。三个运行时共用同一个 `predict()` 契约。

## 1. pt → onnx

```bash
python scripts/export.py --weights weights/yolo26n.safetensors                          # 默认 E2E 融图
python scripts/export.py --weights weights/yolo26n.safetensors --raw                    # 原始头输出
python scripts/export.py --weights weights/yolo26n.safetensors --dynamic                # 开动态 batch 维
python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny  # v3 单输出
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights` | — | `.safetensors` |
| `--out` | `runs/export/<stem>.onnx` | 输出路径 |
| `--raw` | off | 导出原始头 (B, 4+nc, N)（**yolo26 专属**；v3 会报错） |
| `--dynamic` | off | 动态 batch 轴；默认**固定 batch 1**（边缘工具链偏好静态 shape） |
| `--model` / `--scale` | 从文件名推断 | 架构/档位 |
| `--nc` | 模型 yaml 的 nc | 非 COCO 数据训练的模型改头 |
| `--imgsz` | 权重 metadata > 640 | 导出输入尺寸 |
| `--opset` | 脚本默认 | ONNX opset |

**导出图口径**

| 路径 | 输出 | 说明 |
|---|---|---|
| yolo26 默认（E2E） | `(B, 300, 6)` = [x1,y1,x2,y2,conf,cls] | 解码 + 两阶段 top-k 在**图内** |
| yolo26 `--raw` | `(B, 4+nc, N)` | 解码/后处理在外（见 `utils/postprocess.py`） |
| yolov3-tiny | `(B, NA, 5+nc)` | NA = 3·ΣH·W；obj×cls 与 NMS 在外部 |

> **TorchScript 导出器**：`torch.onnx.export` 在 torch 2.9 起默认走 dynamo 导出器，其
> `external_data=True` 默认会把 ~10MB 的模型也拆成 `<name>.onnx.data`（`.onnx` 单独不可用）。
> 本脚本固定 `dynamo=False`（权重内嵌、单文件、`dynamic_axes` 原生可用）。该路径已被标记
> deprecated——若未来 torch 移除，退路是 dynamo + `external_data=False`（同样单文件）。

## 2. onnx → TensorRT

```bash
python scripts/export.py --weights weights/yolo26n.safetensors --trt           # + fp32 引擎
python scripts/export.py --weights weights/yolo26n.safetensors --trt --fp16    # + fp16 引擎
```

- `--fp16` 必须配 `--trt`；它会**另导一份半精度 onnx**（`<stem>.fp16.onnx`，`model.half()`）再
  用 **strongly-typed 网络**构建 `<stem>.fp16.engine`——fp32/fp16 产物共存、互不覆盖。
  （TRT 11 移除了 `BuilderFlag.FP16`，也没有 precision-constraint 标志；精度完全由张量类型决定。
  把 fp32 图交给 fp16 构建会直接 ValueError，不静默产出"看着像 fp16"的 fp32 引擎。）
- **引擎绑定构建机的 GPU 架构 + TensorRT 版本**：换机器/换 TRT 版本必须重新构建
  （在目标机上跑 `--trt`）。
- TensorRT 是可选依赖、不进 `requirements.txt`；安装匹配 wheel（本仓库按 **TensorRT 11.4 /
  onnxruntime 1.30** 构建与实测），引擎代码兼容 TRT 10（`EXPLICIT_BATCH`）与 11（标志移除）。

**fp16 实测（RTX 5090，E2E 引擎，推理段，重复取 min——测量时机器上同时跑着训练）**：

| Model | b1 fp32 | b1 fp16 | b8 fp32 | b8 fp16 |
|---|---|---|---|---|
| YOLO26n @640 | 1.38 ms | 1.20 ms | 0.252 ms/img | **0.141 ms/img** |
| flash-yolo @640 | 1.33 ms | 1.20 ms | 0.213 ms/img | **0.130 ms/img** |

- **b1 贴着启动开销地板**：fp16 只省 ~10-13%，两个架构打平；b1 的可验证口径仍是 fp32 链
  （pt → onnx → engine，引擎输出与 PyTorch < 2e-3）。
- **fp16 的收益在 batch**：b8 每图延迟 −44%（YOLO26n）/ −39%（flash-yolo）。
  （b8 来自 raw 探针——`scripts/export.py` 出的是固定 batch 1，batch>1 需自行用 `--dynamic`
  导出后构建。）
- **精度无可测损失**：同权重 fp32 vs fp16 引擎，COCO val2017 前 500 图——YOLO26n
  0.4432 → 0.4434 mAP@[.5:.95]（0.5994 → 0.6001 mAP@50）；flash-yolo（筛选期权重）
  0.3218 → 0.3223（0.4553 → 0.4558）——差值都在子集噪声内。fp16 的 E2E 图在半精度下解码
  框坐标（640 输入约 0.5 px 量化），真实图片上检测数与类别完全一致（框最大差 0.84 px）。

## 3. 三个运行时

| 引擎 | 设备 | 说明 |
|---|---|---|
| `pt` | GPU / CPU | 基准实现（对齐、验收都走它） |
| `onnx` | CPU（ORT） | 输入尺寸以**图内 shape** 为准；显式 `--imgsz` 冲突会报错而非静默 |
| `trt` | GPU | 需已构建引擎；`--fp16` 引擎的 half 输出由引擎层统一转回 fp32 |

`scripts/infer.py` 与 `scripts/eval.py` 都吃 `--engine pt|onnx|trt`，同一套前后处理。

## 4. 验收阈值（改导出/引擎代码前先跑这些）

| 对拍 | 阈值 | 测试 |
|---|---|---|
| pt ↔ onnx，raw 输出 | < 5e-4 | `tests/test_export_parity.py` |
| pt ↔ onnx，E2E 检测 | < 1e-3 | 同上 |
| pt ↔ TensorRT fp32 | ≤ 2e-3 | `tests/test_trt.py` |
| torch-half ↔ TensorRT fp16（raw） | 逐位一致 | `tests/test_trt.py` |

数值细节：E2E 图在 head 里对 cls 做了 `.float()` 类型提升，所以**半精度引擎的 E2E 输出仍是
fp32**；raw / v3 输出才是半精度（`TRTEngine._forward` 统一 `.astype(np.float32)` 再返回）。
fp16 实测与 torch half 模型逐位一致、精度无可测损失（50 万像素级差异都在子集噪声内）。

## 5. 坑

- `utils/anchors.py::make_anchors` 必须 **fp32 建网格再 cast**——half 的 `torch.arange` 会被
  TorchScript ONNX 导出器拒绝（`model.half()` 的整条导出链会断）。回归闸门
  `test_export_parity.py::test_make_anchors_half_dtype_is_export_safe`，不要改回去。
- batch > 1 的引擎需要 `--dynamic` 导出后自行构建；`TRTEngine` 的 `predict()` 契约是 batch 1。
- 导出后想在边缘工具链里跑，优先 `--dynamic` 或固定 shape 都行，但**不要**手工改 onnx
  的输入名（引擎/I/O 绑定按名称匹配）。

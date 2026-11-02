# Pipeline 公共约定

四条链路：**train → eval → export → infer**。每篇只写自己的参数；本文列出所有链路共享的约定。

## 1. 命令形态

```text
python scripts/<train|eval|export|infer>.py [--weights ...] [--data ...] [选项]
```

- 所有脚本在控制台第一行打印解析后的参数预览：`scripts/xxx.py: key=value, …`。
- 有 run 目录的脚本（train / eval / infer）把日志写到 `<run_dir>/run.log`；没有 run 目录的
  脚本（`bench_io` / `convert_weights` / `download_weights`）走 `logs/` 兜底。
- `--data` 只接受描述符名或 `.yaml` 路径，不接受裸目录（见
  [../datasets/README.md](../datasets/README.md)）。

## 2. 权重与 metadata

- 权重格式：`.safetensors`（主）、`.onnx`、`.engine`（TensorRT）。
- 训练产物与转换器在 safetensors header 写可选 metadata：`arch` / `scale` / `nc` / `imgsz` /
  `names`（含锚模型另有 `anchors`）+ 出处 `version/repo/commit/describe`。
- 解析顺序：**CLI 显式 > metadata > 文件名/默认**；推不出时显式给 `--model/--scale`。
- onnx 输入尺寸以图内 shape 为准；显式 `--imgsz` 冲突会报错，不静默。

## 3. 引擎

| 引擎 | 设备 | 说明 |
|---|---|---|
| `pt` | GPU / CPU | 基准实现；对齐/验收走它 |
| `onnx` | CPU（ORT） | 输入尺寸以图内 shape 为准 |
| `trt` | GPU | 需已构建引擎；绑定构建机 GPU + TRT 版本 |

三个运行时共用同一套前后处理与 `Detections` 契约（含可选 masks）。

## 4. 阈值口径

- **eval 默认 conf = 0.001**（对齐官方评估口径）。
- **infer 默认 conf = 0.25**（出图省事）。
- 两者不要混用；推理图上的数量不能用来心算 mAP。

## 5. run 目录

- train：`runs/train/train-<name>/`（权重、results.csv、diag、samples、meta.json、run.log）。
- eval：`runs/val/valN/`（结果与 run.log）。
- infer：`runs/predict/predictN/`（可视化图/视频 + labels/）。
- 目录已存在时自动加序号；日志跟 run 目录走。

## 6. 环境与测试

- 依赖：`pip install -r requirements.txt`；开发/官方权重转换：
  `pip install -r requirements-dev.txt`。
- 测试：`pytest tests/ -q`；env-gated 闸门：
  `FLASH_YOLO_SMOKE` / `FLASH_YOLO_REPRO` / `FLASH_YOLO_DDP`。
- 容器：见 [../docker.md](../docker.md)。

## 7. 页面

- [train.md](train.md) — 从零/微调/resume、配方、DDP、复现
- [eval.md](eval.md) — pycocotools、E2E/NMS、Box+Mask
- [export.md](export.md) — ONNX / TensorRT / fp16
- [infer.md](infer.md) — 图片 / 目录 / 视频 / 流

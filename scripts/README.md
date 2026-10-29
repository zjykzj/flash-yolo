# scripts/ 脚本索引

按**使用场景**组织。本页只回答"什么场景用哪个脚本"；具体参数、产物与选项看每组末尾的
文档链接，或 `python scripts/<name>.py --help`。

## 1. 模型流水线

日常只用这四个，构成完整闭环：训练 → 评估 → 导出 → 推理。

| 脚本 | 什么场景 |
|---|---|
| `train.py` | 从零训练 / 用已有权重微调 / 多卡 DDP / 断点续训 |
| `eval.py` | 拿正式 COCO 指标（Box / Mask；E2E 为默认路径，NMS 可选）——对外数字的唯一来源 |
| `export.py` | 导出 ONNX，或顺带构建 TensorRT 引擎（fp16 / 动态 batch / raw 输出） |
| `infer.py` | 图片、目录、视频、RTSP 流、摄像头的推理与可视化 |

文档：[pipeline/train.md](../docs/pipeline/train.md) · [eval.md](../docs/pipeline/eval.md) ·
[export.md](../docs/pipeline/export.md) · [infer.md](../docs/pipeline/infer.md)

## 2. 数据集操作

场景：数据格式与本工程要求不一致，或需要一份分钟级能跑通的小数据集。

- `make_coco_subset.py` —— 从 COCO 抽子集（coco-tiny）：种子抽样、图片硬链接、生成数据集
  描述符。用于本机快速验证全流程，以及作为 YOLO 格式转换的输入。
- `convert_coco_seg.py` —— COCO 多边形标注 → YOLO-seg txt。dataflow-cv 的 `coco2yolo`
  即使有 segmentation 也优先写 bbox，多边形要由本脚本按官方 merge 口径生成。

文档：[datasets/README.md](../docs/datasets/README.md) §6 小数据集循环 ·
[datasets/coco.md](../docs/datasets/coco.md) · [datasets/yolo.md](../docs/datasets/yolo.md)

## 3. 官方权重准备

场景：需要官方权重做对照、初始化或基准复核。

- `download_weights.py` —— 按 URL 下载官方 `.pt` / darknet `.weights`。官方权重有许可边界，
  本仓库不分发，脚本只负责获取。
- `convert_weights.py` —— 官方权重 → 纯 `state_dict` safetensors。运行时不依赖
  ultralytics / pickle，metadata（arch/scale/nc/names/anchors + 出处）随权重走。

文档：[models/README.md](../docs/models/README.md) ·
[development/alignment.md](../docs/development/alignment.md)

## 4. 基准比对

场景：需要"官方权重 + 本工程预处理/后处理/指标"的**同口径数字**（对外基准表的 Official
行），或想核对实现是否对齐。

- `compare_official.py` —— dev-only 工具：读取官方 `.pt`（需要 ultralytics），让官方权重走
  本工程管线出指标；分割权重同时输出 Box / Mask 两张表。

文档：[development/alignment.md](../docs/development/alignment.md) §1

## 5. 训练配置标定

场景：换机器、换数据集、换输入尺寸——正式训练前先把配置定对。

- `bench_io.py` —— 训练 I/O 吞吐基准（loader-only 网格 → 端到端），用来定
  workers / threads / batch。**loader-only 快不代表训练快**，决策看端到端。
- `compute_anchors.py` —— 锚点先验评估与重聚类，适用于任何 yaml 带 `anchors` 段的模型；
  换输入尺寸、换数据集或接入新锚模型时用。

文档：[pipeline/train.md](../docs/pipeline/train.md) §6 先把 I/O 调对 ·
[models/yolov3-tiny.md](../docs/models/yolov3-tiny.md)（锚点判定口径）·
[faq.md](../docs/faq.md)

## 6. 架构设计

场景：设计新的轻量 backbone / neck。

- `search_arch.py` —— 按维度组合生成变体 yaml（写到 `/tmp`，不改 `config/`、不注册
  `ARCHS`），比较 GFLOPs / params / 逐层表；方案确认后再手工落到 `config/models/`。

文档：[architecture.md](../docs/architecture.md)

## 共同约定

- 控制台第一行是解析后的参数预览；参数以 `--help` 与各文档为准。
- 日志：`train` / `eval` / `infer` 跟随 run 目录（`<run>/run.log`）；其余走 `logs/` 兜底，
  个别脚本只打控制台。
- 只有 `compare_official`（以及读取官方 `.pt` 的 `convert_weights`）需要 dev 依赖
  ultralytics；其余脚本 `requirements.txt` 即可。
- `weights/`、`runs/`、`logs/` 均 gitignored；数据集路径只写 `config/datasets/local/`。

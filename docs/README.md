# Flash-YOLO 文档

根目录 [README](../README.md) 负责项目概览、安装与快速开始；本目录按 **数据 → 模型 → pipeline** 三条线组织完整手册。

## 我该读哪一篇？

| 我想… | 读这里 |
|---|---|
| 接入 COCO / YOLO 数据集 | [datasets/README.md](datasets/README.md) |
| 选择或了解某个模型 | [models/README.md](models/README.md) |
| 训练（从零 / 微调 / resume / DDP） | [pipeline/train.md](pipeline/train.md) |
| 评估（pycocotools / E2E / NMS / Box+Mask） | [pipeline/eval.md](pipeline/eval.md) |
| 导出 ONNX / TensorRT | [pipeline/export.md](pipeline/export.md) |
| 图片 / 视频 / 流推理 | [pipeline/infer.md](pipeline/infer.md) |
| 容器环境 | [docker.md](docker.md) |
| 常见问题 | [faq.md](faq.md) |
| 代码结构与扩展点 | [architecture.md](architecture.md) |
| 与官方实现的对齐 / 测试闸门 | [development/alignment.md](development/alignment.md) |

## 目录

```text
docs/
├── datasets/
│   ├── README.md      描述符 schema、--data 解析、公共约定
│   ├── coco.md        COCO json（含 RLE / crowd / 缺失图）
│   └── yolo.md        YOLO txt / YOLO-seg 多边形
├── models/
│   ├── README.md      模型矩阵、权重 metadata、公共结构约定
│   ├── yolo26.md
│   ├── yolo26-seg.md
│   ├── flash-yolo.md
│   ├── flash-yolo-seg.md
│   └── yolov3-tiny.md 原版 + SiLU 变体
├── pipeline/
│   ├── README.md      公共 CLI / run 目录 / 引擎约定
│   ├── train.md
│   ├── eval.md
│   ├── export.md
│   └── infer.md
├── development/
│   └── alignment.md   官方权重对齐、训练内部 parity、测试
├── docker.md
├── faq.md
└── architecture.md
```

## 口径约定

- **正式指标** = `scripts/eval.py` + pycocotools，COCO val2017，640 输入（YOLOv3-tiny 原生 416，也支持 640 评估）。
- **E2E** = NMS-free 默认路径（o2o 头）；**NMS** = one-to-many 头 + NMS。
- **官方 YOLO26 权重**是 Objects365 预训练 + COCO 微调；从零训练数字与官方不是 like-for-like。
- **训练 run 记录与消融细节**属于私有档案（`docs/bake/`，不进 git）；本目录只保留公开、可复现的结论与命令。

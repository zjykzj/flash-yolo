# 文档索引

本目录收录 Flash-YOLO 的使用手册、架构说明、权重档案、训练/评估记录与对齐验证。
根目录 [README](../README.md) 只保留性能概览与四个环节的快速上手；完整细节从这里进入。

## 使用手册

| 文档 | 内容 |
|---|---|
| [train.md](train.md) | 训练入口、配方、增强、优化器、断点续训与复现 |
| [val.md](val.md) | pycocotools 评估、FastMetrics、NMS 与评估口径 |
| [export.md](export.md) | ONNX / TensorRT 导出、动态 batch、fp16 |
| [infer.md](infer.md) | pt / onnx / trt 推理与 YOLO-txt 输出 |
| [dataset.md](dataset.md) | COCO / YOLO 描述符、格式转换、分割标签 |
| [docker.md](docker.md) | 容器环境与镜像构建 |
| [faq.md](faq.md) | 跨模块常见问题与坑 |

## 权重与验证

| 文档 | 内容 |
|---|---|
| [weights.md](weights.md) | 发布 checkpoint、完整延迟表、部署产物与溯源 |
| [alignment.md](alignment.md) | 与官方实现的对齐、数值验证、测试套件 |
| [architecture.md](architecture.md) | 工程结构、模块职责、扩展点 |

## 架构设计

| 文档 | 内容 |
|---|---|
| [yolo26-spec.md](yolo26-spec.md) | YOLO26 检测 / 分割结构与实现口径 |
| [flash-yolo-spec.md](flash-yolo-spec.md) | Flash-YOLO 轻量检测架构 |
| [flash-yolo-seg-spec.md](flash-yolo-seg-spec.md) | Flash-YOLO-seg 分割变体与 20/100 轮结果 |
| [yolov3-tiny-spec.md](yolov3-tiny-spec.md) | YOLOv3-tiny / SiLU 的 darknet 结构与训练口径 |

## 训练与实验记录

| 文档 | 内容 |
|---|---|
| [yolo26-runs.md](yolo26-runs.md) | YOLO26 检测训练 run 与配方归因 |
| [yolov3-tiny-runs.md](yolov3-tiny-runs.md) | YOLOv3-tiny 全部训练 run 与消融记录 |

> `bake/` 是发布准备笔记（gitignored），不属于对外文档。

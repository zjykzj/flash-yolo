# 工程结构与模块职责

> 设计原则：每个模块独立、完整、可单独阅读——不复制 ultralytics 的 monolith，运行时不依赖它。
> 核心运行时只有 **PyTorch + NumPy**；`pycocotools`（评估）、`onnx` / `onnxruntime`（导出 / 推理）、
> TensorRT（GPU 部署）都是可选依赖，按使用场景安装。

## 1. 目录结构

```text
config/     配置的单一去处：值在 yaml，机制在 py
  models/     yolo26.yaml · yolo26-seg.yaml · flash-yolo.yaml · flash-yolo-seg.yaml
              yolov3-tiny.yaml · yolov3-tiny-silu.yaml · screening/（历史筛选 yaml）
  datasets/   数据集描述符模板 + local/（gitignored 机器路径）
  recipes/    yolo26-coco-ft · yolo26-o365-pt · yolo26-seg-coco-ft · yolo26-seg-o365-pt
  train.yaml · train_config.py（TrainConfig / 配方合并） · inference.py（推理默认值）
data/        coco.py（COCO json） · yolo.py（YOLO txt / YOLO-seg） · build.py（描述符工厂）
             scan.py（共享扫描口径） · loader.py（增强入口 / collate / worker 契约）
             preprocess.py（letterbox） · augment.py（mosaic/affine/HSV/copy_paste）
             segments.py（多边形解析 / 合并 / 栅格化）
eval/        coco_evaluator.py（pycocotools wrapper，bbox | segm 任务轴）
model/       build.py（yaml 组装器 + ARCHS 注册表） · weights.py（safetensors + metadata）
             summary.py（逐层表 / FLOPs） · basic/（Conv/DWConv/SPPF/C3k2/C2PSA…）
             head/（yolo26_detect · yolo26_seg · v3_detect）
train/       assigner.py（TAL+STAL） · loss/（按家族×任务分文件） · optimizer.py（MuSGD）
             ema.py · lr.py · checkpoint.py（resume/best/last/periodic） · validator.py
             metrics.py（FastMetrics） · trainer.py（算法与训练循环）
scripts/     train · eval · export · infer · download_weights · convert_weights
             compute_anchors · make_coco_subset · convert_coco_seg · bench_io
             compare_official · search_arch
tests/       286 collected（268 passed, 18 skipped；env-gated / 可选依赖）
utils/       engines/（base · pt · onnx · trt） · postprocess.py（解码/NMS/掩码）
             anchors.py · iou.py · visualize.py · plots.py · logger.py · paths.py · progress.py
assets/      demo 图片与渲染结果
docs/        使用手册、架构 spec、训练/评估记录与对齐验证
runs/        训练 / 评估 / 导出 / 推理产物（gitignored）
```

## 2. 依赖方向

```text
scripts/  ->  train/ | eval/ | model/ | data/ | utils/
train/    ->  model/ | data/ | eval/（指标） | utils/
eval/     ->  utils/（后处理 / 引擎）
model/    ->  config/（arch 注册表 / yaml） | utils/（anchors / iou）
data/     ->  config/（描述符） | utils/（进度条）
utils/    ->  config/（推理默认值）；utils/engines/ 例外地调用 model/ 构建与加载模型
```

`config/` 原则上是最底层：只描述值与机制；`train_config.py` 为配方告警保留了一处 lazy
import（不承担 torch 导入）。`model/` 不依赖 `train/`，推理脚本与训练脚本共享同一套模型
构建 / 权重加载路径。

## 3. 模块职责

### config
- `models/*.yaml` 是纯结构数据；`model/build.py` 按 `ARCHS` 注册表组装，并按
  `depth/width/max_channels` 做通道缩放。
- `datasets/spec.py` 定义 `DatasetSpec`；`data/build.py` 是描述符到加载器的唯一工厂。
- `train_config.py` 负责“基础 yaml → 配方覆盖 → CLI”的合并；`TrainConfig` 是训练字段的单一事实源。
- `__version__` 在 `config/__init__.py`，只在发布时 bump。

### data
- COCO 与 YOLO 两种格式共用 `scan.py` 的统计口径；`build.py` 做 names 与 json categories 的硬对账。
- `loader.py` 负责增强入口、collate 与 worker 线程契约；RNG 按 `(seed, epoch)` 重播，保证
  resume 与从头训练逐位等价。
- `augment.py` 的几何 / 色彩口径逐行对齐官方；`segments.py` 处理多边形合并与栅格化。

### model
- `build.py`：yaml → 模块树；`m_.i / m_.f / m_.repeats / m_.args` 是组装元数据，**不要挂 `m_.n`**。
- `head/`：按任务拆分——检测（Detect）、分割（Segment26）、v3（V3Detect）；检测与分割共享
  E2E top-k 选择。
- `weights.py`：safetensors strict load/save；metadata 携带 arch / scale / nc / imgsz / names /
  anchors 与出处，读取顺序为 CLI > metadata > 文件名 / 默认。
- state_dict key 是属性路径，重命名模块属性会破坏官方权重零映射加载；对齐测试会拦截。

### train
- `assigner.py`：TAL + STAL，批量 `(B,N,M)` 掩码实现；分配全程 `no_grad` + `detach`。
- `loss/`：按“家族 × 任务”分文件；检测三项共用 `Σtarget_scores` 归一化，分割额外加掩码 BCE
  与 semseg 辅助项。
- `optimizer.py`：MuSGD（Muon + SGD 分组、Newton-Schulz 正交化）。
- `trainer.py`：纯算法与训练循环；架构经 `cfg.model` 分派，列名由损失项 `item_keys` 派生。
- 训练产物：`weights/best.safetensors`、`last`、`best_raw`、`epochNNN`、`results.csv`、
  `diag/train_diag.csv`、`samples/*.png`、`meta.json`。

### eval
- `coco_evaluator.py` 封装 pycocotools；`iou_type="bbox" | "segm"` 是任务轴。
- 训练中验证用 `train/metrics.py` 的 FastMetrics（快速近似，只用于趋势与 best 选取）；
  正式数字一律以 `scripts/eval.py` 为准。

### utils
- `engines/`：pt / onnx / trt 三个后端，统一 `Detections` 契约（含可选 masks）。
- `postprocess.py`：numpy 解码、类感知 NMS、`process_mask` 原型解码。
- `logger.py` / `paths.py` / `progress.py`：控制台 + 文件日志、run 目录自增、终端进度条。

### scripts
每个入口只做参数解析、组件装配与结果输出；核心逻辑留在 `train/`、`eval/`、`model/`、`data/`。
`compare_official.py` 是 dev-time 对照工具：运行该脚本时会导入 ultralytics，但不进交付链路。

## 4. 常见扩展点

- **新增架构**：`config/models/<name>.yaml` + `model/build.py::ARCHS` 注册；若是新头，在
  `model/head/` 增加实现并在 `MODULES` 登记；同时更新 weight 文件名正则与对齐测试。
- **新增任务头**：在 `model/head/<family>_<task>.py` 实现，损失放 `train/loss/`，验证契约由
  `head.postprocess_val` 提供。
- **新增数据集格式**：在 `data/build.py` 加工厂分支，返回实现 `TrainMixin` / 扫描接口的加载器；
  统计口径走 `data/scan.py`。
- **新增推理后端**：在 `utils/engines/` 继承 `BaseEngine`，实现 `_forward` 与 `summary_line`；
  后处理保持与现有 `Detections` 契约一致。

## 5. 相关文档

- [对齐与验证](alignment.md) · [权重与延迟档案](weights.md) · [评估口径](val.md)
- 使用手册：[训练](train.md) · [导出](export.md) · [推理](infer.md) · [数据集](dataset.md)
- 架构 spec：[YOLO26](yolo26-spec.md) · [Flash-YOLO](flash-yolo-spec.md) ·
  [Flash-YOLO-seg](flash-yolo-seg-spec.md) · [YOLOv3-tiny](yolov3-tiny-spec.md)

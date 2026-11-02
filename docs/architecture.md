# 工程结构与模块职责

> 设计原则：每个模块独立、完整、可单独阅读。核心运行时只有 **PyTorch + NumPy**；
> `pycocotools`（评估）、`onnx` / `onnxruntime`（导出 / 推理）、TensorRT（GPU 部署）都是可选依赖。

## 1. 目录结构

```text
config/     配置的单一去处：值在 yaml，机制在 py
  models/     yolo26.yaml · yolo26-seg.yaml · flash-yolo.yaml · flash-yolo-seg.yaml
              yolov3-tiny.yaml · yolov3-tiny-silu.yaml
  datasets/   描述符模板 + local/（gitignored 机器路径）
  recipes/    yolo26-coco-ft · yolo26-o365-pt · yolo26-seg-coco-ft · yolo26-seg-o365-pt
  train.yaml · train_config.py · inference.py
data/        coco.py · yolo.py · build.py（描述符工厂） · scan.py · loader.py
             preprocess.py（letterbox） · augment.py · segments.py
eval/        coco_evaluator.py（pycocotools wrapper，bbox | segm）
model/       build.py（yaml 组装器 + ARCHS） · weights.py（safetensors + metadata）
             summary.py · basic/（Conv/DWConv/SPPF/C3k2/C2PSA…）
             head/（yolo26_detect · yolo26_seg · v3_detect）
train/       assigner.py（TAL+STAL） · loss/（家族×任务） · optimizer.py（MuSGD）
             ema.py · lr.py · checkpoint.py · validator.py · metrics.py · trainer.py
scripts/     train · eval · export · infer · download_weights · convert_weights
             compute_anchors · make_coco_subset · convert_coco_seg · bench_io
             compare_official
tests/       核心套件 + env-gated smoke / repro / DDP
utils/       engines/（base · pt · onnx · trt） · postprocess.py · anchors.py · iou.py
             visualize.py · plots.py · logger.py · paths.py · progress.py
assets/      demo 图片、GIF 与渲染结果
docs/        使用手册（本目录）
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

`config/` 原则上是最底层：只描述值与机制；`model/` 不依赖 `train/`，推理脚本与训练脚本
共享同一套模型构建/权重加载路径。

## 3. 模块职责

### config

- `models/*.yaml` 是纯结构数据；`model/build.py` 按 `ARCHS` 注册表组装并做通道缩放。
- `datasets/spec.py` 定义 `DatasetSpec`；`data/build.py` 是描述符到加载器的唯一工厂。
- `train_config.py` 负责“基础 yaml → 配方覆盖 → CLI”的合并；`TrainConfig` 是训练字段单一事实源。
- `__version__` 在 `config/__init__.py`，只在发布时 bump。

### data

- COCO 与 YOLO 共用 `scan.py` 的统计口径；`build.py` 做 names 与 json categories 硬对账。
- `loader.py` 负责增强入口、collate 与 worker 线程契约；RNG 按 `(seed, epoch)` 重播。
- `augment.py` 的几何/色彩口径逐行对齐官方；`segments.py` 处理多边形合并与栅格化。

### model

- `build.py`：yaml → 模块树；`m_.i / m_.f / m_.repeats / m_.args` 是组装元数据，
  **不要挂 `m_.n`**（SPPF 的 `self.n` 是池化次数）。
- `head/`：按任务拆分——检测（Detect）、分割（Segment26）、v3（V3Detect）；检测与分割共享
  E2E top-k 选择。
- `weights.py`：safetensors strict load/save；metadata 携带 arch/scale/nc/imgsz/names/anchors
  与出处，读取顺序 CLI > metadata > 文件名/默认。
- state_dict key 是属性路径，重命名模块属性会破坏官方权重零映射加载。

### train

- `assigner.py`：TAL + STAL，批量 `(B,N,M)` 掩码实现；分配全程 `no_grad` + `detach`。
- `loss/`：按“家族 × 任务”分文件；检测三项共用 `Σtarget_scores` 归一化，分割额外加掩码 BCE
  与 semseg 辅助项。
- `optimizer.py`：MuSGD（Muon + SGD 分组、Newton-Schulz 正交化）。
- `trainer.py`：训练循环；架构经 `cfg.model` 分派，列名由损失项 `item_keys` 派生。
- 训练产物：`weights/best.safetensors`、`last`、`best_raw`、`epochNNN`、`results.csv`、
  `diag/train_diag.csv`、`samples/*.png`、`meta.json`。

### eval

- `coco_evaluator.py` 封装 pycocotools；`iou_type="bbox" | "segm"` 是任务轴。
- 训练中验证用 `train/metrics.py` 的 FastMetrics（快速近似，只用于趋势与 best 选取）；
  正式数字一律以 `scripts/eval.py` 为准。

### utils

- `engines/`：pt / onnx / trt 三个后端，统一 `Detections` 契约（含可选 masks）。
- `postprocess.py`：numpy 解码、类感知 NMS、`process_mask` 原型解码。
- `logger.py` / `paths.py` / `progress.py`：日志、run 目录自增、终端进度条。

### scripts

每个入口只做参数解析、组件装配与结果输出；核心逻辑留在 `train/`、`eval/`、`model/`、`data/`。
`compare_official.py` 是 dev-time 对照工具：运行时导入 ultralytics，但不进交付链路。

## 4. 常见扩展点

- **新增架构**：`config/models/<name>.yaml` + `model/build.py::ARCHS` 注册；新头在
  `model/head/` 实现并登记 `MODULES`；同步更新权重文件名正则与对齐测试。
- **新增任务头**：`model/head/<family>_<task>.py`，损失放 `train/loss/`，验证契约由
  `head.postprocess_val` 提供。
- **新增数据集格式**：在 `data/build.py` 加工厂分支，返回实现 `TrainMixin` / 扫描接口的加载器；
  统计口径走 `data/scan.py`。
- **新增推理后端**：在 `utils/engines/` 继承 `BaseEngine`，实现 `_forward` 与 `summary_line`；
  后处理保持 `Detections` 契约。

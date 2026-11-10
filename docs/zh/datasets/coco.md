# COCO 格式

描述符公共字段见 [README.md](README.md)。`--data coco` 解析到
`config/datasets/local/coco.yaml`（本机路径）或包内模板 `config/datasets/coco.yaml`。

## 1. 目录与描述符

```text
<root>/
├── images/train2017/          # 图片
├── images/val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

```yaml
format: coco
path: /path/to/coco
names: [person, bicycle, ...]   # 80 类，index = 类别 id
train: {images: images/train2017, ann: annotations/instances_train2017.json}
val:   {images: images/val2017,   ann: annotations/instances_val2017.json}
```

- `names` 与 json 的 `categories` **硬对账**：首个不一致的下标直接报错。
- 扫描统计（train/eval 都会打印）：`n_backgrounds`（剔 crowd 后仍无框的图）、`n_missing`
  （磁盘上不存在，已剔除并告警）、`n_crowd_excluded`（仅训练侧非 0）。

## 2. 检测标注

- `iscrowd=1`：**训练侧剔除、val 侧保留**。COCO 评估器按官方口径处理 crowd。
- 官方 train2017 缺 1 张图（`000000391895.jpg`）：加载器初始化时按文件存在性过滤并告警，
  属预期行为。
- **坏图惰性处理**：扫描期不逐张 decode；`load_image` 读失败时计数并告警，训练不中断
  （train 返回空白画布 + 空标签；val 返回空白图 + 保留 GT，诚实记为漏检）。
- 背景图（无任何 GT）：训练侧计入 `n_backgrounds`，不报错。

## 3. 分割标注（`--model yolo26-seg` / `flash-yolo-seg`）

每个实例额外读取 `segmentation`：

- **多边形列表**：多段按官方 `merge_multi_segment` 两轮口径合并，增强后逐点变换，
  在 letterbox 分辨率 `cv2.fillPoly`（int32 截断，与官方同口径）后降采样到
  `imgsz / mask_ratio`（默认 4，即 Proto26 的 160×160 网格）。
- **RLE**：压缩字符串或 counts 列表都解码成轮廓多边形；扫描行计入 `seg RLE N decoded`。
- **回退**：RLE 损坏 / 缺失 / 非法 segmentation 时退回 bbox 矩形，计入 `n_seg_fallback`
  并告警；**全量 fallback 直接报错**（多半是描述符或标注选错）。
- 框由多边形外接框导出（官方口径）。
- `overlap_mask=True`（默认）生成面积降序的实例索引图；labels 同步重排，**索引图 ID 与
  targets 行序必须一致**。

## 4. 关键点标注（`--model yolo26-pose`）

COCO-pose 使用 `person_keypoints_{train,val}2017.json`（不是 `instances_*.json`）：

- 每个实例带 `keypoints`（17×3 绝对像素 `x,y,v`，v=0/1/2）与 `num_keypoints`；`bbox`/`area`
  仍用于框损失与 OKS 的面积归一化。
- 描述符额外写 `kpt_shape` / `flip_idx` / `kpt_names` / `skeleton`（模板：
  `config/datasets/coco-pose.yaml`）；`kpt_shape` 必须与权重 metadata 一致。
- `require_keypoints` 按 `num_keypoints > 0` 过滤：COCO-pose train **56,599** 图、
  val **2,346** 图。**pose 的 val 不是检测的 5,000 张**，报告数字时不要混用。
- 没有 `keypoints` 字段的标注在 pose 训练/评估时直接报错（不会安静地当背景）。

```bash
python scripts/train.py --data coco-pose --model yolo26-pose \
    --weights weights/yolo26n-pose.safetensors --recipe yolo26-pose-coco --epochs 10
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose
```

## 5. 小数据集

```bash
python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \
    --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors
```

脚本按种子抽样、硬链接图片、生成 coco/yolo 描述符并拷进 `config/datasets/local/`。
之后 `--data coco-tiny` 直接可用。

## 6. 坑

- `--limit N` 按 json 顺序取前 N，与 YOLO 的文件名排序不是同一批图；跨格式比较用全量。
- COCO 与 YOLO 的 val 数字差异只来自 crowd（coco-tiny 全量 mAP50-95 0.4080 vs 0.4024）；
  训练侧两边都剔 crowd，统计逐项相等。
- 分割全量 fallback 会直接报错，不会安静地退化成框训练。

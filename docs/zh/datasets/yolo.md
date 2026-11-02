# YOLO 格式

描述符公共字段见 [README.md](README.md)。YOLO 格式用 `labels/*.txt`，没有 json 标注。

## 1. 目录与描述符

```text
<root>/
├── images/train/      # 图片（或 .txt 图片清单）
├── images/val/
├── labels/train/      # 与图片 stem 对应的 txt
└── labels/val/
```

```yaml
format: yolo
path: /path/to/mydata
names: [cat, dog]
train: {images: images/train}     # labels 缺省 = images -> labels
val:   {images: images/val}
```

- `labels` 可选：缺省时把 `images` 路径里的 `images` 段替换成 `labels`。
- `images` 也可以是 `.txt` 图片清单（每行一个路径）。
- `ann` 键**禁止**出现（写了会报错）。

## 2. 检测标签

每行 `cls xc yc w h`，坐标归一化到 `[0,1]`；空文件 = 全背景。

**标签查找两级回退**：

1. `<stem>.txt` 优先；
2. 缺失时退 `str(int(stem)).txt`——兼容 dataflow-cv `coco2yolo` 按 **COCO image id** 命名
   （`000000397133.jpg` → `397133.txt`）与 `1.txt ↔ 000001.jpg` 两种命名。

**全背景守卫**：整目录零命中直接报错——否则会安静地训练 100% 背景。

## 3. 分割标签

用于 `--model yolo26-seg` / `flash-yolo-seg`：

- **多边形行**：`cls x1 y1 x2 y2 ...`（归一化，>=3 点）；框取多边形外接框。
- **bbox 行回退**：5 列 `cls xc yc w h` 按矩形处理（混合标注允许），计入 `n_seg_fallback`
  并在扫描行显示。
- 检测解析器遇到 >5 列会报错：`polygon labels require a segmentation dataset/model`。
- 像素 GT 依赖图尺寸：`load_image(idx)` 记录 `(h, w)` 之后 `targets()` / `gt_dict()` 才可用
  （评估循环天然满足；自己写脚本先取 GT 会得到明确报错）。

## 4. 转换

```bash
# COCO -> YOLO 检测（dataflow-cv；只写 labels/ 与 classes.txt）
dataflow-cv convert coco2yolo <json> <out-yolo>

# COCO -> YOLO-seg（本仓库脚本；多边形按官方 merge 口径）
python scripts/convert_coco_seg.py --src <json> --dst <out-yolo>/labels
```

`dataflow-cv` 的 `coco2yolo` 目前优先写 bbox，不生成多边形行；分割转换用
`convert_coco_seg.py`。

## 5. 坑

- YOLO 没有 crowd 概念（转换即丢弃）；COCO 侧 val 保留 crowd，两者 val 数字差异只来自这一项。
- `--limit N` 按文件名排序取前 N，与 COCO 的 json 顺序不同；跨格式比较用全量或对齐集合。
- 整目录零命中会报错（全背景守卫），不要用空 labels 目录训练。
- 分割全量 fallback 会报错；混合标注里允许部分 bbox 行，但需要至少有可用的多边形/框。

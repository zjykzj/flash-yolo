# 数据集（Datasets）

> 数据入口只有一个：**数据集描述符 yaml**。train / eval / bench_io / infer 都通过
> `--data <name|.yaml>` 找到它；模型的类别数（nc）也由它决定（不再改模型 yaml）。

## 1. 描述符 schema

完整模板：`config/datasets/coco.yaml`。

| 键 | 必填 | 说明 |
|---|---|---|
| `format` | ✔ | `coco`（json 标注）或 `yolo`（`labels/*.txt`） |
| `path` | – | 数据集根；相对路径以描述符 yaml 所在目录为基准；空 = 只用绝对路径 |
| `names` | ✔ | 类别名列表（或 `{0: name}` 映射，等价）；**index = 类别 id = nc** |
| `train` / `val` / `test` | train、val ✔ | 角色键（`test` 接受但当前不消费） |
| ├ `images` | ✔ | coco：图片目录；yolo：图片目录或 `.txt` 图片清单 |
| ├ `ann` | coco ✔ | 标注 json 路径；**yolo 禁止**（写了会报错） |
| └ `labels` | yolo 可选 | 标签目录；缺省 = 把 `images` 路径里的 `images` 段换成 `labels` |

```yaml
# COCO（config/datasets/coco.yaml 模板）
format: coco
path: /path/to/coco            # 机器路径只写这里（local/ 描述符，gitignored）
names: [person, bicycle, ...]  # 80 类
train: {images: images/train2017, ann: annotations/instances_train2017.json}
val:   {images: images/val2017,   ann: annotations/instances_val2017.json}
```

```yaml
# YOLO txt（labels 缺省时自动从 images 推断）
format: yolo
path: /path/to/mydata
names: [cat, dog]
train: {images: images/train}
val:   {images: images/val}
```

## 2. 名字寻址与 local/ 覆盖

- `--data coco`：先查 `config/datasets/local/coco.yaml`（gitignored——**机器路径只写这里**），
  没有再落到包内模板 `config/datasets/coco.yaml`（`path` 刻意留空）。
- `--data /path/to/x.yaml`：直接指定路径。
- **裸目录会报迁移错误**（旧接口已废弃）：`--data /data/coco` ✗ → 写成描述符或复制模板。

## 3. COCO 格式

- 标注走 `ann` json；`names` 与 json 的 `categories` **硬对账**（首个不一致的下标直接报错）。
- **crowd 语义**：`iscrowd=1` 训练侧剔除、val 侧保留（评估器按官方口径处理）。
- 扫描统计（train/eval 都会打印）：`n_backgrounds`（剔 crowd 后仍无框的图）、`n_missing`
  （磁盘上不存在，已剔除并告警）、`n_crowd_excluded`（仅训练侧非 0）。
- 坏图**惰性处理**：扫描期不逐张 decode；`load_image` 读失败时计数 + 告警，训练不中断
  （train 侧返回空白画布 + 空标签，val 侧保留 GT 记漏检）。
- **实例分割训练**（`--model yolo26-seg`）额外读取 `segmentation`：多边形列表会合并多段并
  在增强后栅格化为 GT 掩码；RLE dict / 缺失 / 非法 segmentation 退回 bbox 矩形（计数并告警），
  全部退回时直接报错。框由多边形包围盒导出（官方口径）。YOLO-seg txt 同样支持（见 §4）。

## 4. YOLO txt 格式

- 标签行：`cls xc yc w h`（归一化到 [0,1]）；空文件 = 全背景。
- **标签查找两级回退**：`<stem>.txt` 优先，缺失退 `str(int(stem)).txt`——兼容
  dataflow-cv `coco2yolo` 按 **COCO image id** 命名（`000000397133.jpg` → `397133.txt`）与
  `1.txt <-> 000001.jpg` 两种命名。
- **全背景守卫**：整目录零命中直接报错——否则会安静地训练 100% 背景。
- 像素 GT 依赖图尺寸：`load_image(idx)` 记录 (h,w) 之后 `targets()` / `gt_dict()` 才可用
  （评估循环天然满足该顺序；自己写脚本时先取 GT 会得到明确报错提示）。
- 无 crowd 概念（转换即丢弃）。
- **分割标签**（`--model yolo26-seg`）：
  - 多边形行 `cls x1 y1 x2 y2 ...`（归一化，>=3 点）；框由多边形外接框导出；
  - 5 列 bbox 行按矩形回退（混合标注允许，计入 `n_seg_fallback` 并在扫描行显示）；
  - 检测解析器遇到 >5 列会报错 `polygon labels require a segmentation dataset/model`；
  - COCO json 转 YOLO-seg：`python scripts/convert_coco_seg.py --src <json> --dst <labels-dir>`
    （按官方 `merge_multi_segment` 口径合并多段；dataflow-cv 的 `coco2yolo` CLI 目前优先写
    bbox，不生成多边形行）。

**两种格式的 val 口径差只有 crowd 一项**，实测（coco-tiny 全量）：mAP@[.5:.95] 0.4080（coco）
vs 0.4024（yolo）；训练侧统计逐项相等（crowd 训练侧双方都剔）。另注意 `--limit N` 的取样集
两格式不同（COCO 走 json 顺序、YOLO 走文件名排序）——跨格式对比要用全量或对齐集合。

## 5. 自定义数据集检查清单

1. 目录结构：`images/` 与 `labels/`（YOLO）或 `images/` + `annotations/*.json`（COCO）；
2. 描述符：`names` 顺序 = 训练/推理时的类别 id（**改顺序 = 改语义，老权重会错位**）；
3. 自检：`python scripts/eval.py --weights <任一同 nc 权重> --data <desc> --limit 20` 走通扫描与
   加载链路（或 `python scripts/train.py --data <desc> --epochs 1 --val-epochs 0` 冒烟）；
4. 锚点（仅含 `anchors` 段的模型）：先跑 `python scripts/compute_anchors.py --data <desc>`
   看 verdict（见 `yolov3-tiny-spec.md`）——COCO 形状以外的新数据分布才需要重聚类。
5. 分割：COCO json 的每个实例需要 `segmentation` 多边形（RLE 会退回 bbox 矩形并在扫描行
   计数）；YOLO 侧用多边形行或 bbox 回退行；`mask_ratio` 必须整除 `imgsz`
   （默认 4；640 -> 160 与 Proto26 同网格）。

## 6. 小数据集循环（分钟级迭代）

```bash
# ① 从现有 COCO 抽子集（种子抽样 + 硬链接；同时写 coco/yolo 描述符并拷进 local/）
python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \
    --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors
# ② YOLO 检测格式转换（dataflow-cv；只写 labels/ 与 classes.txt，images/ 用符号链接接上）
dataflow-cv convert coco2yolo <json> <out-yolo>
# ②b YOLO 分割格式转换（本仓库脚本；多边形按官方 merge 口径，bbox-only 标注写 5 列回退）
python scripts/convert_coco_seg.py --src <json> --dst <out-yolo>/labels
# ③ 之后 --data coco-tiny / coco-tiny-yolo 直接可用（名字寻址到 local/）
python scripts/train.py --data coco-tiny --epochs 5 --batch 16 --workers 8
```

## 7. 各命令里 `--data` 的用途

| 命令 | 用途 |
|---|---|
| `train.py` | 训练/验证数据、nc、类别名（权重 metadata 会带上 names/nc） |
| `eval.py` | 评估数据（`--split` 选角色，默认 val）；nc 用于建头 |
| `bench_io.py` | I/O 基准的取样数据（只用 train 角色） |
| `infer.py` | 只用 `names` + `nc`（给非 COCO 数据训练的模型配标签名）；不读图片 |
| `export.py` | 不用 `--data`；非 COCO 模型用 `--nc N` 指定 ONNX 头的类别数 |

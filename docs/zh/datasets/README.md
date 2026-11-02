# 数据集（Datasets）

> 数据入口只有一个：**数据集描述符 yaml**。train / eval / bench_io / infer 都通过
> `--data <name|.yaml>` 找到它；模型的类别数（nc）也由它决定，不再改模型 yaml。

格式差异见 [coco.md](coco.md) 与 [yolo.md](yolo.md)；训练侧用法见
[../pipeline/train.md](../pipeline/train.md)。

## 1. 描述符 schema

模板：`config/datasets/coco.yaml`。

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
# COCO
format: coco
path: /path/to/coco
names: [person, bicycle, ...]
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

- `--data coco`：先查 `config/datasets/local/coco.yaml`（gitignored，**机器路径只写这里**），
  没有再落到包内模板 `config/datasets/coco.yaml`（`path` 刻意留空）。
- `--data /path/to/x.yaml`：直接指定路径。
- **裸目录会报迁移错误**（旧接口已废弃）：`--data /data/coco` ✗ → 写成描述符或复制模板。

## 3. 公共约定

- `names` 的顺序就是训练/推理的类别 id；**改顺序 = 改语义**，老权重会错位。
- 描述符里的 `names` 与 COCO json 的 `categories` 会硬对账（首个不一致的下标直接报错）。
- 训练侧剔除 `iscrowd=1` 的 GT；评估侧保留（官方口径），详见 [coco.md](coco.md)。
- 坏图是惰性处理：扫描期不逐张 decode；`load_image` 失败时计数并告警，训练不中断
  （train 返回空白画布 + 空标签，val 保留 GT 记漏检）。
- 两种格式的 `--limit N` 取样集不同：COCO 走 json 顺序、YOLO 走文件名排序；跨格式比较
  要用全量或对齐集合。

## 4. 各命令里 `--data` 的用途

| 命令 | 用途 |
|---|---|
| `train.py` | 训练/验证数据、nc、类别名（权重 metadata 会带上 names/nc） |
| `eval.py` | 评估数据（`--split` 选角色，默认 val）；nc 用于建头 |
| `bench_io.py` | I/O 基准的取样数据（只用 train 角色） |
| `infer.py` | 只用 `names` + `nc`（给非 COCO 数据训练的模型配标签名）；不读图片 |
| `export.py` | 不用 `--data`；非 COCO 模型用 `--nc N` 指定 ONNX 头的类别数 |

## 5. 自定义数据集检查清单

1. 目录结构：`images/` + `labels/`（YOLO）或 `images/` + `annotations/*.json`（COCO）；
2. 描述符：`names` 顺序 = 训练/推理时的类别 id；
3. 自检：`python scripts/eval.py --weights <同 nc 权重> --data <desc> --limit 20`，或
   `python scripts/train.py --data <desc> --epochs 1 --val-epochs 0`；
4. 含 `anchors` 的模型（YOLOv3-tiny）：先跑 `python scripts/compute_anchors.py --data <desc>`
   看 verdict，见 [../models/yolov3-tiny.md](../models/yolov3-tiny.md)；
5. 分割：COCO 多边形 / RLE 或 YOLO-seg 多边形行；`mask_ratio` 必须整除 `imgsz`
   （默认 4；640 -> 160，与 Proto26 同网格）。

## 6. 小数据集循环

```bash
# ① 从 COCO 抽子集（种子抽样 + 硬链接；同时写 coco/yolo 描述符并拷进 local/）
python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \
    --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors

# ② YOLO 检测格式转换（dataflow-cv；只写 labels/ 与 classes.txt，images/ 用符号链接接上）
dataflow-cv convert coco2yolo <json> <out-yolo>

# ②b YOLO 分割格式转换（本仓库脚本；多边形按官方 merge 口径）
python scripts/convert_coco_seg.py --src <json> --dst <out-yolo>/labels

# ③ 之后 --data coco-tiny / coco-tiny-yolo 直接可用
python scripts/train.py --data coco-tiny --epochs 5 --batch 16 --workers 8
```

两种格式的 val 口径差只有 crowd 一项：coco-tiny 全量实测 mAP50-95 0.4080（COCO）vs
0.4024（YOLO）；训练侧统计逐项相等。

# LVIS 数据集（v1）

> LVIS 是 COCO 格式上的 **federated 标注**协议：每张图只标注部分类别，评估走
> `faster_coco_eval` 的 `lvis_style`。本工程用它做 YOLOE 开放词汇的评估与微调；
> 描述符模板：`config/datasets/lvis.yaml`。

## 1. 与 COCO 的差异

| 项 | COCO | LVIS |
|---|---|---|
| 类别数 | 80 | 1,203（r 337 / c 461 / f 405）|
| 每图标注 | 类别穷举 | 只标注部分类别 |
| 图级字段 | — | `neg_category_ids` / `not_exhaustive_category_ids` |
| 评估 | pycocotools | faster_coco_eval `lvis_style` |
| 指标 | AP / AR | AP / APr / APc / APf |
| maxDets | 100 | 300（官方）|
| 原始 json | 有 `file_name` / `iscrowd` | **没有**，需 compat 转换 |

- `neg_category_ids`：该图**确认不存在**的类别；出现预测记 FP。
- `not_exhaustive_category_ids`：该图该类标注**不完整**；该类预测忽略。
- 既不在 GT、也不在 neg / not_exhaustive 列表的 `(图, 类)` 预测直接丢弃。
- **不要用 pycocotools 评 LVIS**：它不认识这两个图级字段，同一份预测可差数倍。

## 2. 目录与标注

```text
<root>/
├── images/train2017/          # LVIS train 的图（来自 COCO train2017）
├── images/val2017/            # LVIS val / minival 的图（见下）
└── annotations/
    ├── lvis_v1_train.json     # 100,170 图 / 1,270,141 标注
    ├── lvis_v1_val.json       # 19,809 图 / 244,707 标注
    ├── lvis_v1_minival.json   # 可选：COCO val2017 的 5,000 图 / 50,672 标注
    └── lvis_v1_*_compat.json  # 转换产物（本工程读这个）
```

- LVIS train 的 100,170 张图都在 COCO train2017 中，可硬链接，不重复占盘。
- LVIS val 的 19,809 张图来自 15,000 张 COCO train2017 + 4,809 张 COCO val2017；
  本仓库示例把它们统一放在 `images/val2017`，共 20,000 张（含 minival 的 191 张背景图）。
- minival 是 COCO val2017 的 5,000 张图，用于和官方 YOLOE 数字对齐；不是另一套标注格式。
- minival 里有 191 张**背景图**（无标注，`neg_category_ids` 也为空）；过滤时不能把它们
  丢掉，否则评估集从 5,000 变成 4,809。

## 3. compat 转换

原始 LVIS 缺 `file_name`（只有 `coco_url`）与 `iscrowd`，类别名是下划线形式；本工程读
compat JSON：

```bash
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_train.json \
    --dst annotations/lvis_v1_train_compat.json
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_val.json \
    --dst annotations/lvis_v1_val_compat.json
# minival：官方 minival 注解直接转
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_minival.json \
    --dst annotations/lvis_v1_minival_compat.json
# 可选：从 val 过滤出 minival（ref 可以是 LVIS/COCO json，或图片 id / 文件名列表）
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_val.json \
    --dst annotations/lvis_v1_minival_compat.json \
    --minival-ref annotations/lvis_v1_minival.json
```

转换做四件事：

1. 图片补 `file_name`：优先 `coco_url` 的 basename，否则 `%012d.jpg`。
2. 标注补 `iscrowd: 0`（LVIS 无 crowd）。
3. 类别按原始 id 排序重编号为 `0..1202`，名字对齐 `--names-from`（默认
   `config/datasets/lvis.yaml`）。
4. `neg_category_ids` / `not_exhaustive_category_ids` 用同一映射重编号。

- `--names-from` 换成自定义描述符时，类别数必须与 LVIS 的 1,203 一致。
- `--minival-ref` 指向 JSON 时，源里缺失的图片会从 ref 补齐（含背景图）；指向文本列表时
  只过滤源中已有的图，不能补齐缺图。
- compat 是派生文件；不要手改，重跑脚本即可。

## 4. 描述符

```bash
cp config/datasets/lvis.yaml config/datasets/local/lvis.yaml   # 然后编辑 `path:`
```

- `format: coco`、`eval_style: lvis`、`names` 1,203 类。
- `train` / `val` 分别指向 `lvis_v1_{train,val}_compat.json`。
- minival：复制一份到 `config/datasets/local/lvis-minival.yaml`，`val.ann` 改
  `lvis_v1_minival_compat.json`，图片仍是 `images/val2017`。
- 之后：`--data lvis` / `--data lvis-minival`。

## 5. 训练（YOLOE）

LVIS 训练只用于 YOLOE 文本模式的 PE / 全量微调；没有从头训练。

```bash
# PE 线性探测（10 轮，lr0 0.01）
python scripts/train.py --data lvis --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 16 --lr0 0.01 \
    --val-limit 5000 --name yoloe-pe-lvis

# 全量微调（10 轮，lr0 0.001）
python scripts/train.py --data lvis --model yoloe-26 --scale n --full-finetune \
    --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 16 --lr0 0.001 \
    --warmup-epochs 3 --val-limit 5000 --name yoloe-ft-lvis
```

- `--val-limit 5000` 只是把训练期验证限制到 val 前 5,000 张（控制耗时），不是 minival；
  minival 只用于最终评估。
- federated 标注自动生成每图分类有效掩码：正类（含 crowd）∪ `neg_category_ids` 参与
  BCE，`not_exhaustive_category_ids` 与未标注类忽略。
- federated 数据自动关闭 mosaic / mixup / cutmix / copy_paste（掩码按中心图构造）。
- `--no-federated-mask` 只用于 A/B 对照或复现旧行为，不要用于正式训练。
- 训练预算是 10 轮；验证曲线未平台再续。

## 6. 评估

```bash
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --eval-max-dets 100      # 对齐 ultralytics 当前 val 口径
```

- 自动走 `faster_coco_eval` 的 `lvis_style`；分割权重同时出 Box / Mask 双表。
- 输出多一行 `LVIS freq APr ... · APc ... · APf ...`。
- LVIS 官方 maxDets=300；`--eval-max-dets` 只对 LVIS 生效。
- 评估前先跑通 `--limit 20`，确认类别名与描述符一致。

## 7. 坑

- **`frequency` 必填**：`lvis_style=True` 要求每个 category 带 `r/c/f`；缺失会在构造
  评估器时报错。
- **`eval_style: lvis` 只接受 `format: coco`**；YOLO txt 不支持 LVIS 协议。
- **不要用 pycocotools 评 LVIS**：它不认识 neg / not_exhaustive，结果不是 LVIS 指标。
- **零检出图要计入**：评估器按本次遍历的全部 image_id 设置 `params.imgIds`；`--limit`
  子集要保留同一批 GT。
- **类别名取第一个同义词**：`config/datasets/lvis.yaml` 使用官方首个同义词（空格形式）；
  自定义描述符必须与 `--names-from` 一致。
- **minival 不是另一套标注**：它只是 val2017 的 5,000 图子集；图仍是 `images/val2017`。

相关页面：[models/yoloe.md](../models/yoloe.md) ·
[pipeline/eval.md](../pipeline/eval.md) · [pipeline/train.md](../pipeline/train.md)。

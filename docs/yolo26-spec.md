# YOLO26 复现规格（结构事实与实现口径）

> 复现对象：YOLO26n（官方权重经本仓库管线：**40.27** mAP@[.5:.95] E2E / **40.89** NMS）。
> 本文记录结构事实与本仓库的实现口径；训练与评估记录、经验教训见 `yolo26-runs.md`。

## 1. 口径声明（读之前先看）

- **官方 40.1（E2E）/ 40.9（NMS）不是从零训练的结果**。官方《YOLO26 Training Recipe》：
  所有 checkpoint 都是 **Objects365v1 预训练 150 轮 + COCO 微调**（n 档 245 轮、batch 128），
  原文明确 "No YOLO26 checkpoint was trained on COCO from random weights … a from-scratch
  comparison against those numbers is not like for like"。
- 本仓库的复现口径 = **同权重、同前后处理、同 pycocotools 指标、同硬件**重跑官方权重
  （逐位一致的前提下的必然结果），外加自己的从零训练基线（见 runs 文档）。
- 代码为独立重写（Apache-2.0）：不复制 ultralytics 任何代码；ultralytics 仅作 dev-time 数值对照。

## 2. 缩放规则（n/s/m/l/x）

| scale | depth | width | max_channels |
|---|---|---|---|
| n | 0.50 | 0.25 | 1024 |
| s | 0.50 | 0.50 | 1024 |
| m | 0.50 | 1.00 | 512 |
| l | 1.00 | 1.00 | 512 |
| x | 1.00 | 1.50 | 512 |

- 重复次数：`n = max(round(repeats * depth), 1)`（repeats > 1 时）
- 通道：`c2 = make_divisible(min(c2, max_ch) * width, 8)`，`make_divisible(x) = ceil(x/8)*8`
- **m/l/x 档所有 C3k2 强制 `c3k=True`**（结构定义里的 False 只对 n/s 生效）
- n 档通道序列：3 → 16 → 32 → 64 → 128 → 256（width 0.25，未被 max_ch 截断）

## 3. 逐层规格（yolo26n，24 行）

| idx | 模块 | 参数 | 通道 (n 档) |
|---|---|---|---|
| 0 | Conv | k3 s2 | 3 → 16 |
| 1 | Conv | k3 s2 | 16 → 32 |
| 2 | C3k2 (n=1) | c3k=False, e=0.25 | 32 → 64 |
| 3 | Conv | k3 s2 | 64 → 64 |
| 4 | C3k2 (n=1) | c3k=False, e=0.25 | 64 → 128 |
| 5 | Conv | k3 s2 | 128 → 128 |
| 6 | C3k2 (n=1) | c3k=True | 128 → 128 |
| 7 | Conv | k3 s2 | 128 → 256 |
| 8 | C3k2 (n=1) | c3k=True | 256 → 256 |
| 9 | SPPF | k=5, n=3, shortcut=True | 256 → 256 |
| 10 | C2PSA (n=1) | e=0.5 | 256 → 256 |
| 11 | Upsample | ×2 nearest | 256 → 256 |
| 12 | Concat | [11, 6] | 384 |
| 13 | C3k2 (n=1) | c3k=True | 384 → 128 |
| 14 | Upsample | ×2 nearest | 128 → 128 |
| 15 | Concat | [14, 4] | 256 |
| 16 | C3k2 (n=1) | c3k=True | 256 → 64 **(P3/8)** |
| 17 | Conv | k3 s2 | 64 → 64 |
| 18 | Concat | [17, 13] | 192 |
| 19 | C3k2 (n=1) | c3k=True | 192 → 128 **(P4/16)** |
| 20 | Conv | k3 s2 | 128 → 128 |
| 21 | Concat | [20, 10] | 384 |
| 22 | C3k2 (n=1) | c3k=True, e=0.5, **attn=True** | 384 → 256 **(P5/32)** |
| 23 | Detect | nc=80, reg_max=1, end2end=True, ch=[64,128,256] | |

- **硬指标：260 层（模块树）/ 2,572,280 参数**（官方 summary；本仓库 `model/summary.py` 逐项一致）。
- 检测层 P3/P4/P5（stride 8/16/32），输入 640×640，输出最多 300 检测/图。

## 4. 基础模块语义

分三类：**卷积类**（Conv / DWConv）、**池化类**（SPPF）、**组合块**（Bottleneck / C3k / C3k2 /
C2PSA / PSABlock / Attention）。

- **Conv**：`Conv2d(k, s, pad=k//2, bias=False)` → BN(**eps=0.001**) → SiLU；属性 `conv/bn/act`
  - 官方 BN 统一 **eps=0.001**（非 PyTorch 默认 1e-5）——数值对齐必须一致，否则从第 0 层起
    activation 就分叉；对齐测试能抓到。
  - BN momentum = 0.03（官方口径，只影响 running stats）。
- **DWConv**：Conv with `groups=gcd(c1,c2)`。
- **Bottleneck**：cv1/cv2 = Conv 3×3；`c_ = int(c2·e)`；c1==c2 时加残差。
- **C3k**：cv1/cv2 = Conv 1×1 (c1→c_)，cv3 = Conv 1×1 (2c_→c2)，m = 2×Bottleneck(3×3)。
- **C3k2**：cv1 = Conv 1×1 (c1→2c)，cv2 = Conv 1×1 ((2+n)·c→c2)，`c = int(c2·e)`；m 按模式：
  `attn=True` → Sequential(Bottleneck, PSABlock) · `c3k=True` → C3k · 否则 Bottleneck；
  forward：cv1 输出 chunk 两半，一半过 m 链，cat 后过 cv2。
- **SPPF**：cv1 = Conv 1×1 (c1→c1//2, **act=False**)，m = MaxPool2d(5,1,2) 串行 n=3 次，
  cv2 = Conv 1×1 ((n+1)·c_→c2)；c1==c2 时输出加恒等残差。**注意 `self.n` 是池化次数**
  （组装器元数据用 `m_.repeats`，绝不占用 `m_.n`）。
- **C2PSA**：cv1 = Conv 1×1 (c1→2c)，`c = int(c1·e)`；m = n×PSABlock；split(a,b)，b=m(b)，
  cv2 = Conv 1×1 (2c→c1)。
- **PSABlock**：attn = Attention(c, attn_ratio=0.5, num_heads=max(c//64,1))；
  ffn = Conv(c→2c,1) → Conv(2c→c,1,act=False)；`x = x + attn(x)`；`x = x + ffn(x)`。
- **Attention**：head_dim = c//num_heads，key_dim = int(head_dim·0.5)，scale = key_dim^-0.5；
  qkv = Conv(c→c+2·nh·kd, 1, act=False)；proj = Conv(c→c, 1, act=False)；
  pe = Conv(c→c, 3, 1, g=c, act=False)（3×3 depthwise 位置编码）；
  `attn = softmax((q·scale)ᵀ @ k)`；`out = v @ attnᵀ + pe(v)`；再 proj。
  - 属性树（state_dict 对齐）：`attn.qkv / attn.proj / attn.pe / ffn.0 / ffn.1`

## 5. Detect 双头（o2m + o2o）

- **Box 分支** cv2[i]：Conv(x→c2,3) → Conv(c2→c2,3) → Conv2d(c2→4,1,bias=True)；
  `c2 = max(16, ch[0]//4, 4·reg_max)`（n 档 c2 = 16）；reg_max=1 → 4 通道 ltrb 距离（无界）。
- **Cls 分支** cv3[i]：(DWConv(x→x,3) → Conv(x→c3,1)) → (DWConv(c3→c3,3) → Conv(c3→c3,1))
  → Conv2d(c3→nc,1)；`c3 = max(ch[0], min(nc,100))`（n 档 c3 = 80）。
- **one2one_cv2 / one2one_cv3 = deepcopy(cv2 / cv3)**。
- **bias_init**（构造时执行一次）：box 末层 bias = 2.0；cls 末层 bias = log(5/nc/(S/stride)²)，
  stride=[8,16,32]，S = 训练输入边长（先验：每图 5 个目标摊到 nc 类与 (S/stride)² 个格子）。
  官方把 S 写死 640——非 640 从零训练时先验偏 (640/S)² 倍（只影响起点，加载权重后全被覆盖）。
- dfl = Identity（reg_max=1）。
- 属性树（state_dict 对齐）：`cv2.{i}.{0,1,2}`、`cv3.{i}.{0.0,0.1,1.0,1.1,2}`、
  `one2one_cv2.*`、`one2one_cv3.*`；head 在 index 23。

## 6. 解码与后处理（两条路径）

- anchors = 每层网格 `(arange(w)+0.5, arange(h)+0.5)` 拼接 → **(2, N) 转置形式**（grid 单位），
  便于与 (B, 4, N) 预测按最后一维广播；乘 stride 得像素坐标。
- **E2E 路径**（o2o，默认）：ltrb → `xyxy = anchor ± dist × stride`；scores = sigmoid；
  **两阶段 top-k(300)**：① 每 anchor 取 max 类分 → top-300 anchor ② 在这 300×80 候选上再取
  top-300（同一 anchor 可重复出现在不同类别）→ 输出 **(B, 300, 6) = [x1,y1,x2,y2,conf,cls]**。
  顺序必须逐位一致，否则对拍过不去。
- **NMS 路径**（o2m）：解码 (B, 4+nc, 8400)（**xywh**——官方 `xywh=not end2end`，解成 xyxy 再
  比对会差 ~632）→ 类感知贪心 NMS（conf 0.001 / iou 0.7 / max_det 300），候选全局按分数降序、
  跨类用坐标偏移隔离、凑满 max_det 即早停（"保留序即分数序"，与逐类全量等价）。

## 7. 前处理（与官方 val 口径一致）

- letterbox：`r = min(640/h, 640/w)` → 缩放尺寸 `round(dims·r)` → pad 至 640×640（灰度 114），
  返回 ratio/pad 用于坐标还原；BGR→RGB；/255（无 mean/std）。

## 8. 权重与 state_dict

- key = **属性路径**（`model.23.one2one_cv2.0.0.conv.weight`）——属性树必须与官方布局逐位对齐
  （重命名模块属性 = 破坏零映射 strict load，对齐测试会抓到）。
- 官方 ckpt：`ckpt["model"]` 是完整模块对象（`ema` 为 None）→ `scripts/convert_weights.py`
  解包一次落成纯 safetensors（运行时零 ultralytics 依赖）。
- **加载成功即拓扑一致**：`strict=True` 全 key 匹配是结构对齐的第一道闸门。

## 9. 工程接入

- 选择方式：`--model yolo26 --scale n|s|m|l|x`（默认 n）；`config/models/yolo26.yaml` 是唯一结构定义。
- 导出：`scripts/export.py`（默认 E2E 融图单输出 (B,300,6)；`--raw` 导出原始头 (B,4+nc,N)）。
- 引擎：pt / onnx / trt 三运行时共用 `predict()` 契约；E2E 图在 head 里对 cls 做 `.float()`
  类型提升，所以半精度引擎的 E2E 输出仍是 fp32（raw 输出才是 fp16）。

## 10. 验收

| 项 | 标准 | 实测 |
|---|---|---|
| 参数数 | 2,572,280 | 2,572,280 |
| strict load | 全 key 匹配 | 通过 |
| 与官方推理输出 | 逐位一致 | 一致（两条路径） |
| COCO val2017 @640 | E2E ≈ 40.1 / NMS ≈ 40.9（官方指标实现） | **40.27 / 40.89** |
| 同权重 ultralytics 实现（本管线） | 与本实现逐项一致 | 40.27 / 55.80 · 40.89 / 56.87，P/R 一致（`compare_official.py`，见 runs 文档） |
| onnx 与 pt | < 5e-4（raw）/ < 1e-3（E2E 检测） | 通过 |
| TensorRT fp32 | ≤ 2e-3 | 通过 |

（发布值与本表 ~0.1 的差是**指标实现噪声**，不是模型差异。）

## 11. 分割版（yolo26-seg）

**结构**：backbone/neck 与检测版**逐行相同**（官方 yaml diff 的唯一一行是最后一行），
头换成 `Segment26` = `Detect` + 两个分割专属件：

| 件 | 内容 |
|---|---|
| `cv4` / `one2one_cv4` | 每级 3×3 conv stack -> `nm=32` mask 系数（`c4 = max(ch[0]//4, 32)`） |
| `proto`（Proto26） | P4/P5 -> 1×1 投到 P3 通道 -> nearest 上采样 ×2/×4 相加 -> 3×3 -> ConvTranspose2d ×2 -> 3×3 -> 1×1，输出 (B, 32, H/4, W/4) |
| `proto.semseg` | 训练期辅助语义头，**官方 ckpt 里存在但从未训练**（`train_metrics["val/sem_loss"] == 0.0`）：保留子模块只为 strict load 零映射，前向不计算（推理零成本） |

- `npr`（yaml 256）随 width 缩放：n 档 64（官方 parse_model 同款 `make_divisible(min(npr, max_ch) * width)`）；
  `nm` 是任务常数不缩放。n 档合计 **3,126,280 参数 / 309 层**（与本工程 summary 逐位一致）。
- o2o 分支的 proto 走 `detach()`（与特征 detach 同口径：梯度只经 o2m 回流）。

**推理契约**：`(检测输出, 原型)` 二元组——E2E 下检测输出 `(B, 300, 6+nm)`，raw 下
`(B, 4+nc+nm, N)`；E2E 的两阶段 top-k 与检测**共用同一实现**（系数按同一组 anchor 索引取回）。
掩码解码 `utils/postprocess.process_mask`：系数 @ 原型 -> bilinear 上采样到网络输入尺寸 ->
`>0` 二值化 -> 按框裁剪 -> 反 letterbox 重采样到原图 -> `>0.5` 再二值化（与官方
`process_mask(upsample=True)` + `scale_masks` 两段口径一致）。

**训练**：`ComputeLossSeg` 复用检测双分支损失（box/cls/l1 + ProgLoss），每个分支额外计算掩码项：
系数 @ 原型 -> BCE（按责任 GT 框裁剪、按归一化框面积缩放，全 batch 求和 / 正样本数），
掩码增益 = `box_gain`（官方 `loss[1] *= hyp.box`）；o2m/o2o 的 proto 分别取自身输出（o2o 已
detach），alpha 权重作用于整条分支。数据侧 COCO 多边形随几何增强变换，最终栅格化到
`imgsz/mask_ratio`（默认 4，与 Proto26 同网格；`overlap_mask=True` 为面积降序实例索引图）。
训练中验证同时算 Box + Mask（低分辨率快速近似），best fitness = box + mask mAP@[.5:.95]。
官方 `semseg` 辅助语义项未实现（结构与现有 head 契约一致，见上表）。

**验收**：

| 项 | 标准 | 实测 |
|---|---|---|
| 键面（vs 官方 ckpt） | 844 张量逐键一致 | 0 missing / 0 extra / 0 形状不符 |
| strict load | 转换产物全 key 匹配 | 通过 |
| 与官方推理输出 | 逐位一致 | (1,300,38) 与 (1,32,160,160) 均 `torch.equal` |
| 掩码解码 vs 官方管线 | 逐像素一致 | 29 掩码 7,925,120 像素 0 差异 |
| COCO val2017 @640（本管线） | 接近官方 | Box **39.67 / 55.48** · Mask **33.67 / 52.76**（mAP50-95 / mAP50） |
| 官方训练期记录 | 参考 | box 39.75 · mask 33.86（差 ~0.002，指标实现噪声） |

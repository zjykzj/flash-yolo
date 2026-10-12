# Docker 开发环境

> 用途：一个开箱即用的容器，复刻本仓库已验证的运行环境（含 TensorRT），让 train / eval /
> export / infer 四条链路在干净机器上直接可跑。
> 相关文件：根目录 `Dockerfile` + `.dockerignore`，`docker/run.sh`（唯一运行入口），
> `docker/constraints.txt`（版本快照）；**用法速查（构建 → 运行 → 验收 → 常见坑）见
> `docker/README.md`**，本文只讲"是什么 / 为什么"。

## 1. 镜像内容与宿主要求

| 项 | 值 |
|---|---|
| 基础镜像 | `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`（含 Python 3.12 + torch/torchvision） |
| Python / torch / CUDA | 3.12 · torch 2.13.0+cu130 · CUDA 13.0 |
| TensorRT | 11.4.0.106（来自 NVIDIA 索引的真包，**默认含在镜像内**） |
| 其他依赖 | `requirements.txt` + `requirements-dev.txt`（ultralytics/pytest），版本由 `docker/constraints.txt` 钉死 = py312 快照 |
| apt | `libgl1` / `libglib2.0-0`（opencv-python 导入依赖）+ git/curl |
| 宿主 | Docker + **nvidia-container-toolkit** + NVIDIA 驱动 **>= 580**（CUDA 13），运行时 `--gpus all` |

镜像内自带一份源码快照（独立可用：推理 / demo / CI）；开发时 bind mount 覆盖它。

## 2. 构建

```bash
docker build -t flash-yolo:dev .

# 非阿里云网络：覆盖 pip 索引
docker build -t flash-yolo:dev . \
    --build-arg PIP_INDEX_URL=https://pypi.org/simple --build-arg PIP_TRUSTED_HOST=
```

## 3. 运行（`docker/run.sh`）

```bash
docker/run.sh                              # 交互式 shell（仓库挂载在 /workspace）
docker/run.sh pytest tests/                # 在容器内执行任意命令
DATA_DIR=/path/to/datasets docker/run.sh   # 数据集挂到 /data
IMAGE=flash-yolo:dev docker/run.sh         # 指定镜像名
```

`run.sh` 固定带上的东西：`--gpus all`（无 GPU 机器删这一行）、`--shm-size=8g`（DataLoader 多
worker 需要，Docker 默认 64M 不够）、`--user $(id -u):$(id -g)`（bind mount 下产物属主与宿主一致）、
仓库 bind mount 到 `/workspace`、`DATA_DIR` 挂到 `/data`（相对路径会先归一化为绝对路径）。

## 4. 约束与注意

- **TensorRT 引擎绑定构建机的 GPU 架构 + TRT 版本**：容器里用 `--engine trt` 需要重建引擎
  （`scripts/export.py --trt [--fp16]`），宿主机上 build 的 `.engine` 不一定能直接喂给容器。
- 需要临时 apt/pip 安装时，删掉 `run.sh` 里的 `--user` 一行（以 root 运行）。
- 没有用 docker compose（服务数=1、且是交互式使用）。
- **本机（AutoDL 容器实例）无法构建/运行 Docker**（无 docker CLI、系统盘空间不足）——
  以下"待验证清单"要在别的机器上完成。

## 5. 首次在其他机器构建的检查点

1. base tag 是否存在：`pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`（Docker Hub 以实际发布
   为准；缺则退 `-devel` 变体——工具链等价、体积更大）；
2. base 的 Python 是否确为 3.12（`docker run --rm flash-yolo:dev python -V`）；
3. TensorRT 直连索引安装是否成功（失败则按 Dockerfile 注释里的 curl 备选路线，手工装
   `tensorrt_cu13_libs` / `_bindings` / `tensorrt_cu13` 三个产物）；
4. apt 包名随 base 发行版（`libgl1`/`libglib2.0-0` 之外若有变化，按报错补）。

构建成功后建议按顺序验证：`pytest tests/` → 单图推理 → `export.py --trt`（容器内建引擎）→
短跑一次训练（`--data <小数据集> --epochs 1`）。

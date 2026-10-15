# docker/ —— 开发镜像怎么用

> 这里只讲**怎么用**。镜像内容与设计取舍、换机验证清单在手册页 `docs/docker.md`。
> 一句话：`docker build -t flash-yolo:dev .` → `docker/run.sh`。

## 1. 三个文件的角色

| 文件 | 作用 |
|---|---|
| `../Dockerfile` | 镜像定义：py312 + torch 2.13.0+cu130 + CUDA 13 + TensorRT 11.4.0.106 + 依赖快照；**构建入口在仓库根目录** |
| `run.sh` | 唯一运行入口：一条命令进容器，自动带 GPU / 共享内存 / 属主对齐 / 仓库挂载 / 命令透传 |
| `constraints.txt` | 版本快照（`requirements*.txt` 定"装什么"，它钉"装哪个版本"），**只在构建期使用** |

镜像里同时装了 `requirements-dev.txt`（ultralytics / pytest），所以 `pytest tests/` 和
`scripts/compare_official.py` 这类开发期对照在容器里直接可用。

## 2. 三步走

```bash
# 0) 宿主前置：Docker + nvidia-container-toolkit + NVIDIA 驱动 >= 580（CUDA 13）
docker info | grep -i nvidia            # 应能看到 nvidia runtime

# 1) 构建（在仓库根目录；耗时主要在依赖与 TensorRT 的下载）
docker build -t flash-yolo:dev .
#    非阿里云网络：https 源只覆盖索引即可（trusted-host 保持默认；
#    别传空值——pip 对空 trusted-host 直接报错）
docker build -t flash-yolo:dev . --build-arg PIP_INDEX_URL=https://pypi.org/simple
#    http 自建镜像源：两个都要给，且 PIP_TRUSTED_HOST 是真实主机名
docker build -t flash-yolo:dev . \
    --build-arg PIP_INDEX_URL=http://mirror.example.com/pypi/simple --build-arg PIP_TRUSTED_HOST=mirror.example.com

# 2) 运行（仓库根目录已挂到 /workspace，工作目录就是它）
docker/run.sh                              # 交互式 shell
docker/run.sh pytest tests/ -q             # 任意命令透传，跑完容器自动删除

# 3) 验收四条链路（容器内）
pytest tests/ -q                                                   # 全量测试
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/traffic.jpg
python scripts/export.py --weights weights/yolo26n.safetensors --trt
python scripts/train.py --data <小数据集> --epochs 1               # 冒烟
```

## 3. `run.sh` 会自动带上什么

| 行为 | 说明 |
|---|---|
| `--gpus all` | 透传 GPU（**无 GPU 的机器删掉这一行**） |
| `--shm-size=8g` | DataLoader 多 worker 需要共享内存，Docker 默认 64M 会炸 |
| `--user $(id -u):$(id -g)` | bind mount 下产物属主与宿主机一致（**要临时 apt/pip 装东西就删掉这行**，改以 root 运行） |
| `-v <repo>:/workspace -w /workspace` | 整个仓库挂进去——`runs/`、`weights/`、`logs/` 的读写直接落回宿主机 |
| `DATA_DIR=/path docker/run.sh` | 数据集挂到 `/data`（相对路径会先归一化为绝对路径） |
| `IMAGE=xxx docker/run.sh` | 换镜像名（默认 `flash-yolo:dev`） |
| `--rm -i`（有 TTY 再加 `-t`） | 一次性容器；管道 / CI 下保持纯净输出 |

## 4. 数据、权重、产物分别在哪

`.dockerignore` 把 `runs/` `weights/` `logs/` `docs/` `config/datasets/local/` `.git` 挡在**镜像快照**
之外，但开发流程里整个仓库是 bind mount —— 所以：

- **开发流程（`run.sh`）**：宿主机上的 `weights/`、`config/datasets/local/`、`runs/` 容器内都看得见，
  训练/评估/导出的产物也直接写回宿主机，无需任何额外挂载；
- **数据集**：不进镜像也不进仓库，用 `DATA_DIR=` 挂到 `/data`，描述符里写 `/data/...`
  （`config/datasets/` 下只有模板，机器路径模板 `path: ""`）；
- **脱离挂载单独用镜像**（CI 基镜像等）：`/workspace` 是构建时拷进去的源码快照，可用
  `python scripts/download_weights.py --model yolo26n` 在容器内现取权重，但快照里没有
  `runs/` `weights/` `local/`。

## 5. 已知坑

1. **TensorRT 引擎与 GPU 架构 + TRT 版本绑定**：宿主机上 build 的 `.engine` 不能直接喂给容器
   （反之亦然）。容器里要用 TRT 就在容器里 `scripts/export.py --trt [--fp16]` 重建。
2. **权重不在镜像里**（只随 bind mount 可见）——想独立跑就先下载。
3. **多 worker 训练必须靠 `--shm-size`**：`run.sh` 已给 8g；手写 `docker run` 时别漏。
4. **pip 默认走 aliyun 源**：换网络环境用 `--build-arg` 覆盖（见第 2 节）。
5. **临时装包**要删 `run.sh` 的 `--user` 行，否则 `apt`/全局 `pip` 会因权限失败。

## 6. 首次在其他机器构建

base tag 是否存在、Python 版本、TensorRT 索引安装、apt 包名四个检查点，以及构建后的验收顺序，
见 `docs/docker.md` §5 —— 那份清单随报错更新，这里不复制一份以免两边漂移。

# Flash-YOLO 开发镜像 —— 复刻已验证的 py312 环境
# （Python 3.12 · torch 2.13.0+cu130 · CUDA 13.0 · TensorRT 11.4.0.106 · ultralytics 8.4.173）。
#
# 宿主要求：NVIDIA 驱动 >= 580（CUDA 13）+ nvidia-container-toolkit，运行时 --gpus all。
# 构建:  docker build -t flash-yolo:dev .
# 运行:  docker/run.sh [命令...]（交互式开发入口，见 docker/run.sh）

# base tag 以 Docker Hub 实际发布为准（需 Python 3.12）：2.13.0-cuda13.0-cudnn9 系列；
# 若无 -runtime 变体则改用 -devel（工具链等价、体积更大）。
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

# 默认走 aliyun 镜像（http，与 trusted-host 配对）；其他网络环境构建时覆盖：
#   https 源：--build-arg PIP_INDEX_URL=https://pypi.org/simple   （trusted-host 保持默认即可）
#   http 源：两个都要给，PIP_TRUSTED_HOST 必须是真实主机名
# 注意 PIP_TRUSTED_HOST 不能传空值——pip 对空 trusted-host 直接 ValueError，
# 构建会挂在下面装依赖那步（实测 pip 26.2.1）。
ARG PIP_INDEX_URL=http://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com

# libgl1 / libglib2.0-0：opencv-python（非 headless）的导入依赖
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 依赖安装：requirements.txt 定"装什么"，docker/constraints.txt 钉死版本（= py312 快照）。
# torch / torchvision 由 base 镜像提供，不会重装。
COPY requirements.txt requirements-dev.txt docker/constraints.txt /tmp/req/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
      -c /tmp/req/constraints.txt \
      -r /tmp/req/requirements.txt -r /tmp/req/requirements-dev.txt

# TensorRT 11.4.0.106（tensorrt-cu13 全家桶）。PyPI 上的同名包是 ~17KB 壳包、安装钩子会去
# pypi.nvidia.com 拉真实二进制且可能静默挂起 —— 直接走 NVIDIA 索引取真产物；关 build
# isolation 是因为 meta 包以源码分发（构建依赖 setuptools，先装好再从其他索引解析）。
RUN pip install --no-cache-dir setuptools wheel \
 && pip install --no-cache-dir --no-build-isolation \
      --index-url https://pypi.nvidia.com tensorrt-cu13==11.4.0.106
# 备选路线（手工已验证）：从 https://pypi.nvidia.com/<包名>/<文件名> curl 三个真实产物
# （tensorrt_cu13_libs wheel / tensorrt_cu13_bindings wheel / tensorrt_cu13 sdist），然后：
#   pip install --no-build-isolation tensorrt_cu13_libs-*.whl tensorrt_cu13_bindings-*.whl tensorrt_cu13-*.tar.gz

WORKDIR /workspace
# 镜像内自带一份源码快照（独立可用：推理 / demo / CI）；开发时 bind mount 覆盖此副本。
COPY . /workspace

CMD ["bash"]

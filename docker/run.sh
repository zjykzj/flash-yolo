#!/usr/bin/env bash
# Flash-YOLO 开发容器入口。
#   docker/run.sh                  交互式 shell
#   docker/run.sh pytest tests/    在容器内执行命令
# 先构建：docker build -t flash-yolo:dev .
# 机器本地数据目录：DATA_DIR=/path/to/datasets docker/run.sh
# 需要以 root 操作（临时 apt / pip 安装）时：删掉下面的 --user 一行。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-flash-yolo:dev}"

args=(--rm -i
      --gpus all                    # 无 GPU 的机器删掉这一行
      --shm-size=8g                 # DataLoader 多 worker 需要共享内存（Docker 默认 64M 不够）
      --user "$(id -u):$(id -g)"    # 绑定挂载下保持产物属主与宿主机一致
      -e HOME=/tmp
      -v "$REPO_ROOT":/workspace
      -w /workspace)
# 有 TTY 才加 -t（管道 / CI 场景下保持纯净输出）
[[ -t 0 && -t 1 ]] && args+=(-t)
# 数据目录挂到 /data（相对路径先归一化为绝对路径，docker -v 只收绝对路径）
[[ -n "${DATA_DIR:-}" ]] && args+=(-v "$(cd "$DATA_DIR" && pwd)":/data)

[[ $# -eq 0 ]] && set -- bash
exec docker run "${args[@]}" "$IMAGE" "$@"

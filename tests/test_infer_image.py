"""图片推理输出路径：`--output` 语义（单图文件/目录、目录模式）+ 端到端 CLI（FLASH_YOLO_SMOKE）

纯单测只覆盖 `_resolve_image_output`（无需权重）；门控用例训练迷你模型后跑
`scripts/infer.py --image`，验证"目录输出"与"显式文件名（labels 跟随同名）"两条路径。
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_infer():
    """按路径加载 scripts/infer.py（不是包，importlib 避免 sys.path 歧义）"""
    spec = importlib.util.spec_from_file_location("infer_script", ROOT / "scripts" / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_image_explicit_file(tmp_path):
    """单图 + 图片文件路径：文件名生效，父目录作为 run 目录"""
    infer = _load_infer()
    out = tmp_path / "renamed.jpg"
    run_dir, out_file = infer._resolve_image_output(str(out), multi=False)
    assert run_dir == tmp_path.resolve() and out_file == out.resolve()


def test_single_image_existing_directory(tmp_path):
    """单图 + 已存在目录：按目录处理，文件名沿用源图名"""
    infer = _load_infer()
    d = tmp_path / "results"
    d.mkdir()
    assert infer._resolve_image_output(str(d), multi=False) == (d.resolve(), None)


def test_single_image_new_directory_without_suffix(tmp_path):
    """单图 + 无后缀路径：按目录处理（此前会被当成文件路径、只取父目录）"""
    infer = _load_infer()
    d = tmp_path / "results"
    assert infer._resolve_image_output(str(d), multi=False) == (d.resolve(), None)


def test_single_image_dotted_directory_is_not_a_file(tmp_path):
    """带点的目录名（run.v1）不应被误判为文件"""
    infer = _load_infer()
    d = tmp_path / "run.v1"
    assert infer._resolve_image_output(str(d), multi=False) == (d.resolve(), None)


def test_directory_mode_requires_directory(tmp_path):
    """目录输入只接受目录输出；图片文件路径直接报错"""
    infer = _load_infer()
    d = tmp_path / "results"
    assert infer._resolve_image_output(str(d), multi=True) == (d.resolve(), None)
    file_like = tmp_path / "results.jpg"
    file_like.write_bytes(b"x")
    with pytest.raises(ValueError):
        infer._resolve_image_output(str(file_like), multi=True)


def test_directory_mode_accepts_directory_with_image_suffix(tmp_path):
    """已存在目录即使名字带图片后缀，也应按目录处理"""
    infer = _load_infer()
    d = tmp_path / "results.jpg"
    d.mkdir()
    assert infer._resolve_image_output(str(d), multi=True) == (d.resolve(), None)


@pytest.mark.skipif(os.environ.get("FLASH_YOLO_SMOKE") != "1",
                    reason="FLASH_YOLO_SMOKE=1 才执行图片推理 CLI 冒烟")
def test_infer_image_output_cli(tmp_path):
    """训练 1 轮迷你模型 -> 目录输出 + 显式文件名输出（含 labels 同名）"""
    import torch

    from config.train_config import TrainConfig
    from test_train_smoke import _make_mini_coco
    from train.trainer import Trainer

    data = _make_mini_coco(tmp_path, n_img=8, size=160)
    cfg = TrainConfig(
        data=str(data), epochs=1, batch=4, nbs=4, imgsz=160, workers=0, val_epochs=0,
        amp=False, mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, warmup_epochs=0.0,
        aug_samples=0, plots=False, save_period=0, diag_interval=0,
    )
    torch.manual_seed(0)
    run = tmp_path / "train"
    Trainer(cfg, device="cpu", run_dir=run).train()
    weights = run / "weights" / "last.safetensors"

    def infer(output):
        subprocess.run(
            [sys.executable, "scripts/infer.py", "--weights", str(weights),
             "--image", "assets/traffic.jpg", "--output", str(output), "--imgsz", "160"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )

    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    infer(out_dir)
    assert (out_dir / "traffic.jpg").exists(), "directory output missing"
    assert (out_dir / "labels" / "traffic.txt").exists(), "labels missing"

    out_file = tmp_path / "renamed.png"
    infer(out_file)
    assert out_file.exists(), "explicit output filename was not honored"
    assert (tmp_path / "labels" / "renamed.txt").exists(), "labels should follow the output stem"

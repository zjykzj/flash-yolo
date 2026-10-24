"""config 包：配置的单一去处——值在 yaml，机制在 py

    inference.py          推理/评估默认值：IMGSZ · CONF_THRES · IOU_THRES · MAX_DET
    models/<model>.yaml   模型结构（yolo26.yaml 起；纯数据目录，无 __init__.py）
    datasets/<ds>.yaml    数据集描述符（format/path/names + train·val 角色块；spec.py 解析校验，
                          load_dataset()/load_names() 读取；local/ 为本机覆盖目录，gitignored）
    train.yaml            训练超参基线 + recipes/<name>.yaml 命名配方（base + 按 scale 分档增量）
    train_config.py       训练配置机制：TrainConfig + 合并优先序（yaml -> 配方 -> CLI）

__version__ / __repo__ 为项目身份的单一事实源（`from config import __version__, __repo__`）——版本号
另用于 README badge / CHANGELOG，两者都作为出处信息写进权重 metadata（model/weights.py::
provenance_meta）；其余一律显式子模块导入，不做包级再导出。
"""

__version__ = "0.7.0"
__repo__ = "https://github.com/zjykzj/flash-yolo.git"

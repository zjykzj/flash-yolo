"""任务头（task heads）：按「结构家族 × 任务」分文件

家族短名对照表（文件名里的家族名 -> config/models 里的架构）：
    yolo26_*  ->  yolo26 / yolo26-seg / flash-yolo（flash-yolo 复用 yolo26 系的 Detect 头）
    v3_*      ->  yolov3-tiny / yolov3-tiny-silu（silu 只换算子，头不动）

命名规则（新增头时照此办理）：
    1. 文件名 = <家族>_<任务>.py；任务取官方词表（detect / seg / pose / obb / cls / depth / sem）
    2. 变体不进文件名——silu / 档位 / head 参数都由 yaml 决定，不产生新文件
    3. 无家族的纯机制头（如将来的 RT-DETR decoder）用 <机制>.py
    4. 类名贴官方（Detect / Segment26 / Proto26），便于与官方源码、权重、yaml 三方对账
    5. 共用的头专属件先跟首个使用者走，出现第二个消费者再升级为独立文件

分层的触发条件（现在不建）：某家族 >= 3 个头文件，或本目录平铺 >= 8 个文件。
"""

from model.head.v3_detect import V3Detect, decode_level
from model.head.yolo26_detect import Detect

__all__ = ["Detect", "V3Detect", "decode_level"]

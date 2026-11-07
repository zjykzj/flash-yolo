"""评估器入口：按协议分派——COCO（pycocotools）/ LVIS（faster_coco_eval lvis_style）"""

from eval.coco_evaluator import CocoEvaluator, build_vocab_map, encode_mask_rle
from eval.lvis_evaluator import LvisEvaluator

__all__ = ["CocoEvaluator", "LvisEvaluator", "build_evaluator", "build_vocab_map", "encode_mask_rle"]


def build_evaluator(gt, nc=None, iou_type="bbox", eval_style="coco", max_dets=None):
    """评估器工厂：COCO 协议 -> CocoEvaluator；LVIS 协议 -> LvisEvaluator

    LVIS 的 maxDets 默认 300（官方口径）；与 Ultralytics 对比时可显式传 100。
    """
    if eval_style == "lvis":
        return LvisEvaluator(gt, nc=nc, iou_type=iou_type,
                             max_dets=300 if max_dets is None else max_dets)
    if eval_style == "coco":
        return CocoEvaluator(gt, nc=nc, iou_type=iou_type)
    raise ValueError(f"eval_style must be 'coco' or 'lvis', got {eval_style!r}")

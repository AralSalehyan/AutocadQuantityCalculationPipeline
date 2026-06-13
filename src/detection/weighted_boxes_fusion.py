from src.geometry.primitives import DetectionObject


def weighted_boxes_fusion(detections: list[DetectionObject], iou_threshold: float = 0.5) -> list[DetectionObject]:
    # Placeholder for later improvement. NMS remains the Phase 2 default.
    from src.detection.nms import nms

    return nms(detections, iou_threshold)


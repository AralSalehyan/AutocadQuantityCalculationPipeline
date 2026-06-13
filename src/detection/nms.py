from src.geometry.geometry_utils import bbox_iou
from src.geometry.primitives import DetectionObject


def nms(detections: list[DetectionObject], iou_threshold: float = 0.5) -> list[DetectionObject]:
    grouped: dict[str, list[DetectionObject]] = {}
    passthrough: list[DetectionObject] = []
    for detection in detections:
        if detection.geometry_type != "bbox":
            passthrough.append(detection)
            continue
        grouped.setdefault(detection.type, []).append(detection)

    kept: list[DetectionObject] = []
    for items in grouped.values():
        candidates = sorted(items, key=lambda item: item.confidence or 0.0, reverse=True)
        while candidates:
            current = candidates.pop(0)
            kept.append(current)
            candidates = [
                item
                for item in candidates
                if bbox_iou(current.geometry, item.geometry) < iou_threshold
            ]
    return [*kept, *passthrough]


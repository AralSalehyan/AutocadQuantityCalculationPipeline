from pathlib import Path

from src.detection.base_detector import BaseDetector
from src.detection.nms import nms
from src.detection.weighted_boxes_fusion import weighted_boxes_fusion
from src.geometry.primitives import DetectionObject


class EnsembleDetector(BaseDetector):
    def __init__(
        self,
        detectors: list[BaseDetector],
        merge_method: str = "nms",
        iou_threshold: float = 0.5,
    ):
        self.detectors = detectors
        self.merge_method = merge_method
        self.iou_threshold = iou_threshold

    def predict(self, image_path: Path) -> list[DetectionObject]:
        detections: list[DetectionObject] = []
        for detector in self.detectors:
            detections.extend(detector.predict(image_path))
        if self.merge_method == "weighted_boxes_fusion":
            return weighted_boxes_fusion(detections, self.iou_threshold)
        return nms(detections, self.iou_threshold)


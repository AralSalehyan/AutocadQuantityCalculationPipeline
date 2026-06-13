from pathlib import Path

from src.detection.base_detector import BaseDetector
from src.detection.ensemble_detector import EnsembleDetector
from src.detection.mock_detector import MockDetector
from src.detection.rtdetr_detector import RTDETRDetector
from src.detection.yolo_detector import YOLODetector
from src.utils.config import deep_get


def create_detector(name: str, config: dict) -> BaseDetector:
    normalized = name.lower().strip()
    if normalized == "mock":
        return MockDetector()
    if normalized == "yolo":
        return YOLODetector(
            model_path=Path(deep_get(config, "yolo.model_path", "models/yolo/door_window.pt")),
            confidence_threshold=float(deep_get(config, "yolo.confidence_threshold", 0.25)),
        )
    if normalized == "rtdetr":
        return RTDETRDetector(
            model_path=Path(deep_get(config, "rtdetr.model_path", "models/rtdetr/door_window.pt")),
            confidence_threshold=float(deep_get(config, "rtdetr.confidence_threshold", 0.25)),
        )
    raise ValueError(f"Unknown detector: {name}")


def create_ensemble(detector_names: list[str], config: dict) -> EnsembleDetector:
    detectors = [create_detector(name, config) for name in detector_names]
    return EnsembleDetector(
        detectors=detectors,
        merge_method=str(deep_get(config, "ensemble.merge_method", "nms")),
        iou_threshold=float(deep_get(config, "ensemble.iou_threshold", 0.5)),
    )


from pathlib import Path

from src.detection.base_detector import BaseDetector
from src.detection.model_loading import ModelUnavailableError, require_model_file
from src.detection.yolo_detector import _parse_ultralytics_results
from src.geometry.primitives import DetectionObject
from src.utils.runtime_paths import configure_workspace_runtime_dirs


class RTDETRDetector(BaseDetector):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None

    def predict(self, image_path: Path) -> list[DetectionObject]:
        model = self._load_model()
        results = model.predict(str(image_path), conf=self.confidence_threshold, verbose=False)
        return _parse_ultralytics_results(results, "rtdetr")

    def _load_model(self):
        require_model_file(self.model_path)
        if self._model is None:
            try:
                configure_workspace_runtime_dirs()
                from ultralytics import RTDETR
            except ImportError as exc:
                raise ModelUnavailableError("ultralytics is not installed; cannot run RT-DETR detector.") from exc
            self._model = RTDETR(str(self.model_path))
        return self._model

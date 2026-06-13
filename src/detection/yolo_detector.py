from pathlib import Path

from src.detection.base_detector import BaseDetector
from src.detection.model_loading import ModelUnavailableError, require_model_file
from src.geometry.primitives import DetectionObject
from src.utils.runtime_paths import configure_workspace_runtime_dirs
from src.utils.ids import new_id


class YOLODetector(BaseDetector):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None

    def predict(self, image_path: Path) -> list[DetectionObject]:
        model = self._load_model()
        results = model.predict(str(image_path), conf=self.confidence_threshold, verbose=False)
        return _parse_ultralytics_results(results, "yolo")

    def _load_model(self):
        require_model_file(self.model_path)
        if self._model is None:
            try:
                configure_workspace_runtime_dirs()
                from ultralytics import YOLO
            except ImportError as exc:
                raise ModelUnavailableError("ultralytics is not installed; cannot run YOLO detector.") from exc
            self._model = YOLO(str(self.model_path))
        return self._model


def _parse_ultralytics_results(results, source: str) -> list[DetectionObject]:
    objects: list[DetectionObject] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            confidence = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0])
            class_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0])
            label = str(names.get(class_id, class_id)).lower()
            object_type = _class_to_type(label)
            if object_type not in {"door", "window"}:
                continue
            xyxy = box.xyxy[0].tolist()
            objects.append(
                DetectionObject(
                    id=new_id(object_type),
                    type=object_type,
                    source=source,
                    geometry_type="bbox",
                    geometry={
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2),
                    },
                    label=label,
                    confidence=round(confidence, 4),
                    metadata={"model_path": source},
                )
            )
    return objects


def _class_to_type(label: str) -> str:
    normalized = label.lower()
    if "door" in normalized or "kapi" in normalized or "kapı" in normalized:
        return "door"
    if "window" in normalized or "pencere" in normalized:
        return "window"
    return normalized

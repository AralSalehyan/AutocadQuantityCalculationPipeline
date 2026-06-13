from pathlib import Path

from src.geometry.primitives import DetectionObject


class BaseDetector:
    def predict(self, image_path: Path) -> list[DetectionObject]:
        raise NotImplementedError


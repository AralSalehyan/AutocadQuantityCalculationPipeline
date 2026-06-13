from pathlib import Path

from src.geometry.primitives import DetectionObject


class BaseSegmenter:
    def predict(self, image_path: Path) -> list[DetectionObject]:
        raise NotImplementedError


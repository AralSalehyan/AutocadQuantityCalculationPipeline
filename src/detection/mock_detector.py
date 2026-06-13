from pathlib import Path

from PIL import Image

from src.detection.base_detector import BaseDetector
from src.geometry.primitives import DetectionObject
from src.utils.ids import new_id


class MockDetector(BaseDetector):
    def predict(self, image_path: Path) -> list[DetectionObject]:
        with Image.open(image_path) as image:
            width, height = image.size

        boxes = [
            ("door", [0.47, 0.28, 0.54, 0.33], 0.92),
            ("door", [0.18, 0.62, 0.25, 0.68], 0.88),
            ("window", [0.70, 0.09, 0.84, 0.14], 0.90),
            ("window", [0.05, 0.41, 0.11, 0.55], 0.86),
        ]
        objects: list[DetectionObject] = []
        for object_type, rel_box, confidence in boxes:
            x1, y1, x2, y2 = rel_box
            objects.append(
                DetectionObject(
                    id=new_id(object_type),
                    type=object_type,
                    source="mock",
                    geometry_type="bbox",
                    geometry={
                        "x1": round(x1 * width, 2),
                        "y1": round(y1 * height, 2),
                        "x2": round(x2 * width, 2),
                        "y2": round(y2 * height, 2),
                    },
                    label=object_type,
                    confidence=confidence,
                    metadata={"model": "mock_detector"},
                )
            )
        return objects


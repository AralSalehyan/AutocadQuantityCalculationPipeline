from pathlib import Path

from PIL import Image

from src.geometry.primitives import DetectionObject
from src.segmentation.base_segmenter import BaseSegmenter
from src.utils.ids import new_id


class MockSegmenter(BaseSegmenter):
    def predict(self, image_path: Path) -> list[DetectionObject]:
        with Image.open(image_path) as image:
            width, height = image.size

        left_room = [
            [0.12 * width, 0.16 * height],
            [0.48 * width, 0.16 * height],
            [0.48 * width, 0.78 * height],
            [0.12 * width, 0.78 * height],
        ]
        right_room = [
            [0.52 * width, 0.16 * height],
            [0.88 * width, 0.16 * height],
            [0.88 * width, 0.78 * height],
            [0.52 * width, 0.78 * height],
        ]
        wall_lines = [
            [[0.10 * width, 0.14 * height], [0.90 * width, 0.14 * height], [0.90 * width, 0.80 * height], [0.10 * width, 0.80 * height], [0.10 * width, 0.14 * height]],
            [[0.50 * width, 0.14 * height], [0.50 * width, 0.80 * height]],
            [[0.10 * width, 0.48 * height], [0.50 * width, 0.48 * height]],
        ]

        objects = [
            DetectionObject(
                id=new_id("room"),
                type="room",
                source="mock",
                geometry_type="polygon",
                geometry={"points": _round_points(left_room)},
                label="Room 1",
                confidence=0.80,
                metadata={"model": "mock_segmenter"},
            ),
            DetectionObject(
                id=new_id("room"),
                type="room",
                source="mock",
                geometry_type="polygon",
                geometry={"points": _round_points(right_room)},
                label="Room 2",
                confidence=0.80,
                metadata={"model": "mock_segmenter"},
            ),
        ]
        for wall in wall_lines:
            objects.append(
                DetectionObject(
                    id=new_id("wall"),
                    type="wall",
                    source="mock",
                    geometry_type="polyline",
                    geometry={"points": _round_points(wall)},
                    label="Wall",
                    confidence=0.75,
                    metadata={"model": "mock_segmenter"},
                )
            )
        return objects


def _round_points(points: list[list[float]]) -> list[list[float]]:
    return [[round(x, 2), round(y, 2)] for x, y in points]


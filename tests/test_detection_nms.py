from pathlib import Path

from src.detection.nms import nms
from src.geometry.primitives import DetectionObject, TileInfo
from src.preprocessing.tile_merger import TileMerger


def test_restore_detection_coordinates() -> None:
    tile = TileInfo(id="tile_0001", image_path=str(Path("tile.png")), x_offset=100, y_offset=200, width=300, height=300)
    detection = DetectionObject(
        id="door_1",
        type="door",
        source="mock",
        geometry_type="bbox",
        geometry={"x1": 10, "y1": 20, "x2": 40, "y2": 60},
        confidence=0.9,
    )

    restored = TileMerger().restore_detection_coordinates([detection], tile)

    assert restored[0].geometry == {"x1": 110, "y1": 220, "x2": 140, "y2": 260}


def test_nms_keeps_highest_confidence_overlapping_box() -> None:
    detections = [
        DetectionObject(
            id="door_low",
            type="door",
            source="yolo",
            geometry_type="bbox",
            geometry={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            confidence=0.5,
        ),
        DetectionObject(
            id="door_high",
            type="door",
            source="rtdetr",
            geometry_type="bbox",
            geometry={"x1": 10, "y1": 10, "x2": 110, "y2": 110},
            confidence=0.9,
        ),
        DetectionObject(
            id="window",
            type="window",
            source="yolo",
            geometry_type="bbox",
            geometry={"x1": 10, "y1": 10, "x2": 110, "y2": 110},
            confidence=0.7,
        ),
    ]

    kept = nms(detections, iou_threshold=0.5)

    assert {item.id for item in kept} == {"door_high", "window"}


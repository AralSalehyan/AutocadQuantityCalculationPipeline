import numpy as np

from src.segmentation.mask_to_geometry import mask_to_polygons
from src.segmentation.room_polygonizer import room_mask_to_polygons
from src.segmentation.wall_skeletonizer import wall_mask_to_centerlines


def test_mask_to_polygons_extracts_rectangle() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:16, 5:25] = 1

    polygons = mask_to_polygons(mask)

    assert len(polygons) == 1
    assert len(polygons[0]) >= 3


def test_room_mask_to_polygons_returns_room_objects() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:16, 5:25] = 1

    rooms = room_mask_to_polygons(mask, source="segformer", confidence=0.8)

    assert len(rooms) == 1
    assert rooms[0].type == "room"
    assert rooms[0].source == "segformer"
    assert rooms[0].geometry_type == "polygon"


def test_wall_mask_to_centerlines_returns_wall_objects() -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[14:17, 3:27] = 1

    walls = wall_mask_to_centerlines(mask, source="segformer", confidence=0.7)

    assert walls
    assert walls[0].type == "wall"
    assert walls[0].geometry_type == "polyline"
    assert len(walls[0].geometry["points"]) >= 2


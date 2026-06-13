from __future__ import annotations

import numpy as np

from src.geometry.primitives import DetectionObject
from src.segmentation.mask_to_geometry import mask_to_polygons
from src.utils.ids import new_id


def room_mask_to_polygons(mask: np.ndarray, source: str = "segmentation", confidence: float | None = None) -> list[DetectionObject]:
    objects: list[DetectionObject] = []
    for index, polygon in enumerate(mask_to_polygons(mask), start=1):
        objects.append(
            DetectionObject(
                id=new_id("room"),
                type="room",
                source=source,
                geometry_type="polygon",
                geometry={"points": [[round(x, 2), round(y, 2)] for x, y in polygon]},
                label=f"Room {index}",
                confidence=confidence,
                metadata={"method": "mask_to_polygons"},
            )
        )
    return objects


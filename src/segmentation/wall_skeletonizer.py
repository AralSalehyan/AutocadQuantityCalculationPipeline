from __future__ import annotations

import math

import numpy as np

from src.geometry.primitives import DetectionObject
from src.utils.ids import new_id


def wall_mask_to_centerlines(
    mask: np.ndarray,
    source: str = "segmentation",
    confidence: float | None = None,
    min_length_px: float = 32.0,
) -> list[DetectionObject]:
    binary = (mask > 0).astype("uint8")
    if binary.max() == 0:
        return []

    try:
        from skimage.measure import find_contours
        from skimage.morphology import skeletonize
    except Exception:
        return _bbox_centerline(binary, source, confidence)

    skeleton = skeletonize(binary > 0).astype("uint8")
    hough_objects = _hough_centerlines(skeleton, source, confidence, min_length_px)
    if hough_objects:
        return hough_objects

    contours = find_contours(skeleton, level=0.5)
    objects: list[DetectionObject] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        points = [[round(float(col), 2), round(float(row), 2)] for row, col in contour]
        if len(points) >= 2 and _polyline_length(points) >= min_length_px:
            objects.append(
                DetectionObject(
                    id=new_id("wall"),
                    type="wall",
                    source=source,
                    geometry_type="polyline",
                    geometry={"points": points},
                    label="Wall",
                    confidence=confidence,
                    metadata={"method": "skeletonize"},
                )
            )
    return objects or _bbox_centerline(binary, source, confidence)


def _hough_centerlines(
    skeleton: np.ndarray,
    source: str,
    confidence: float | None,
    min_length_px: float,
) -> list[DetectionObject]:
    try:
        import cv2
    except ImportError:
        return []

    image = (skeleton > 0).astype("uint8") * 255
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=max(16, int(min_length_px)),
        maxLineGap=18,
    )
    if lines is None:
        return []

    objects: list[DetectionObject] = []
    seen: set[tuple[int, int, int, int]] = set()
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length_px:
            continue
        key = tuple(round(value / 4) for value in (x1, y1, x2, y2))
        reverse_key = (key[2], key[3], key[0], key[1])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        objects.append(
            DetectionObject(
                id=new_id("wall"),
                type="wall",
                source=source,
                geometry_type="polyline",
                geometry={"points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]]},
                label="Wall",
                confidence=confidence,
                metadata={"method": "hough_skeleton", "length_px": round(length, 2)},
            )
        )
    return objects


def _polyline_length(points: list[list[float]]) -> float:
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _bbox_centerline(binary: np.ndarray, source: str, confidence: float | None) -> list[DetectionObject]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return []
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    if (x2 - x1) >= (y2 - y1):
        points = [[x1, (y1 + y2) / 2.0], [x2, (y1 + y2) / 2.0]]
    else:
        points = [[(x1 + x2) / 2.0, y1], [(x1 + x2) / 2.0, y2]]
    return [
        DetectionObject(
            id=new_id("wall"),
            type="wall",
            source=source,
            geometry_type="polyline",
            geometry={"points": [[round(x, 2), round(y, 2)] for x, y in points]},
            label="Wall",
            confidence=confidence,
            metadata={"method": "bbox_centerline"},
        )
    ]

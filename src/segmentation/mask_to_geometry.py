from __future__ import annotations

import numpy as np


def mask_to_polygons(mask: np.ndarray, min_area: float = 16.0) -> list[list[list[float]]]:
    binary = (mask > 0).astype("uint8")
    if binary.max() == 0:
        return []

    try:
        import cv2
    except ImportError:
        return _mask_to_bbox_polygon(binary) if binary.any() else []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [[float(point[0][0]), float(point[0][1])] for point in approx]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def _mask_to_bbox_polygon(binary: np.ndarray) -> list[list[list[float]]]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return []
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]


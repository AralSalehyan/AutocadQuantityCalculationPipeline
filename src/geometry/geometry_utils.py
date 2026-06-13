from math import hypot


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def polygon_perimeter(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    closed = points + [points[0]]
    return polyline_length(closed)


def polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += hypot(end[0] - start[0], end[1] - start[1])
    return total


def bbox_iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1, ax2, ay2 = _bbox_edges(box_a)
    bx1, by1, bx2, by2 = _bbox_edges(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_center(box: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox_edges(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_dimensions(box: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox_edges(box)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def point_to_segment_distance(point: tuple[float, float], start: list[float], end: list[float]) -> float:
    px, py = point
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    projection = (x1 + t * dx, y1 + t * dy)
    return point_distance(point, projection)


def point_to_polyline_distance(point: tuple[float, float], points: list[list[float]]) -> float:
    if len(points) < 2:
        return float("inf")
    return min(point_to_segment_distance(point, start, end) for start, end in zip(points, points[1:]))


def bbox_to_polyline_distance(box: dict, points: list[list[float]]) -> float:
    center = bbox_center(box)
    width, height = bbox_dimensions(box)
    return max(0.0, point_to_polyline_distance(center, points) - max(width, height) / 2.0)


def polygon_boundary_distance(point: tuple[float, float], polygon: list[list[float]]) -> float:
    if len(polygon) < 2:
        return float("inf")
    closed = polygon if polygon[0] == polygon[-1] else [*polygon, polygon[0]]
    return point_to_polyline_distance(point, closed)


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def scale_length(value: float, scale_ratio: float | None) -> float:
    return value * scale_ratio if scale_ratio is not None else value


def scale_area(value: float, scale_ratio: float | None) -> float:
    return value * scale_ratio * scale_ratio if scale_ratio is not None else value


def _bbox_edges(box: dict) -> tuple[float, float, float, float]:
    if {"x1", "y1", "x2", "y2"}.issubset(box):
        return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    if {"x", "y", "width", "height"}.issubset(box):
        x = float(box["x"])
        y = float(box["y"])
        return x, y, x + float(box["width"]), y + float(box["height"])
    raise ValueError(f"Unsupported bbox shape: {box}")

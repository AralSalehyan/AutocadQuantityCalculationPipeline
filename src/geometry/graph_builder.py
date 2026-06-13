from src.geometry.geometry_utils import (
    bbox_center,
    bbox_to_polyline_distance,
    point_distance,
    point_in_polygon,
    point_to_polyline_distance,
    polygon_boundary_distance,
)
from src.geometry.primitives import DetectionObject


class DrawingGraphBuilder:
    def __init__(self, room_wall_threshold: float = 60.0, opening_wall_threshold: float = 80.0, text_near_threshold: float = 120.0):
        self.room_wall_threshold = room_wall_threshold
        self.opening_wall_threshold = opening_wall_threshold
        self.text_near_threshold = text_near_threshold

    def build(self, objects: list[DetectionObject]) -> dict:
        nodes = [
            {
                "id": item.id,
                "type": item.type,
                "source": item.source,
                "label": item.label,
                "confidence": item.confidence,
            }
            for item in objects
        ]
        edges: list[dict] = []
        rooms = [item for item in objects if item.type == "room" and item.geometry_type == "polygon"]
        walls = [item for item in objects if item.type == "wall" and item.geometry_type == "polyline"]
        texts = [item for item in objects if item.type.endswith("_text")]
        openings = [item for item in objects if item.type in {"door", "window"} and item.geometry_type == "bbox"]

        for room in rooms:
            polygon = room.geometry.get("points", [])
            for text in texts:
                center = _object_center(text)
                if center and point_in_polygon(center, polygon):
                    _add_edge(edges, room.id, text.id, "contains", confidence=text.confidence)
            for opening in openings:
                center = _object_center(opening)
                if center and point_in_polygon(center, polygon):
                    _add_edge(edges, room.id, opening.id, "contains_opening", confidence=opening.confidence)
            for wall in walls:
                distance = _room_wall_distance(room, wall)
                if distance <= self.room_wall_threshold:
                    _add_edge(edges, room.id, wall.id, "adjacent_to", distance=round(distance, 3))

        for wall in walls:
            wall_points = wall.geometry.get("points", [])
            for opening in openings:
                distance = bbox_to_polyline_distance(opening.geometry, wall_points)
                if distance <= self.opening_wall_threshold:
                    _add_edge(edges, wall.id, opening.id, "has_opening", distance=round(distance, 3), confidence=opening.confidence)
                    _add_edge(edges, opening.id, wall.id, "near_wall", distance=round(distance, 3), confidence=opening.confidence)

        non_text_objects = [item for item in objects if not item.type.endswith("_text")]
        for text in texts:
            text_center = _object_center(text)
            if text_center is None:
                continue
            for item in non_text_objects:
                distance = _object_distance_from_point(item, text_center)
                if distance <= self.text_near_threshold:
                    _add_edge(edges, text.id, item.id, "near", distance=round(distance, 3), confidence=text.confidence)

        return {"nodes": nodes, "edges": edges}


def _object_center(item: DetectionObject) -> tuple[float, float] | None:
    if item.geometry_type == "bbox":
        return bbox_center(item.geometry)
    if item.geometry_type == "point":
        return (float(item.geometry["x"]), float(item.geometry["y"]))
    return None


def _room_wall_distance(room: DetectionObject, wall: DetectionObject) -> float:
    polygon = room.geometry.get("points", [])
    wall_points = wall.geometry.get("points", [])
    if not polygon or not wall_points:
        return float("inf")
    distances = []
    for point in wall_points:
        as_tuple = (float(point[0]), float(point[1]))
        distances.append(0.0 if point_in_polygon(as_tuple, polygon) else polygon_boundary_distance(as_tuple, polygon))
    return min(distances) if distances else float("inf")


def _object_distance_from_point(item: DetectionObject, point: tuple[float, float]) -> float:
    if item.geometry_type == "bbox":
        return point_distance(point, bbox_center(item.geometry))
    if item.geometry_type == "polygon":
        polygon = item.geometry.get("points", [])
        return 0.0 if point_in_polygon(point, polygon) else polygon_boundary_distance(point, polygon)
    if item.geometry_type == "polyline":
        return point_to_polyline_distance(point, item.geometry.get("points", []))
    center = _object_center(item)
    return point_distance(point, center) if center else float("inf")


def _add_edge(edges: list[dict], source: str, target: str, relationship: str, **metadata) -> None:
    candidate = {"source": source, "target": target, "relationship": relationship}
    candidate.update({key: value for key, value in metadata.items() if value is not None})
    if candidate not in edges:
        edges.append(candidate)

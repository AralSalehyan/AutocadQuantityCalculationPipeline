from src.geometry.graph_builder import DrawingGraphBuilder
from src.geometry.primitives import DetectionObject
from src.vector.vector_raster_fusion import VectorRasterFusion


def test_fusion_prefers_vector_walls_and_merges_opening_evidence() -> None:
    room = _room("room_1", [[0, 0], [300, 0], [300, 200], [0, 200]])
    segmented_wall = _wall("seg_wall", "segformer", [[0, 0], [300, 0]], 0.70)
    vector_wall = _wall("vec_wall", "dxf_vector", [[0, 0], [300, 0]], 0.85)
    vector_door = _bbox("vec_door", "door", "dxf_vector", [95, 80, 145, 130], 0.85)
    raster_door = _bbox("raster_door", "door", "yolo", [100, 85, 150, 135], 0.90)
    text = _bbox("text_1", "room_text", "dxf_vector", [40, 40, 100, 58], 0.85, label="KITCHEN")

    fused = VectorRasterFusion().fuse([raster_door], [room, segmented_wall], [vector_wall, vector_door], [text])

    walls = [item for item in fused if item.type == "wall"]
    doors = [item for item in fused if item.type == "door"]
    fused_room = next(item for item in fused if item.id == "room_1")
    fused_text = next(item for item in fused if item.id == "text_1")

    assert [item.id for item in walls] == ["vec_wall"]
    assert len(doors) == 1
    assert doors[0].source == "fusion"
    assert doors[0].confidence == 0.98
    assert set(doors[0].metadata["source_evidence"]) == {"vec_door", "raster_door"}
    assert fused_text.metadata["attached_room_id"] == "room_1"
    assert "text_1" in fused_room.metadata["attached_text_ids"]
    assert fused_room.label == "KITCHEN"


def test_graph_builder_adds_room_wall_opening_and_text_relationships() -> None:
    room = _room("room_1", [[0, 0], [300, 0], [300, 200], [0, 200]])
    wall = _wall("wall_1", "dxf_vector", [[0, 0], [300, 0]], 0.85)
    door = _bbox("door_1", "door", "fusion", [120, -10, 160, 30], 0.95)
    text = _bbox("text_1", "room_text", "dxf_vector", [40, 40, 100, 58], 0.85, label="KITCHEN")

    graph = DrawingGraphBuilder().build([room, wall, door, text])
    relationships = {(edge["source"], edge["target"], edge["relationship"]) for edge in graph["edges"]}

    assert ("room_1", "text_1", "contains") in relationships
    assert ("room_1", "wall_1", "adjacent_to") in relationships
    assert ("wall_1", "door_1", "has_opening") in relationships
    assert ("door_1", "wall_1", "near_wall") in relationships
    assert ("text_1", "room_1", "near") in relationships


def _room(object_id: str, points: list[list[float]]) -> DetectionObject:
    return DetectionObject(
        id=object_id,
        type="room",
        source="segformer",
        geometry_type="polygon",
        geometry={"points": points},
        label="Room 1",
        confidence=0.80,
    )


def _wall(object_id: str, source: str, points: list[list[float]], confidence: float) -> DetectionObject:
    return DetectionObject(
        id=object_id,
        type="wall",
        source=source,
        geometry_type="polyline",
        geometry={"points": points},
        label="Wall",
        confidence=confidence,
    )


def _bbox(object_id: str, object_type: str, source: str, coords: list[float], confidence: float, label: str | None = None) -> DetectionObject:
    return DetectionObject(
        id=object_id,
        type=object_type,
        source=source,
        geometry_type="bbox",
        geometry={"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
        label=label or object_type,
        confidence=confidence,
    )

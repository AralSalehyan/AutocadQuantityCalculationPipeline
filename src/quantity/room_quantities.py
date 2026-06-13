from src.geometry.geometry_utils import polygon_area, polygon_perimeter, scale_area, scale_length
from src.geometry.primitives import DetectionObject, QuantityItem
from src.utils.ids import new_id


def calculate_room_quantities(objects: list[DetectionObject], scale_ratio: float | None) -> list[QuantityItem]:
    room_items = [item for item in objects if item.type == "room" and item.geometry_type == "polygon"]
    quantities: list[QuantityItem] = []
    unit = "m" if scale_ratio is not None else "px"
    area_unit = "m2" if scale_ratio is not None else "px2"
    for index, room in enumerate(room_items, start=1):
        points = room.geometry.get("points", [])
        room_name = room.label or f"Room {index}"
        raw_area = polygon_area(points)
        raw_perimeter = polygon_perimeter(points)
        quantities.append(
            QuantityItem(
                id=new_id("qty_room_area"),
                category="room_area",
                name=f"{room_name} area",
                quantity=round(scale_area(raw_area, scale_ratio), 4),
                unit=area_unit,
                source_object_ids=[room.id],
                confidence=room.confidence,
                calculation={"raw_area_px2": raw_area, "scale_ratio": scale_ratio},
            )
        )
        quantities.append(
            QuantityItem(
                id=new_id("qty_room_perimeter"),
                category="room_perimeter",
                name=f"{room_name} perimeter",
                quantity=round(scale_length(raw_perimeter, scale_ratio), 4),
                unit=unit,
                source_object_ids=[room.id],
                confidence=room.confidence,
                calculation={"raw_perimeter_px": raw_perimeter, "scale_ratio": scale_ratio},
            )
        )
    return quantities


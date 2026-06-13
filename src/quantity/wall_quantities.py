from src.geometry.geometry_utils import polyline_length, scale_length
from src.geometry.primitives import DetectionObject, QuantityItem
from src.utils.ids import new_id


def calculate_wall_quantities(objects: list[DetectionObject], scale_ratio: float | None) -> list[QuantityItem]:
    walls = [item for item in objects if item.type == "wall" and item.geometry_type == "polyline"]
    unit = "m" if scale_ratio is not None else "px"
    quantities: list[QuantityItem] = []
    raw_total = 0.0
    source_ids: list[str] = []
    for index, wall in enumerate(walls, start=1):
        raw_length = polyline_length(wall.geometry.get("points", []))
        raw_total += raw_length
        source_ids.append(wall.id)
        quantities.append(
            QuantityItem(
                id=new_id("qty_wall_length"),
                category="wall_length",
                name=f"Wall {index}",
                quantity=round(scale_length(raw_length, scale_ratio), 4),
                unit=unit,
                source_object_ids=[wall.id],
                confidence=wall.confidence,
                calculation={"raw_length_px": raw_length, "scale_ratio": scale_ratio},
            )
        )
    if walls:
        quantities.append(
            QuantityItem(
                id=new_id("qty_wall_total"),
                category="wall_length_total",
                name="Gross wall length",
                quantity=round(scale_length(raw_total, scale_ratio), 4),
                unit=unit,
                source_object_ids=source_ids,
                confidence=min([wall.confidence or 0.0 for wall in walls]),
                calculation={"raw_length_px": raw_total, "scale_ratio": scale_ratio},
            )
        )
    return quantities


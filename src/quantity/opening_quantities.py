from src.geometry.primitives import DetectionObject, QuantityItem
from src.utils.ids import new_id


def calculate_opening_quantities(objects: list[DetectionObject]) -> list[QuantityItem]:
    quantities: list[QuantityItem] = []
    for object_type in ("door", "window"):
        items = [item for item in objects if item.type == object_type]
        if not items:
            continue
        quantities.append(
            QuantityItem(
                id=new_id(f"qty_{object_type}_count"),
                category=f"{object_type}_count",
                name=f"{object_type.title()} count",
                quantity=float(len(items)),
                unit="count",
                source_object_ids=[item.id for item in items],
                confidence=round(sum(item.confidence or 0.0 for item in items) / len(items), 4),
                calculation={"count": len(items)},
            )
        )
    return quantities


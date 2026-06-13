from src.geometry.primitives import DetectionObject, QuantityItem
from src.quantity.opening_quantities import calculate_opening_quantities
from src.quantity.room_quantities import calculate_room_quantities
from src.quantity.wall_quantities import calculate_wall_quantities


class QuantityEngine:
    def calculate(
        self,
        objects: list[DetectionObject],
        graph: dict | None,
        scale_ratio: float | None,
    ) -> list[QuantityItem]:
        del graph
        return [
            *calculate_room_quantities(objects, scale_ratio),
            *calculate_wall_quantities(objects, scale_ratio),
            *calculate_opening_quantities(objects),
        ]


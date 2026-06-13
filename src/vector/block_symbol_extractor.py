from src.geometry.primitives import DetectionObject, VectorPrimitive
from src.utils.ids import new_id
from src.vector.layer_classifier import LayerClassifier
from src.vector.layer_classifier import _normalize


class BlockSymbolExtractor:
    def __init__(self):
        self.layer_classifier = LayerClassifier()

    def extract(self, primitives: list[VectorPrimitive]) -> list[DetectionObject]:
        objects: list[DetectionObject] = []
        for primitive in primitives:
            if primitive.type != "block" or not primitive.points:
                continue
            name = str((primitive.raw or {}).get("name") or primitive.label or "")
            symbol_type = _classify_block_name(name) or self.layer_classifier.classify(primitive.layer)
            if symbol_type not in {"door", "window"}:
                continue
            x, y = primitive.points[0]
            width = float((primitive.raw or {}).get("width") or 80.0)
            height = float((primitive.raw or {}).get("height") or 40.0)
            objects.append(
                DetectionObject(
                    id=new_id(f"{symbol_type}_vec"),
                    type=symbol_type,
                    source="dxf_vector",
                    geometry_type="bbox",
                    geometry={"x1": x - width / 2.0, "y1": y - height / 2.0, "x2": x + width / 2.0, "y2": y + height / 2.0},
                    label=name or primitive.layer,
                    confidence=0.85,
                    metadata={"primitive_id": primitive.id, "source_evidence": [primitive.id]},
                )
            )
        return objects


def _classify_block_name(name: str) -> str | None:
    normalized = _normalize(name)
    if "door" in normalized or "kapi" in normalized:
        return "door"
    if "window" in normalized or "pencere" in normalized:
        return "window"
    return None

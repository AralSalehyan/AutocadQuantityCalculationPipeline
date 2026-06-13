from src.geometry.geometry_utils import polyline_length
from src.geometry.primitives import DetectionObject, VectorPrimitive
from src.utils.ids import new_id
from src.vector.layer_classifier import LayerClassifier


class WallCandidateExtractor:
    def __init__(self, min_length: float = 60.0):
        self.min_length = min_length
        self.layer_classifier = LayerClassifier()

    def extract(self, primitives: list[VectorPrimitive]) -> list[DetectionObject]:
        candidates: list[DetectionObject] = []
        for primitive in primitives:
            if primitive.type not in {"line", "polyline", "polygon"}:
                continue
            points = primitive.points
            if primitive.type == "polygon" and points and points[0] != points[-1]:
                points = [*points, points[0]]
            if len(points) < 2:
                continue
            layer_class = self.layer_classifier.classify(primitive.layer)
            strong_layer = layer_class == "wall"
            geometric_hint = polyline_length(points) >= self.min_length and _mostly_orthogonal(points)
            if not strong_layer and not geometric_hint:
                continue
            candidates.append(
                DetectionObject(
                    id=new_id("wall_vec"),
                    type="wall",
                    source="dxf_vector",
                    geometry_type="polyline",
                    geometry={"points": points},
                    label=primitive.layer,
                    confidence=0.85 if strong_layer else 0.60,
                    metadata={
                        "primitive_id": primitive.id,
                        "layer_class": layer_class,
                        "source_evidence": [primitive.id],
                    },
                )
            )
        return candidates


def _mostly_orthogonal(points: list[list[float]]) -> bool:
    checked = 0
    orthogonal = 0
    for start, end in zip(points, points[1:]):
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dx < 1e-6 and dy < 1e-6:
            continue
        checked += 1
        if dx < 1e-6 or dy < 1e-6:
            orthogonal += 1
    return checked > 0 and orthogonal / checked >= 0.75

from src.geometry.geometry_utils import bbox_center, bbox_iou, point_in_polygon
from src.geometry.primitives import DetectionObject


class VectorRasterFusion:
    def __init__(self, opening_iou_threshold: float = 0.25):
        self.opening_iou_threshold = opening_iou_threshold

    def fuse(
        self,
        raster_detections: list[DetectionObject],
        segmentation_objects: list[DetectionObject],
        vector_objects: list[DetectionObject],
        text_objects: list[DetectionObject],
    ) -> list[DetectionObject]:
        fused: list[DetectionObject] = []

        rooms = [_clone_with_evidence(item) for item in segmentation_objects if item.type == "room"]
        segmentation_walls = [_clone_with_evidence(item) for item in segmentation_objects if item.type == "wall"]
        vector_walls = [_clone_with_evidence(item) for item in vector_objects if item.type == "wall"]
        vector_openings = [_clone_with_evidence(item) for item in vector_objects if item.type in {"door", "window"}]
        other_segmented = [_clone_with_evidence(item) for item in segmentation_objects if item.type not in {"room", "wall"}]
        other_vectors = [_clone_with_evidence(item) for item in vector_objects if item.type not in {"wall", "door", "window"}]

        fused.extend(rooms)
        fused.extend(vector_walls if vector_walls else segmentation_walls)
        fused.extend(other_segmented)
        fused.extend(other_vectors)

        openings = vector_openings
        for detection in raster_detections:
            if detection.type not in {"door", "window"}:
                fused.append(_clone_with_evidence(detection))
                continue
            duplicate = _find_duplicate(detection, openings, self.opening_iou_threshold)
            if duplicate is None:
                openings.append(_clone_with_evidence(detection))
                continue
            merged = _merge_duplicate(duplicate, detection)
            openings[openings.index(duplicate)] = merged

        fused.extend(openings)
        fused.extend(_attach_text_to_rooms([_clone_with_evidence(item) for item in text_objects], rooms))
        return fused


def _find_duplicate(candidate: DetectionObject, existing: list[DetectionObject], threshold: float) -> DetectionObject | None:
    if candidate.type not in {"door", "window"} or candidate.geometry_type != "bbox":
        return None
    for item in existing:
        if item.type == candidate.type and item.geometry_type == "bbox" and bbox_iou(candidate.geometry, item.geometry) >= threshold:
            return item
    return None


def _clone_with_evidence(item: DetectionObject) -> DetectionObject:
    cloned = _copy_detection(item)
    cloned.metadata = dict(cloned.metadata or {})
    cloned.metadata.setdefault("source_evidence", [cloned.id])
    cloned.metadata.setdefault("source_types", [cloned.source])
    return cloned


def _merge_duplicate(existing: DetectionObject, candidate: DetectionObject) -> DetectionObject:
    existing_conf = existing.confidence or 0.0
    candidate_conf = candidate.confidence or 0.0
    base = existing if existing_conf >= candidate_conf else candidate
    merged = _copy_detection(base)
    evidence = []
    source_types = []
    for item in (existing, candidate):
        metadata = item.metadata or {}
        evidence.extend(metadata.get("source_evidence") or [item.id])
        source_types.extend(metadata.get("source_types") or [item.source])
    merged.source = "fusion"
    merged.confidence = min(0.98, max(existing_conf, candidate_conf) + 0.10)
    merged.metadata = dict(merged.metadata or {})
    merged.metadata["source_evidence"] = _dedupe(evidence)
    merged.metadata["source_types"] = _dedupe(source_types)
    merged.metadata["fusion_rule"] = "bbox_iou"
    merged.metadata["merged_object_ids"] = _dedupe([existing.id, candidate.id])
    return merged


def _attach_text_to_rooms(texts: list[DetectionObject], rooms: list[DetectionObject]) -> list[DetectionObject]:
    for text in texts:
        if text.geometry_type != "bbox":
            continue
        center = bbox_center(text.geometry)
        for room in rooms:
            if room.geometry_type != "polygon":
                continue
            if point_in_polygon(center, room.geometry.get("points", [])):
                text.metadata = dict(text.metadata or {})
                text.metadata["attached_room_id"] = room.id
                room.metadata = dict(room.metadata or {})
                room.metadata.setdefault("attached_text_ids", []).append(text.id)
                if text.label and (not room.label or room.label.lower().startswith("room")):
                    room.label = text.label
                break
    return texts


def _copy_detection(item: DetectionObject) -> DetectionObject:
    if hasattr(item, "model_copy"):
        return item.model_copy(deep=True)
    return item.copy(deep=True)


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result

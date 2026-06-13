from src.detection.nms import nms
from src.geometry.primitives import DetectionObject, TileInfo


class TileMerger:
    def restore_detection_coordinates(
        self,
        detections: list[DetectionObject],
        tile: TileInfo,
    ) -> list[DetectionObject]:
        restored = []
        for detection in detections:
            if hasattr(detection, "model_copy"):
                data = detection.model_copy(deep=True)
            else:
                data = detection.copy(deep=True)
            if data.geometry_type == "bbox":
                data.geometry["x1"] += tile.x_offset
                data.geometry["x2"] += tile.x_offset
                data.geometry["y1"] += tile.y_offset
                data.geometry["y2"] += tile.y_offset
            elif data.geometry_type in {"polygon", "polyline"}:
                data.geometry["points"] = [
                    [point[0] + tile.x_offset, point[1] + tile.y_offset]
                    for point in data.geometry.get("points", [])
                ]
            elif data.geometry_type == "point":
                data.geometry["x"] += tile.x_offset
                data.geometry["y"] += tile.y_offset
            restored.append(data)
        return restored

    def merge_detections(
        self,
        detections: list[DetectionObject],
        iou_threshold: float = 0.5,
    ) -> list[DetectionObject]:
        return nms(detections, iou_threshold)

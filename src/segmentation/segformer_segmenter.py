from pathlib import Path

import numpy as np
from PIL import Image

from src.detection.model_loading import ModelUnavailableError
from src.geometry.primitives import DetectionObject
from src.segmentation.base_segmenter import BaseSegmenter
from src.segmentation.room_polygonizer import room_mask_to_polygons
from src.segmentation.wall_skeletonizer import wall_mask_to_centerlines
from src.utils.runtime_paths import configure_workspace_runtime_dirs


class SegFormerSegmenter(BaseSegmenter):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._processor = None
        self._model = None

    def predict(self, image_path: Path) -> list[DetectionObject]:
        processor, model = self._load_model()
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError("torch is not installed; cannot run SegFormer segmenter.") from exc

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
        inputs = processor(images=rgb, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(upsampled, dim=1)
            labels = probabilities.argmax(dim=1)[0].detach().cpu().numpy()
            confidence_map = probabilities.max(dim=1)[0][0].detach().cpu().numpy()

        return _labels_to_objects(labels, confidence_map, "segformer", self.confidence_threshold)

    def _load_model(self):
        if not self.model_path.exists():
            raise ModelUnavailableError(f"SegFormer model path does not exist: {self.model_path}")
        if self._model is None or self._processor is None:
            configure_workspace_runtime_dirs()
            try:
                from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            except ImportError as exc:
                raise ModelUnavailableError("transformers is not installed; cannot run SegFormer segmenter.") from exc
            self._processor = AutoImageProcessor.from_pretrained(self.model_path, use_fast=False)
            self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_path)
            self._model.eval()
        return self._processor, self._model


def _labels_to_objects(
    labels: np.ndarray,
    confidence_map: np.ndarray,
    source: str,
    confidence_threshold: float,
) -> list[DetectionObject]:
    objects: list[DetectionObject] = []
    class_map = {1: "room", 2: "wall"}
    for class_id, object_type in class_map.items():
        mask = labels == class_id
        if not mask.any():
            continue
        confidence = float(confidence_map[mask].mean())
        if confidence < confidence_threshold:
            continue
        if object_type == "room":
            objects.extend(room_mask_to_polygons(mask.astype("uint8"), source=source, confidence=round(confidence, 4)))
        elif object_type == "wall":
            objects.extend(wall_mask_to_centerlines(mask.astype("uint8"), source=source, confidence=round(confidence, 4)))
    return objects

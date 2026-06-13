from pathlib import Path

from src.detection.model_loading import ModelUnavailableError
from src.geometry.primitives import DetectionObject
from src.segmentation.base_segmenter import BaseSegmenter
from src.segmentation.segformer_segmenter import SegFormerSegmenter


class Mask2FormerSegmenter(BaseSegmenter):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def predict(self, image_path: Path) -> list[DetectionObject]:
        if not self.model_path.exists():
            raise ModelUnavailableError(f"Mask2Former model path does not exist: {self.model_path}")
        # The conversion from semantic masks to room/wall geometry is shared with SegFormer.
        # A local Mask2Former model that exports semantic logits can be loaded by this adapter
        # once a trained room/wall checkpoint is available.
        raise ModelUnavailableError("Mask2Former inference adapter is a placeholder until a local room/wall checkpoint format is selected.")


class SemanticMask2FormerSegmenter(SegFormerSegmenter):
    pass


from pathlib import Path

from src.segmentation.base_segmenter import BaseSegmenter
from src.segmentation.mask2former_segmenter import Mask2FormerSegmenter
from src.segmentation.mock_segmenter import MockSegmenter
from src.segmentation.segformer_segmenter import SegFormerSegmenter
from src.utils.config import deep_get


def create_segmenter(name: str, config: dict) -> BaseSegmenter:
    normalized = name.lower().strip()
    if normalized == "mock":
        return MockSegmenter()
    if normalized == "segformer":
        return SegFormerSegmenter(
            model_path=Path(deep_get(config, "segformer.model_path", "models/segformer/room_wall")),
            confidence_threshold=float(deep_get(config, "segformer.confidence_threshold", 0.25)),
        )
    if normalized == "mask2former":
        return Mask2FormerSegmenter(
            model_path=Path(deep_get(config, "mask2former.model_path", "models/mask2former/room_wall")),
            confidence_threshold=float(deep_get(config, "mask2former.confidence_threshold", 0.25)),
        )
    raise ValueError(f"Unknown segmenter: {name}")


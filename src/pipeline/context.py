from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineContext:
    input_path: Path
    output_dir: Path
    file_type: str = "unknown"

    rendered_image_path: Path | None = None
    preprocessed_image_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None

    scale_ratio: float | None = None
    scale_unit: str = "m"

    tiles: list[Any] = field(default_factory=list)
    vector_primitives: list[Any] = field(default_factory=list)
    vector_objects: list[Any] = field(default_factory=list)
    text_blocks: list[Any] = field(default_factory=list)

    raster_detections: list[Any] = field(default_factory=list)
    segmentation_objects: list[Any] = field(default_factory=list)
    merged_objects: list[Any] = field(default_factory=list)
    drawing_graph: dict | None = None
    quantities: list[Any] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    detector_names: list[str] = field(default_factory=list)
    segmenter_name: str = "mock"
    debug: bool = False

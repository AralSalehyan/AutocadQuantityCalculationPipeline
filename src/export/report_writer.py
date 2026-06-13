from pathlib import Path

from src.pipeline.context import PipelineContext


class ProcessingReportWriter:
    def write(self, context: PipelineContext, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        quantity_lines = [
            f"- {item.category}: {item.name} = {item.quantity} {item.unit}"
            for item in context.quantities
        ] or ["- None"]
        warning_lines = [f"- {item}" for item in context.warnings] or ["- None"]
        error_lines = [f"- {item}" for item in context.errors] or ["- None"]
        metric_lines = [f"- {key}: `{value}`" for key, value in sorted(context.metrics.items())] or ["- None"]
        content = "\n".join(
            [
                "# Processing Report",
                "",
                f"- Input file: `{context.input_path}`",
                f"- File type: `{context.file_type}`",
                f"- Image size: `{context.image_width} x {context.image_height}`",
                f"- Scale ratio: `{context.scale_ratio}`",
                f"- Tiles: `{len(context.tiles)}`",
                f"- Vector primitives: `{len(context.vector_primitives)}`",
                f"- Vector objects: `{len(context.vector_objects)}`",
                f"- OCR text blocks: `{len(context.text_blocks)}`",
                f"- Raster detections: `{len(context.raster_detections)}`",
                f"- Segmentation objects: `{len(context.segmentation_objects)}`",
                f"- Merged objects: `{len(context.merged_objects)}`",
                "",
                "## Quantity Summary",
                *quantity_lines,
                "",
                "## Warnings",
                *warning_lines,
                "",
                "## Errors",
                *error_lines,
                "",
                "## Metrics",
                *metric_lines,
                "",
                "## Model Configuration",
                f"- Detectors: `{', '.join(context.detector_names)}`",
                f"- Segmenter: `{context.segmenter_name}`",
            ]
        )
        output_path.write_text(content + "\n", encoding="utf-8")
        return output_path

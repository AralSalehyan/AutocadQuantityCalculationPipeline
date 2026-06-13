import logging
from pathlib import Path

from PIL import Image

from src.detection.detector_factory import create_detector
from src.detection.model_loading import ModelUnavailableError
from src.detection.mock_detector import MockDetector
from src.export.debug_visualizer import DebugVisualizer
from src.export.excel_exporter import ExcelExporter
from src.export.json_exporter import JSONExporter
from src.export.report_writer import ProcessingReportWriter
from src.geometry.graph_builder import DrawingGraphBuilder
from src.io.file_loader import detect_file_type
from src.io.image_loader import ImageLoader
from src.io.pdf_loader import PDFLoader
from src.pipeline.context import PipelineContext
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.tile_merger import TileMerger
from src.preprocessing.tiler import ImageTiler
from src.quantity.quantity_engine import QuantityEngine
from src.segmentation.mock_segmenter import MockSegmenter
from src.segmentation.segmenter_factory import create_segmenter
from src.utils.config import deep_get
from src.vector.vector_raster_fusion import VectorRasterFusion
from src.vector.block_symbol_extractor import BlockSymbolExtractor
from src.vector.dxf_parser import DXFParser
from src.vector.wall_candidate_extractor import WallCandidateExtractor


class CADQuantityPipeline:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def run(self, context: PipelineContext) -> PipelineContext:
        stages = [
            self.detect_file_type,
            self.render_or_load_input,
            self.preprocess_image,
            self.create_tiles,
            self.extract_vectors,
            self.run_symbol_detectors,
            self.merge_tiled_detections,
            self.run_room_wall_segmentation,
            self.run_ocr,
            self.fuse_raster_vector_results,
            self.build_drawing_graph,
            self.calculate_quantities,
            self.export_outputs,
            self.write_processing_report,
        ]
        for stage in stages:
            if context.errors:
                break
            self.logger.info("Running stage: %s", stage.__name__)
            try:
                context = stage(context)
            except Exception as exc:
                context.errors.append(f"{stage.__name__} failed: {exc}")
                self.logger.exception("Stage failed: %s", stage.__name__)
                break
        return context

    def detect_file_type(self, context: PipelineContext) -> PipelineContext:
        context.file_type = detect_file_type(context.input_path)
        if context.file_type == "unsupported":
            context.errors.append(f"Unsupported input file type: {context.input_path.suffix}")
        return context

    def render_or_load_input(self, context: PipelineContext) -> PipelineContext:
        output_path = context.output_dir / "rendered.png"
        if context.file_type == "image":
            context.rendered_image_path = ImageLoader().load(context.input_path, output_path)
        elif context.file_type == "pdf":
            dpi = int(deep_get(context.config, "rendering.pdf_dpi", 300))
            context.rendered_image_path = PDFLoader().render_first_page(context.input_path, output_path, dpi=dpi)
        elif context.file_type == "dxf":
            context.rendered_image_path = DXFParser().render_preview(context.input_path, output_path)
        else:
            context.errors.append(f"{context.file_type.upper()} input is not supported.")
            return context
        with Image.open(context.rendered_image_path) as image:
            context.image_width, context.image_height = image.size
        return context

    def preprocess_image(self, context: PipelineContext) -> PipelineContext:
        assert context.rendered_image_path is not None
        output_path = context.output_dir / "preprocessed.png"
        context.preprocessed_image_path = ImagePreprocessor().preprocess(context.rendered_image_path, output_path)
        return context

    def create_tiles(self, context: PipelineContext) -> PipelineContext:
        assert context.preprocessed_image_path is not None
        tile_size = int(deep_get(context.config, "tiling.tile_size", 1280))
        overlap = float(deep_get(context.config, "tiling.overlap", 0.25))
        context.tiles = ImageTiler().create_tiles(context.preprocessed_image_path, context.output_dir / "tiles", tile_size, overlap)
        return context

    def extract_vectors(self, context: PipelineContext) -> PipelineContext:
        if context.file_type == "pdf":
            loader = PDFLoader()
            try:
                context.vector_primitives = loader.extract_vector_primitives(context.input_path)
            except Exception as exc:
                context.vector_primitives = []
                context.warnings.append(f"PDF vector extraction failed: {exc}")
            try:
                context.text_blocks = loader.extract_text_blocks(context.input_path)
            except Exception as exc:
                context.text_blocks = []
                context.warnings.append(f"PDF text extraction failed: {exc}")
            return context
        if context.file_type == "dxf":
            parser = DXFParser()
            try:
                context.vector_primitives = parser.parse(context.input_path)
                wall_objects = WallCandidateExtractor().extract(context.vector_primitives)
                block_objects = BlockSymbolExtractor().extract(context.vector_primitives)
                context.vector_objects = [*wall_objects, *block_objects]
                context.text_blocks = _dxf_text_objects(context.vector_primitives)
                context.metrics["dxf_vector_objects"] = len(context.vector_objects)
            except Exception as exc:
                context.vector_primitives = []
                context.vector_objects = []
                context.warnings.append(f"DXF vector extraction failed: {exc}")
            return context
        context.vector_primitives = []
        return context

    def run_symbol_detectors(self, context: PipelineContext) -> PipelineContext:
        assert context.preprocessed_image_path is not None
        detections = []
        fallback_to_mock = bool(deep_get(context.config, "detectors.fallback_to_mock", True))
        merger = TileMerger()
        for name in context.detector_names or ["mock"]:
            normalized = name.lower().strip()
            try:
                if normalized == "mock":
                    detections.extend(MockDetector().predict(context.preprocessed_image_path))
                    continue

                detector = create_detector(normalized, context.config)
                tile_detections = []
                for tile in context.tiles:
                    local_detections = detector.predict(Path(tile.image_path))
                    for detection in local_detections:
                        detection.metadata = detection.metadata or {}
                        detection.metadata["tile_id"] = tile.id
                    tile_detections.extend(merger.restore_detection_coordinates(local_detections, tile))
                detections.extend(tile_detections)
                context.metrics[f"{normalized}_tile_detections"] = len(tile_detections)
            except ModelUnavailableError as exc:
                warning = f"Detector '{normalized}' unavailable: {exc}"
                if fallback_to_mock:
                    warning += "; using mock detector fallback."
                    detections.extend(MockDetector().predict(context.preprocessed_image_path))
                else:
                    context.errors.append(warning)
                context.warnings.append(warning)
            except ValueError as exc:
                context.errors.append(str(exc))
                break
        context.raster_detections = detections
        return context

    def merge_tiled_detections(self, context: PipelineContext) -> PipelineContext:
        threshold = float(deep_get(context.config, "fusion.bbox_iou_threshold", 0.5))
        context.raster_detections = TileMerger().merge_detections(context.raster_detections, threshold)
        return context

    def run_room_wall_segmentation(self, context: PipelineContext) -> PipelineContext:
        assert context.preprocessed_image_path is not None
        normalized = context.segmenter_name.lower().strip()
        fallback_to_mock = bool(deep_get(context.config, "segmenter.fallback_to_mock", True))
        try:
            segmenter = create_segmenter(normalized, context.config)
            context.segmentation_objects = segmenter.predict(context.preprocessed_image_path)
            context.metrics[f"{normalized}_segmentation_objects"] = len(context.segmentation_objects)
        except ModelUnavailableError as exc:
            warning = f"Segmenter '{normalized}' unavailable: {exc}"
            if fallback_to_mock:
                warning += "; using mock segmenter fallback."
                context.segmentation_objects = MockSegmenter().predict(context.preprocessed_image_path)
            else:
                context.errors.append(warning)
            context.warnings.append(warning)
        except ValueError as exc:
            context.errors.append(str(exc))
        return context

    def run_ocr(self, context: PipelineContext) -> PipelineContext:
        context.text_blocks = context.text_blocks or []
        return context

    def fuse_raster_vector_results(self, context: PipelineContext) -> PipelineContext:
        context.merged_objects = VectorRasterFusion().fuse(
            context.raster_detections,
            context.segmentation_objects,
            context.vector_objects,
            context.text_blocks,
        )
        return context

    def build_drawing_graph(self, context: PipelineContext) -> PipelineContext:
        context.drawing_graph = DrawingGraphBuilder().build(context.merged_objects)
        return context

    def calculate_quantities(self, context: PipelineContext) -> PipelineContext:
        if context.scale_ratio is None:
            context.warnings.append("Scale ratio missing; quantities remain in pixels or pixel squared.")
        context.quantities = QuantityEngine().calculate(context.merged_objects, context.drawing_graph, context.scale_ratio)
        return context

    def export_outputs(self, context: PipelineContext) -> PipelineContext:
        assert context.preprocessed_image_path is not None
        json_exporter = JSONExporter()
        json_exporter.write(context.output_dir / "vector_primitives.json", context.vector_primitives)
        json_exporter.write(context.output_dir / "raster_detections.json", context.raster_detections)
        json_exporter.write(context.output_dir / "segmentation_objects.json", context.segmentation_objects)
        json_exporter.write(context.output_dir / "merged_objects.json", context.merged_objects)
        json_exporter.write(context.output_dir / "drawing_graph.json", context.drawing_graph or {})
        json_exporter.write(context.output_dir / "quantities.json", context.quantities)
        ExcelExporter().write(
            context.output_dir / "quantities.xlsx",
            context.quantities,
            context.raster_detections,
            context.merged_objects,
            context.warnings,
        )
        visualizer = DebugVisualizer()
        visualizer.draw_overlay(context.preprocessed_image_path, context.output_dir / "debug_overlay.png", context.merged_objects)
        visualizer.draw_tiles(context.preprocessed_image_path, context.output_dir / "debug_overlay_tiles.png", context.tiles)
        return context

    def write_processing_report(self, context: PipelineContext) -> PipelineContext:
        ProcessingReportWriter().write(context, context.output_dir / "processing_report.md")
        return context


def _dxf_text_objects(primitives):
    from src.geometry.primitives import DetectionObject
    from src.utils.ids import new_id

    objects = []
    for primitive in primitives:
        if primitive.type != "text" or not primitive.points:
            continue
        x, y = primitive.points[0]
        label = primitive.label or ""
        width = max(40.0, len(label) * 7.0)
        objects.append(
            DetectionObject(
                id=new_id("text_vec"),
                type=_classify_text(label),
                source="dxf_vector",
                geometry_type="bbox",
                geometry={"x1": x, "y1": y - 12.0, "x2": x + width, "y2": y + 6.0},
                label=label,
                confidence=0.85,
                metadata={"primitive_id": primitive.id},
            )
        )
    return objects


def _classify_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("m2", "m^2", "oda", "salon", "mutfak", "banyo", "wc", "hol", "room")):
        return "room_text"
    if any(token in lowered for token in ("cm", "mm", " m")):
        return "dimension_text"
    return "room_text"

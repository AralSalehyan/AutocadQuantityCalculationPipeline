# plan.md — Performance-First CAD Quantity Calculation Pipeline

## 0. Project Goal

Build a performance-first technical pipeline for autonomous architectural CAD/PDF quantity calculation.

The first phase is **not** a polished MVP web app. The first phase is a strong technical pipeline that can process architectural drawings, detect relevant objects, extract geometry, calculate quantities, and produce measurable/debuggable outputs.

Target use case:

```text
Architectural PDF/DXF/image input
↓
High-resolution raster rendering + CAD/vector extraction
↓
Multi-model detection and segmentation
↓
Raster/vector fusion
↓
Drawing graph construction
↓
Rule-based quantity calculation
↓
JSON + Excel + debug overlay + evaluation report
```

Initial product scope combines:

- Option A: Room/area calculation
- Option C: Wall/opening quantity calculation

The prototype should calculate:

- Room areas
- Room perimeters
- Gross wall lengths
- Door counts
- Window counts
- Optional later: room labels, wall types, door/window types, net wall quantities

---

## 1. Core Design Principle

This system must not rely on one VLM/LLM or one YOLO-style detector.

Use a **multi-branch architecture**:

```text
Input drawing
├── Raster branch
│   ├── high-resolution rendering
│   ├── tiled inference
│   ├── transformer/object detector for doors/windows/symbols
│   ├── segmentation model for rooms/walls
│   └── OCR for text/dimensions
│
├── Vector/CAD branch
│   ├── PDF vector extraction
│   ├── DXF primitive extraction
│   ├── layer/block/text parsing
│   ├── wall candidate extraction
│   └── primitive graph construction
│
├── Fusion layer
│   ├── merge raster detections with vector primitives
│   ├── deduplicate detections
│   ├── attach doors/windows to walls
│   ├── attach text labels to rooms
│   └── compute confidence
│
└── Quantity engine
    ├── room area
    ├── room perimeter
    ├── wall length
    ├── door count
    ├── window count
    └── Excel/JSON/debug outputs
```

The detection model identifies objects. The segmentation model extracts shapes. The vector branch extracts exact geometry when available. The quantity engine must be deterministic and auditable.

---

## 2. Important MVP Philosophy

Although we are building the pipeline first, it should be designed as the core of a future MVP.

Do **not** implement in phase 1:

- User authentication
- React frontend
- Database-backed project management
- Cloud deployment
- Full BOQ/cost estimation
- MEP/electrical/plumbing modules
- LLM-based BOQ mapping

Do implement in phase 1:

- Command-line pipeline
- High-resolution rendering
- Tiled inference
- Pluggable detection models
- Pluggable segmentation models
- DXF/PDF vector extraction
- Raster/vector fusion
- Quantity calculation
- Debug visualizations
- JSON/Excel exports
- Evaluation scripts

---

## 3. First Prototype Inputs and Outputs

### 3.1 Supported inputs

Support:

```text
.pdf
.dxf
.png
.jpg
.jpeg
```

For `.dwg`, do not implement direct parsing in phase 1. Return a clear error:

```text
DWG is not supported directly in this prototype. Please convert DWG to DXF first.
```

### 3.2 Expected command

```bash
python run_pipeline.py \
  --input sample_data/pdf/example.pdf \
  --output outputs/example_pdf \
  --detectors rtdetr,yolo \
  --segmenter segformer \
  --scale 0.01 \
  --debug
```

Also support:

```bash
python run_pipeline.py --input sample_data/images/example.png --output outputs/example_img --detectors mock --segmenter mock --scale 0.01 --debug

python run_pipeline.py --input sample_data/dxf/example.dxf --output outputs/example_dxf --detectors mock --segmenter mock --scale 0.01 --debug
```

### 3.3 Expected output folder

Each run should create:

```text
outputs/project_name/
├── rendered.png
├── preprocessed.png
├── tiles/
│   ├── tile_0000.png
│   ├── tile_0001.png
│   └── ...
├── vector_primitives.json
├── raster_detections.json
├── segmentation_objects.json
├── merged_objects.json
├── drawing_graph.json
├── quantities.json
├── quantities.xlsx
├── debug_overlay.png
├── debug_overlay_tiles.png
├── evaluation_report.json optional
└── processing_report.md
```

---

## 4. Recommended Repository Structure

Create this structure:

```text
cad-quantity-pipeline/
├── plan.md
├── README.md
├── requirements.txt
├── .env.example
├── run_pipeline.py
├── evaluate_outputs.py
├── configs/
│   ├── default.yaml
│   ├── detectors.yaml
│   ├── segmenters.yaml
│   ├── tiling.yaml
│   └── quantity_rules.yaml
├── sample_data/
│   ├── README.md
│   ├── pdf/
│   ├── dxf/
│   ├── images/
│   └── ground_truth/
├── datasets/
│   ├── README.md
│   ├── door_window/
│   └── room_wall/
├── models/
│   ├── README.md
│   ├── yolo/
│   ├── rtdetr/
│   ├── segformer/
│   ├── mask2former/
│   └── ocr/
├── outputs/
│   └── .gitkeep
├── notebooks/
│   ├── inspect_dataset.ipynb
│   └── compare_models.ipynb
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── context.py
│   │   ├── pipeline.py
│   │   └── stages.py
│   ├── io/
│   │   ├── file_loader.py
│   │   ├── pdf_loader.py
│   │   ├── dxf_loader.py
│   │   └── image_loader.py
│   ├── preprocessing/
│   │   ├── image_preprocessor.py
│   │   ├── tiler.py
│   │   ├── tile_merger.py
│   │   └── scale_detection.py
│   ├── detection/
│   │   ├── base_detector.py
│   │   ├── mock_detector.py
│   │   ├── yolo_detector.py
│   │   ├── rtdetr_detector.py
│   │   ├── ensemble_detector.py
│   │   ├── nms.py
│   │   ├── weighted_boxes_fusion.py
│   │   └── detector_evaluator.py
│   ├── segmentation/
│   │   ├── base_segmenter.py
│   │   ├── mock_segmenter.py
│   │   ├── segformer_segmenter.py
│   │   ├── mask2former_segmenter.py
│   │   ├── mask_to_geometry.py
│   │   ├── wall_skeletonizer.py
│   │   ├── room_polygonizer.py
│   │   └── segmentation_evaluator.py
│   ├── ocr/
│   │   ├── base_ocr.py
│   │   ├── paddle_ocr.py
│   │   ├── tesseract_ocr.py
│   │   └── text_classifier.py
│   ├── vector/
│   │   ├── pdf_vector_extractor.py
│   │   ├── dxf_parser.py
│   │   ├── layer_classifier.py
│   │   ├── block_symbol_extractor.py
│   │   ├── wall_candidate_extractor.py
│   │   ├── primitive_graph_builder.py
│   │   └── vector_raster_fusion.py
│   ├── geometry/
│   │   ├── primitives.py
│   │   ├── geometry_utils.py
│   │   ├── graph_builder.py
│   │   └── coordinate_transform.py
│   ├── quantity/
│   │   ├── quantity_engine.py
│   │   ├── room_quantities.py
│   │   ├── wall_quantities.py
│   │   ├── opening_quantities.py
│   │   └── validation_rules.py
│   ├── export/
│   │   ├── json_exporter.py
│   │   ├── excel_exporter.py
│   │   ├── debug_visualizer.py
│   │   └── report_writer.py
│   ├── training/
│   │   ├── prepare_detection_dataset.py
│   │   ├── prepare_segmentation_dataset.py
│   │   ├── train_yolo.py
│   │   ├── train_rtdetr.py
│   │   ├── train_segformer.py
│   │   ├── train_mask2former.py
│   │   └── dataset_notes.md
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── ids.py
└── tests/
    ├── test_geometry_utils.py
    ├── test_tiler.py
    ├── test_tile_merger.py
    ├── test_quantity_engine.py
    ├── test_pdf_loader.py
    ├── test_dxf_parser.py
    ├── test_detection_nms.py
    ├── test_fusion.py
    └── test_pipeline_smoke.py
```

---

## 5. Dependencies

Create `requirements.txt` with practical first dependencies:

```text
numpy
opencv-python
pillow
pydantic
pyyaml
rich
tqdm
pandas
openpyxl
shapely
networkx
matplotlib
PyMuPDF
ezdxf
pytest
pytest-cov
ultralytics
scikit-image
```

Optional but useful later:

```text
transformers
torch
torchvision
accelerate
paddleocr
paddlepaddle
onnxruntime
```

Keep heavyweight dependencies optional if installation becomes difficult.

---

## 6. Data Schemas

Create shared schemas in:

```text
src/geometry/primitives.py
```

Use Pydantic models.

### 6.1 DetectionObject

```python
class DetectionObject(BaseModel):
    id: str
    type: str
    source: str
    geometry_type: str
    geometry: dict
    label: str | None = None
    confidence: float | None = None
    metadata: dict | None = None
```

Supported `type` values:

```text
room
wall
door
window
room_text
dimension_text
block_symbol
unknown_symbol
```

Supported `source` values:

```text
mock
yolo
rtdetr
segformer
mask2former
ocr
pdf_vector
dxf_vector
fusion
user_later
```

Supported `geometry_type` values:

```text
bbox
polygon
polyline
point
mask_rle
```

### 6.2 QuantityItem

```python
class QuantityItem(BaseModel):
    id: str
    category: str
    name: str
    quantity: float
    unit: str
    source_object_ids: list[str]
    confidence: float | None = None
    calculation: dict | None = None
```

Supported `category` values:

```text
room_area
room_perimeter
wall_length
wall_length_total
door_count
window_count
```

### 6.3 VectorPrimitive

```python
class VectorPrimitive(BaseModel):
    id: str
    type: str
    points: list[list[float]]
    layer: str | None = None
    label: str | None = None
    raw: dict | None = None
```

Supported `type` values:

```text
line
polyline
polygon
block
text
arc
circle
dimension
```

### 6.4 TileInfo

```python
class TileInfo(BaseModel):
    id: str
    image_path: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    scale: float = 1.0
```

---

## 7. Pipeline Context

Create:

```text
src/pipeline/context.py
```

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class PipelineContext:
    input_path: Path
    output_dir: Path
    file_type: str

    rendered_image_path: Path | None = None
    preprocessed_image_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None

    scale_ratio: float | None = None
    scale_unit: str = "m"

    tiles: list[Any] = field(default_factory=list)
    vector_primitives: list[Any] = field(default_factory=list)
    text_blocks: list[Any] = field(default_factory=list)

    raster_detections: list[Any] = field(default_factory=list)
    segmentation_objects: list[Any] = field(default_factory=list)
    merged_objects: list[Any] = field(default_factory=list)
    drawing_graph: dict | None = None
    quantities: list[Any] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
```

---

## 8. Pipeline Stages

Create orchestrator:

```text
src/pipeline/pipeline.py
```

Implement stages in this order:

```text
1. detect_file_type
2. render_or_load_input
3. preprocess_image
4. create_tiles
5. extract_vectors
6. run_symbol_detectors
7. merge_tiled_detections
8. run_room_wall_segmentation
9. run_ocr
10. fuse_raster_vector_results
11. build_drawing_graph
12. calculate_quantities
13. export_outputs
14. write_processing_report
```

Each stage must:

- Receive `PipelineContext`
- Modify `PipelineContext`
- Return `PipelineContext`
- Log what it did
- Add non-fatal problems to `context.warnings`
- Add fatal problems to `context.errors`

---

## 9. File Loading

### 9.1 File type detection

File:

```text
src/io/file_loader.py
```

Implement:

```python
def detect_file_type(input_path: Path) -> str:
    ...
```

Return:

```text
pdf
dxf
image
dwg
unsupported
```

If DWG:

```text
Raise a user-friendly error explaining that DWG should be converted to DXF for phase 1.
```

---

## 10. Input Rendering and Parsing

### 10.1 PDF loader

File:

```text
src/io/pdf_loader.py
```

Implement:

```python
class PDFLoader:
    def render_first_page(self, input_path: Path, output_path: Path, dpi: int = 300) -> Path:
        pass

    def extract_text_blocks(self, input_path: Path) -> list[DetectionObject]:
        pass

    def extract_vector_primitives(self, input_path: Path) -> list[VectorPrimitive]:
        pass
```

Requirements:

- Render first page to `rendered.png`
- Default DPI should be 300 for better small-symbol detection
- Extract text blocks when possible
- Attempt PDF vector extraction
- If vector extraction fails, continue with raster pipeline

### 10.2 DXF loader/parser

File:

```text
src/vector/dxf_parser.py
```

Implement:

```python
class DXFParser:
    def parse(self, input_path: Path) -> list[VectorPrimitive]:
        pass

    def render_preview(self, input_path: Path, output_path: Path) -> Path:
        pass
```

Extract:

```text
LINE
LWPOLYLINE
POLYLINE
INSERT
TEXT
MTEXT
CIRCLE
ARC
DIMENSION if possible
```

For preview rendering:

- Prefer a simple generated preview from extracted line/polyline primitives.
- Exact CAD rendering is not required for the first version.
- Continue even if preview is imperfect.

### 10.3 Image loader

File:

```text
src/io/image_loader.py
```

Implement:

```python
class ImageLoader:
    def load(self, input_path: Path, output_path: Path) -> Path:
        pass
```

For image input:

- Copy or normalize input image to `rendered.png`
- Save image dimensions to context

---

## 11. Image Preprocessing

File:

```text
src/preprocessing/image_preprocessor.py
```

Implement:

```python
class ImagePreprocessor:
    def preprocess(self, image_path: Path, output_path: Path) -> Path:
        pass
```

Use conservative preprocessing:

```text
convert to RGB
optional grayscale copy internally
light denoising only
contrast normalization if needed
no aggressive thresholding by default
```

Important:

- Architectural drawings often have thin lines.
- Do not destroy line details.
- Save `preprocessed.png`.

---

## 12. Tiled Inference

Architectural drawings are large and symbols are small. Use tiling from phase 1.

### 12.1 Tiler

File:

```text
src/preprocessing/tiler.py
```

Implement:

```python
class ImageTiler:
    def create_tiles(
        self,
        image_path: Path,
        output_dir: Path,
        tile_size: int = 1280,
        overlap: float = 0.25,
    ) -> list[TileInfo]:
        pass
```

Config defaults:

```yaml
tiling:
  enabled: true
  tile_size: 1280
  overlap: 0.25
```

### 12.2 Tile coordinate restoration

File:

```text
src/preprocessing/tile_merger.py
```

Implement:

```python
class TileMerger:
    def restore_detection_coordinates(
        self,
        detections: list[DetectionObject],
        tile: TileInfo,
    ) -> list[DetectionObject]:
        pass

    def merge_detections(
        self,
        detections: list[DetectionObject],
        iou_threshold: float = 0.5,
    ) -> list[DetectionObject]:
        pass
```

Use NMS first. Add Weighted Boxes Fusion later.

---

## 13. Symbol Detection Models

Detection is for countable objects:

```text
door
window
stair later
column later
sink/toilet later
electrical symbols later
```

### 13.1 Base detector

File:

```text
src/detection/base_detector.py
```

```python
class BaseDetector:
    def predict(self, image_path: Path) -> list[DetectionObject]:
        raise NotImplementedError
```

### 13.2 Mock detector

File:

```text
src/detection/mock_detector.py
```

Purpose:

- Keep pipeline end-to-end runnable
- Return sample door/window boxes
- Useful for tests/debugging

Mock output:

```text
2 doors
2 windows
```

### 13.3 YOLO detector

File:

```text
src/detection/yolo_detector.py
```

Implement:

```python
class YOLODetector(BaseDetector):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        pass

    def predict(self, image_path: Path) -> list[DetectionObject]:
        pass
```

Use if model exists. If not, warn clearly.

### 13.4 RT-DETR detector

File:

```text
src/detection/rtdetr_detector.py
```

Implement:

```python
class RTDETRDetector(BaseDetector):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        pass

    def predict(self, image_path: Path) -> list[DetectionObject]:
        pass
```

If implementation uses Ultralytics-compatible RT-DETR, keep API similar to YOLO detector.

### 13.5 Ensemble detector

File:

```text
src/detection/ensemble_detector.py
```

Implement:

```python
class EnsembleDetector(BaseDetector):
    def __init__(self, detectors: list[BaseDetector]):
        pass

    def predict(self, image_path: Path) -> list[DetectionObject]:
        pass
```

Behavior:

- Run multiple detectors
- Combine detections
- Deduplicate with NMS or Weighted Boxes Fusion
- Preserve source model in metadata

---

## 14. Segmentation Models

Segmentation is for shape extraction:

```text
room polygons
wall masks/wall centerlines
```

YOLO boxes are not enough for room area or wall length.

### 14.1 Base segmenter

File:

```text
src/segmentation/base_segmenter.py
```

```python
class BaseSegmenter:
    def predict(self, image_path: Path) -> list[DetectionObject]:
        raise NotImplementedError
```

### 14.2 Mock segmenter

File:

```text
src/segmentation/mock_segmenter.py
```

Purpose:

- Return simple room polygons and wall polylines
- Keep quantity engine testable

Mock output:

```text
2 room polygons
4 wall polylines
```

### 14.3 SegFormer segmenter

File:

```text
src/segmentation/segformer_segmenter.py
```

Implement interface and model loading.

Expected output:

```text
room polygons or masks
wall masks or wall polylines
```

If model is missing:

- Warn clearly
- Allow fallback to mock if configured

### 14.4 Mask2Former segmenter

File:

```text
src/segmentation/mask2former_segmenter.py
```

Implement interface and model loading.

This can initially be a placeholder with clear `NotImplementedError`, but the pipeline architecture must support it.

### 14.5 Mask to geometry

File:

```text
src/segmentation/mask_to_geometry.py
```

Implement:

```python
def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    pass
```

Use OpenCV contours or scikit-image.

### 14.6 Wall skeletonizer

File:

```text
src/segmentation/wall_skeletonizer.py
```

Implement:

```python
def wall_mask_to_centerlines(mask: np.ndarray) -> list[DetectionObject]:
    pass
```

Purpose:

- Convert wall segmentation mask into polylines
- Allow wall length calculation

---

## 15. OCR and Text Classification

OCR is useful for room labels, dimensions, and scale notes.

### 15.1 Base OCR

File:

```text
src/ocr/base_ocr.py
```

```python
class BaseOCR:
    def predict(self, image_path: Path) -> list[DetectionObject]:
        raise NotImplementedError
```

### 15.2 OCR implementation

Add one implementation first:

```text
src/ocr/tesseract_ocr.py
```

or:

```text
src/ocr/paddle_ocr.py
```

OCR output should become `DetectionObject` with type:

```text
room_text
dimension_text
```

### 15.3 Text classifier

File:

```text
src/ocr/text_classifier.py
```

Classify OCR text into:

```text
room_name
dimension
scale_note
unknown
```

Simple rules first:

- Contains `m2`, `m²`, `m^2` → likely room area text
- Contains numbers with `cm`, `mm`, `m` → likely dimension
- Turkish room words like `SALON`, `ODA`, `MUTFAK`, `BANYO`, `WC`, `HOL` → likely room name

---

## 16. Vector/CAD Branch

### 16.1 PDF vector extractor

File:

```text
src/vector/pdf_vector_extractor.py
```

Extract line/path/text primitives from digital PDF when possible.

Output:

```text
vector_primitives.json
```

### 16.2 DXF parser

Already defined in section 10.2.

### 16.3 Layer classifier

File:

```text
src/vector/layer_classifier.py
```

Implement:

```python
class LayerClassifier:
    def classify(self, layer_name: str | None) -> str:
        pass
```

Use rule hints:

```text
wall
walls
duvar
a-wall
a_walls
architectural
kapı
kapi
door
pencere
window
win
text
dim
dimension
ölçü
olcu
```

### 16.4 Wall candidate extractor

File:

```text
src/vector/wall_candidate_extractor.py
```

Convert vector lines/polylines into wall candidates.

Implement:

```python
class WallCandidateExtractor:
    def extract(self, primitives: list[VectorPrimitive]) -> list[DetectionObject]:
        pass
```

Rules:

- If layer classification is `wall`, create wall objects from lines/polylines
- If polyline is long and orthogonal, consider wall candidate
- Store raw primitive id in metadata

### 16.5 Block symbol extractor

File:

```text
src/vector/block_symbol_extractor.py
```

Use block names as strong hints.

Rules:

- Block name includes `door`, `kapi`, `kapı` → door
- Block name includes `window`, `pencere` → window
- Create `DetectionObject` with source `dxf_vector`

### 16.6 Primitive graph builder

File:

```text
src/vector/primitive_graph_builder.py
```

Build graph:

```text
nodes: vector endpoints, symbols, text, room polygons
edges: adjacency/intersections/containment/nearby relationships
```

Use NetworkX.

Output:

```text
drawing_graph.json
```

---

## 17. Raster/Vector Fusion

File:

```text
src/vector/vector_raster_fusion.py
```

Implement:

```python
class VectorRasterFusion:
    def fuse(
        self,
        raster_detections: list[DetectionObject],
        segmentation_objects: list[DetectionObject],
        vector_objects: list[DetectionObject],
        text_objects: list[DetectionObject],
    ) -> list[DetectionObject]:
        pass
```

Fusion rules for phase 1:

1. Keep all segmentation room polygons.
2. Keep vector wall candidates if available.
3. Keep segmentation wall polylines if vector walls are missing.
4. Merge door/window detections from raster and vector branch.
5. If raster bbox and vector block overlap, keep one merged object with higher confidence.
6. Attach nearby text objects to room polygons if text center is inside polygon.
7. Add metadata with source evidence.

Confidence logic:

```text
model-only object: model confidence
vector-only object: 0.85 if layer/block name is strong, else 0.60
model + vector agreement: min(0.98, max_confidence + 0.10)
user-corrected later: 1.00
```

---

## 18. Geometry Utilities

File:

```text
src/geometry/geometry_utils.py
```

Implement:

```python
def polygon_area(points: list[list[float]]) -> float:
    pass

def polygon_perimeter(points: list[list[float]]) -> float:
    pass

def polyline_length(points: list[list[float]]) -> float:
    pass

def bbox_iou(box_a: dict, box_b: dict) -> float:
    pass

def bbox_center(box: dict) -> tuple[float, float]:
    pass

def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    pass

def scale_length(value: float, scale_ratio: float | None) -> float:
    pass

def scale_area(value: float, scale_ratio: float | None) -> float:
    pass
```

Scale behavior:

If `scale_ratio` is missing:

- Quantities remain in pixels or pixel²
- Add warning to report

If `scale_ratio` exists:

```text
length_m = length_px * scale_ratio
area_m2 = area_px2 * scale_ratio * scale_ratio
```

---

## 19. Drawing Graph

File:

```text
src/geometry/graph_builder.py
```

Build graph from fused objects.

Graph relationships:

```text
room contains room_text
room adjacent_to wall
wall has_opening door/window
door/window near wall
text near object
```

Implement simple geometric relationships first:

- Point inside polygon
- Bbox center inside polygon
- Distance from bbox center to polyline
- Polygon boundary near wall polyline

Output graph as JSON.

---

## 20. Quantity Engine

File:

```text
src/quantity/quantity_engine.py
```

Implement:

```python
class QuantityEngine:
    def calculate(
        self,
        objects: list[DetectionObject],
        graph: dict | None,
        scale_ratio: float | None,
    ) -> list[QuantityItem]:
        pass
```

### 20.1 Room quantities

File:

```text
src/quantity/room_quantities.py
```

For each room polygon:

- Calculate area
- Calculate perimeter
- Use attached room text as room name if available
- Otherwise name as `Room 1`, `Room 2`, etc.

Output categories:

```text
room_area
room_perimeter
```

### 20.2 Wall quantities

File:

```text
src/quantity/wall_quantities.py
```

For wall polylines:

- Calculate individual wall length
- Calculate total gross wall length

Output categories:

```text
wall_length
wall_length_total
```

### 20.3 Opening quantities

File:

```text
src/quantity/opening_quantities.py
```

Count:

```text
door_count
window_count
```

For phase 1:

- Count all non-deleted door/window objects
- Do not yet subtract openings from wall lengths

---

## 21. Exports

### 21.1 JSON exporter

File:

```text
src/export/json_exporter.py
```

Write:

```text
vector_primitives.json
raster_detections.json
segmentation_objects.json
merged_objects.json
drawing_graph.json
quantities.json
```

### 21.2 Excel exporter

File:

```text
src/export/excel_exporter.py
```

Create:

```text
quantities.xlsx
```

Workbook sheets:

```text
Room Areas
Openings
Wall Quantities
Raw Detections
Merged Objects
Warnings
```

Room Areas columns:

```text
Object ID
Room Name
Area
Unit
Perimeter
Perimeter Unit
Confidence
Source
```

Openings columns:

```text
Type
Count
Unit
Confidence
Source Object IDs
```

Wall Quantities columns:

```text
Object ID
Name
Length
Unit
Confidence
Source
```

Raw Detections columns:

```text
Object ID
Type
Source
Geometry Type
Label
Confidence
Geometry JSON
```

### 21.3 Debug visualizer

File:

```text
src/export/debug_visualizer.py
```

Create:

```text
debug_overlay.png
debug_overlay_tiles.png
```

Draw:

- Room polygons
- Wall polylines
- Door boxes
- Window boxes
- OCR text boxes
- Object labels
- Confidence scores

Use distinct visual styles but do not hard-code too many dependencies.

### 21.4 Processing report

File:

```text
src/export/report_writer.py
```

Create:

```text
processing_report.md
```

Include:

```text
Input file
File type
Image size
Scale ratio
Tile size and overlap
Number of vector primitives
Number of OCR text blocks
Number of raster detections
Number of segmentation objects
Number of merged objects
Quantity summary
Warnings
Errors
Model configuration
```

---

## 22. Evaluation

Performance-first development requires evaluation from the beginning.

Create:

```text
evaluate_outputs.py
```

Support:

```bash
python evaluate_outputs.py \
  --pred outputs/example/merged_objects.json \
  --gt sample_data/ground_truth/example_objects.json
```

### 22.1 Detection metrics

For door/window detection:

```text
mAP50 if possible later
precision
recall
F1
false positives
false negatives
count error per class
average IoU
```

### 22.2 Quantity metrics

For quantities:

```text
room area absolute error
room area percentage error
wall length absolute error
wall length percentage error
door count error
window count error
```

### 22.3 Report output

Write:

```text
evaluation_report.json
evaluation_report.md
```

---

## 23. Training Strategy

Create scripts but do not make main pipeline depend on training completion.

### 23.1 Detection dataset

File:

```text
src/training/prepare_detection_dataset.py
```

Classes phase 1:

```text
door
window
```

Expected YOLO/RT-DETR style dataset:

```text
datasets/door_window/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml`:

```yaml
path: datasets/door_window
train: images/train
val: images/val
names:
  0: door
  1: window
```

### 23.2 Segmentation dataset

File:

```text
src/training/prepare_segmentation_dataset.py
```

Classes phase 1:

```text
room
wall
```

Support masks/polygons.

### 23.3 Dataset notes

File:

```text
src/training/dataset_notes.md
```

Mention:

```text
CubiCasa5K for room/wall/opening segmentation
CVC-FP for floor-plan structural parsing
FloorPlanCAD for CAD symbol spotting
ArchCAD-400K for large-scale architectural CAD symbol detection
Turkish real project drawings are required for production quality
```

### 23.4 Real Turkish data strategy

Add note:

```text
Open datasets are useful for pretraining and prototype validation, but production quality requires Turkish architectural drawings with local symbols, Turkish room labels, layer names, scales, legends, and drawing conventions.
```

---

## 24. Configuration Files

### 24.1 `configs/default.yaml`

```yaml
rendering:
  pdf_dpi: 300

preprocessing:
  enabled: true

tiling:
  enabled: true
  tile_size: 1280
  overlap: 0.25

detectors:
  active: ["mock"]
  fallback_to_mock: true

segmenter:
  active: "mock"
  fallback_to_mock: true

ocr:
  enabled: false
  engine: "tesseract"

fusion:
  bbox_iou_threshold: 0.5
  strong_vector_confidence: 0.85
  weak_vector_confidence: 0.60

quantity:
  default_scale_ratio: null
  unit: "m"
```

### 24.2 `configs/detectors.yaml`

```yaml
yolo:
  model_path: "models/yolo/door_window.pt"
  confidence_threshold: 0.25

rtdetr:
  model_path: "models/rtdetr/door_window.pt"
  confidence_threshold: 0.25

ensemble:
  enabled: false
  merge_method: "nms"
  iou_threshold: 0.5
```

### 24.3 `configs/segmenters.yaml`

```yaml
segformer:
  model_path: "models/segformer/room_wall"
  confidence_threshold: 0.25

mask2former:
  model_path: "models/mask2former/room_wall"
  confidence_threshold: 0.25
```

---

## 25. Implementation Phases

## Phase 1 — Performance-first pipeline skeleton with mock models

Implement:

```text
repository structure
requirements.txt
run_pipeline.py
config loader
logger
PipelineContext
pipeline orchestrator
image input support
conservative image preprocessing
tiling
mock detector
mock segmenter
quantity engine
JSON exporter
Excel exporter
debug visualizer
processing report writer
basic smoke test
```

Acceptance command:

```bash
python run_pipeline.py \
  --input sample_data/images/example.png \
  --output outputs/first_test \
  --detectors mock \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

Expected outputs:

```text
outputs/first_test/rendered.png
outputs/first_test/preprocessed.png
outputs/first_test/tiles/
outputs/first_test/raster_detections.json
outputs/first_test/segmentation_objects.json
outputs/first_test/merged_objects.json
outputs/first_test/quantities.json
outputs/first_test/quantities.xlsx
outputs/first_test/debug_overlay.png
outputs/first_test/processing_report.md
```

---

## Phase 2 — Real detection infrastructure

Implement:

```text
YOLODetector
RTDETRDetector
EnsembleDetector
NMS
optional Weighted Boxes Fusion
per-tile inference
coordinate restoration
model-missing fallback behavior
```

Acceptance commands:

```bash
python run_pipeline.py \
  --input sample_data/images/example.png \
  --output outputs/yolo_test \
  --detectors yolo \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

```bash
python run_pipeline.py \
  --input sample_data/images/example.png \
  --output outputs/rtdetr_test \
  --detectors rtdetr \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

If model file is missing:

- Print clear warning
- Fallback to mock only if config allows

---

## Phase 3 — Real segmentation infrastructure

Implement:

```text
SegFormer segmenter interface
Mask2Former segmenter interface or placeholder
mask-to-polygon conversion
wall mask skeletonization
room polygonization
```

Acceptance command:

```bash
python run_pipeline.py \
  --input sample_data/images/example.png \
  --output outputs/segmentation_test \
  --detectors mock \
  --segmenter segformer \
  --scale 0.01 \
  --debug
```

If model file is missing:

- Print clear warning
- Fallback to mock only if config allows

---

## Phase 4 — PDF support

Implement:

```text
PDF first-page rendering at 300 DPI
PDF text extraction
PDF vector primitive extraction where possible
```

Acceptance command:

```bash
python run_pipeline.py \
  --input sample_data/pdf/example.pdf \
  --output outputs/pdf_test \
  --detectors mock \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

The pipeline must complete even if vector extraction is weak.

---

## Phase 5 — DXF vector branch

Implement:

```text
DXF primitive extraction
basic DXF preview rendering
layer classification
block symbol extraction
wall candidate extraction
vector_primitives.json export
```

Acceptance command:

```bash
python run_pipeline.py \
  --input sample_data/dxf/example.dxf \
  --output outputs/dxf_test \
  --detectors mock \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

Expected:

- `vector_primitives.json` contains extracted primitives
- Wall candidates are created from strong wall layers
- Door/window objects can be created from block names when available

---

## Phase 6 — Fusion and graph construction

Implement:

```text
raster/vector fusion
text-to-room attachment
wall-to-room adjacency
opening-to-wall attachment
drawing_graph.json
confidence merging
```

Acceptance criteria:

- Merged objects contain evidence metadata
- Duplicate door/window detections are removed
- Graph relationships are exported

---

## Phase 7 — Evaluation pipeline

Implement:

```text
evaluate_outputs.py
detection count metrics
IoU metrics if ground truth boxes exist
quantity error metrics
report output
```

Acceptance command:

```bash
python evaluate_outputs.py \
  --pred outputs/first_test/merged_objects.json \
  --gt sample_data/ground_truth/example_objects.json
```

---

## Phase 8 — Training scripts and dataset preparation

Implement:

```text
prepare_detection_dataset.py
prepare_segmentation_dataset.py
train_yolo.py
train_rtdetr.py
train_segformer.py
train_mask2former.py placeholder if needed
```

Acceptance criteria:

- Dataset format is documented
- Training scripts have clear CLI arguments
- Model output location matches detector config

---

## 26. First Exact Codex Task

Start with Phase 1 only.

Implement:

```text
1. Repository structure
2. requirements.txt
3. run_pipeline.py
4. configs/default.yaml
5. PipelineContext
6. Pipeline orchestrator
7. Image loader
8. Image preprocessor
9. Tiler
10. MockDetector
11. MockSegmenter
12. Simple fusion that combines mock outputs
13. QuantityEngine
14. JSON exporter
15. Excel exporter
16. Debug overlay writer
17. Processing report writer
18. One smoke test
```

Do not implement PDF, DXF, YOLO, RT-DETR, SegFormer, or Mask2Former in the first commit.

The first command must work:

```bash
python run_pipeline.py \
  --input sample_data/images/example.png \
  --output outputs/first_test \
  --detectors mock \
  --segmenter mock \
  --scale 0.01 \
  --debug
```

If no sample image exists, create a simple synthetic floor-plan image automatically under:

```text
sample_data/images/example.png
```

Expected outputs:

```text
outputs/first_test/rendered.png
outputs/first_test/preprocessed.png
outputs/first_test/tiles/
outputs/first_test/raster_detections.json
outputs/first_test/segmentation_objects.json
outputs/first_test/merged_objects.json
outputs/first_test/quantities.json
outputs/first_test/quantities.xlsx
outputs/first_test/debug_overlay.png
outputs/first_test/processing_report.md
```

---

## 27. Second Exact Codex Task

After Phase 1 works, implement real detection infrastructure.

Implement:

```text
1. YOLODetector
2. RTDETRDetector
3. per-tile detector execution
4. tile coordinate restoration
5. NMS deduplication
6. model missing fallback
7. detector config loading
8. model placement documentation
```

Commands:

```bash
python run_pipeline.py --input sample_data/images/example.png --output outputs/yolo_test --detectors yolo --segmenter mock --scale 0.01 --debug

python run_pipeline.py --input sample_data/images/example.png --output outputs/rtdetr_test --detectors rtdetr --segmenter mock --scale 0.01 --debug
```

---

## 28. Third Exact Codex Task

After detection infrastructure works, implement PDF and DXF branches.

Implement:

```text
1. PDF rendering
2. PDF text extraction
3. PDF vector extraction attempt
4. DXF primitive extraction
5. DXF preview rendering
6. layer classifier
7. wall candidate extractor
8. block symbol extractor
```

Commands:

```bash
python run_pipeline.py --input sample_data/pdf/example.pdf --output outputs/pdf_test --detectors mock --segmenter mock --scale 0.01 --debug

python run_pipeline.py --input sample_data/dxf/example.dxf --output outputs/dxf_test --detectors mock --segmenter mock --scale 0.01 --debug
```

---

## 29. Definition of Done for Pipeline Prototype

The pipeline prototype is done when:

```text
1. It accepts image, PDF, and DXF input.
2. It renders or normalizes input to an image.
3. It tiles high-resolution drawings.
4. It can run mock detection/segmentation end-to-end.
5. It can optionally run YOLO detection.
6. It can optionally run RT-DETR detection.
7. It can support SegFormer/Mask2Former-style segmentation interfaces.
8. It extracts basic PDF/DXF vector primitives.
9. It creates wall candidates from vector data.
10. It fuses raster and vector outputs.
11. It builds a simple drawing graph.
12. It calculates room area, room perimeter, wall length, door count, and window count.
13. It exports JSON outputs.
14. It exports Excel quantities.
15. It exports debug overlay images.
16. It writes a processing report.
17. It has basic tests for geometry, tiling, detection merging, fusion, and quantity calculation.
18. It has an evaluation script for comparing predictions against ground truth.
```

---

## 30. Later MVP Product Layer

Only after this pipeline produces useful output, build the MVP product layer:

```text
FastAPI backend
React review UI
Project upload page
Drawing viewer
Manual correction tools
Database
Human verification workflow
Training data export from corrections
Excel/PDF reporting
```

The future UI should wrap this pipeline. Do not build the UI before the pipeline proves useful outputs.

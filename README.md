# AutoCAD Quantity Calculation Pipeline

Prototype pipeline for extracting quantities from architectural drawings. It accepts image, PDF, and DXF inputs, detects doors/windows, segments rooms/walls, fuses raster and vector evidence, builds a drawing graph, calculates quantities, and exports validation artifacts.

This repository intentionally contains source code, configs, scripts, tests, and small sample inputs only. Large generated assets such as datasets, trained checkpoints, run outputs, logs, and caches are ignored by git. Recreate them with the commands below.

## What It Does

The pipeline produces a structured quantity package from a CAD-like drawing:

- Renders or loads the drawing into an image.
- Splits large drawings into overlapping tiles.
- Detects door and window symbols with YOLO and/or RT-DETR.
- Segments room and wall pixels with SegFormer.
- Extracts vector primitives and text from PDF/DXF when available.
- Fuses raster detections, segmentation geometry, and vector hints.
- Builds graph relationships such as room contains text, room contains opening, wall has opening, and object adjacency.
- Calculates room areas/perimeters, wall lengths, and opening counts.
- Exports JSON, Excel, debug overlays, and a processing report.

The current model-backed path is:

- `YOLO + RT-DETR` for doors/windows.
- `SegFormer` for rooms/walls.
- DXF/PDF vector extraction where available.

The project also includes a temporary local validation UI for trying uploads before a production UI is designed.

## How It Works

The main entrypoint is `run_pipeline.py`. It creates a `PipelineContext` and runs these stages:

1. `detect_file_type`: classifies input as image, PDF, DXF, or unsupported.
2. `render_or_load_input`: loads image files, renders PDF first page with PyMuPDF, or renders a DXF preview with `ezdxf`.
3. `preprocess_image`: normalizes the rendered image for downstream models.
4. `create_tiles`: creates overlapping image tiles for detection.
5. `extract_vectors`: extracts PDF/DXF vectors, DXF blocks, wall candidates, and text blocks.
6. `run_symbol_detectors`: runs selected detectors per tile and restores global coordinates.
7. `merge_tiled_detections`: merges duplicate tiled detections.
8. `run_room_wall_segmentation`: runs selected segmenter and converts masks into room polygons and wall centerlines.
9. `run_ocr`: placeholder stage; text currently comes from vector/PDF/DXF extraction.
10. `fuse_raster_vector_results`: merges raster, segmentation, vector, and text evidence.
11. `build_drawing_graph`: creates semantic relationships among rooms, walls, openings, and text.
12. `calculate_quantities`: calculates area, perimeter, wall length, and opening counts.
13. `export_outputs`: writes JSON, Excel, and debug overlay artifacts.
14. `write_processing_report`: writes a markdown report with warnings, errors, metrics, and configuration.

Important source folders:

- `src/detection`: YOLO, RT-DETR, detector factory, tiled model loading.
- `src/segmentation`: SegFormer runtime, mask-to-geometry conversion, room/wall post-processing.
- `src/vector`: DXF/PDF vector-to-object helpers and raster/vector fusion.
- `src/geometry`: drawing objects, graph construction, geometry utilities.
- `src/quantity`: quantity calculation.
- `src/export`: JSON, Excel, debug overlays, reports.
- `src/training`: dataset preparation and training entrypoints.
- `tests`: smoke, geometry, fallback, evaluation, and pipeline tests.

## Repository Setup

Use Windows PowerShell from a fresh clone:

```powershell
git clone https://github.com/AralSalehyan/AutocadQuantityCalculationPipeline.git
cd AutocadQuantityCalculationPipeline
```

Recommended Python:

- Python `3.12`
- For GPU training/inference, use an environment with CUDA-enabled PyTorch.
- During development on this machine, the working runtime was `C:\Users\arals\anaconda3\python.exe`.

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA PyTorch first if you want GPU inference/training. Pick the command that matches your CUDA version from the official PyTorch installer. For CUDA 12.1:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you only want CPU smoke tests, normal `pip install -r requirements.txt` is enough, but trained-model inference will be slow.

## Required Model Files

The current runtime checkpoints for the best pipeline are committed to this repository:

```text
models/yolo/door_window.pt
models/rtdetr/door_window.pt
models/segformer/room_wall/model.safetensors
models/segformer/room_wall/config.json
models/segformer/room_wall/preprocessor_config.json
```

After cloning and installing dependencies, these files let you run the trained pipeline without retraining. The runtime configs point to these paths by default.

Training artifacts, datasets, base pretrained weights, and smoke/benchmark checkpoints are still ignored by git. If you want to replace the included runtime checkpoints, train the models with the commands below or copy new checkpoints into the same paths.

Mock mode does not require model files:

```powershell
python run_pipeline.py --input sample_data/images/example.png --output outputs/mock_test --detectors mock --segmenter mock --scale 0.01 --debug
```

## Dataset Setup

Large datasets are ignored by git. Put raw data under `datasets/raw`.

Download and prepare CubiCasa5K for door/window detection:

```powershell
python src/training/download_dataset.py --dataset cubicasa5k --output-dir datasets/raw --extract
python src/training/prepare_detection_dataset.py --source datasets/raw/cubicasa5k --output datasets/door_window
```

Prepare room/wall segmentation masks:

```powershell
python src/training/prepare_segmentation_dataset.py --source datasets/raw/cubicasa5k.zip --output datasets/room_wall
```

Expected segmentation labels:

- `0`: background
- `1`: room
- `2`: wall

Detection labels use YOLO text format:

- class id
- normalized center x
- normalized center y
- normalized width
- normalized height

## Training

Train YOLO door/window detector:

```powershell
python src/training/train_yolo.py --data datasets/door_window/data.yaml --model yolo11m.pt --epochs 50 --imgsz 1280 --batch 2 --workers 0 --device 0
```

Train RT-DETR door/window detector:

```powershell
python src/training/train_rtdetr.py --data datasets/door_window/data.yaml --epochs 50 --imgsz 1024 --batch 1 --workers 0 --device 0 --no-plots
```

Train SegFormer room/wall segmenter:

```powershell
python src/training/train_segformer.py --data datasets/room_wall/data.yaml --output models/segformer/room_wall --epochs 20 --imgsz 512 --batch 2 --device cuda
```

Resume SegFormer after an interrupted run:

```powershell
python src/training/train_segformer.py --data datasets/room_wall/data.yaml --output models/segformer/room_wall --resume models/segformer/room_wall --epochs 20 --imgsz 512 --batch 2 --device cuda
```

Windows helper scripts are available in `scripts/` for long RT-DETR and SegFormer runs.

## Run The Pipeline

Mock end-to-end run:

```powershell
python run_pipeline.py --input sample_data/images/example.png --output outputs/first_test --detectors mock --segmenter mock --scale 0.01 --debug
```

Full trained image run:

```powershell
python run_pipeline.py --input sample_data/images/example.png --output outputs/full_trained --detectors yolo,rtdetr --segmenter segformer --scale 0.01 --debug
```

This is the best current end-to-end path: YOLO and RT-DETR detect doors/windows, SegFormer segments rooms/walls, and the pipeline exports quantities, graph relationships, overlays, JSON, Excel, and a processing report.

PDF run:

```powershell
python run_pipeline.py --input sample_data/pdf/example.pdf --output outputs/pdf_test --detectors yolo,rtdetr --segmenter segformer --scale 0.01 --debug
```

DXF run:

```powershell
python run_pipeline.py --input sample_data/dxf/example.dxf --output outputs/dxf_test --detectors yolo,rtdetr --segmenter segformer --scale 0.01 --debug
```

If the sample image/PDF/DXF does not exist, `run_pipeline.py` creates a small synthetic sample automatically.

## Temporary Validation UI

Start the local validation UI:

```powershell
python ui_app.py
```

Open:

```text
http://127.0.0.1:8765
```

The UI lets you:

- Upload `.png`, `.jpg`, `.jpeg`, `.pdf`, or `.dxf`.
- Choose detectors and segmenter.
- Run the pipeline in the background.
- Inspect debug overlay, quantities, report, object counts, and graph edge count.
- Save a validation decision: `approved`, `needs_review`, or `rejected`.

UI runs are stored under:

```text
outputs/ui_runs/
```

This UI is intentionally basic and exists only for pipeline validation. It is not the final product UI.

## Outputs

Each pipeline run writes:

- `rendered.png`
- `preprocessed.png`
- `tiles/`
- `vector_primitives.json`
- `raster_detections.json`
- `segmentation_objects.json`
- `merged_objects.json`
- `drawing_graph.json`
- `quantities.json`
- `quantities.xlsx`
- `debug_overlay.png`
- `debug_overlay_tiles.png`
- `processing_report.md`

Evaluation runs also write:

- `evaluation_report.json`
- `evaluation_report.md`

## Evaluation

Compare predictions against ground truth:

```powershell
python evaluate_outputs.py --pred outputs/first_test/merged_objects.json --gt sample_data/ground_truth/example_objects.json --output outputs/first_test
```

Evaluation reports include:

- door/window IoU matching
- precision
- recall
- F1
- false positives
- false negatives
- count errors
- quantity errors

## Tests

Run tests:

```powershell
python -m pytest -q
```

Compile check:

```powershell
python -m compileall ui_app.py run_pipeline.py evaluate_outputs.py src tests
```

## Current Limitations

- OCR is currently a placeholder; text comes from PDF/DXF extraction where available.
- SegFormer wall geometry still benefits from post-processing and domain-specific cleanup.
- DXF block/layer classification uses practical naming heuristics.
- The validation UI is intentionally temporary and single-machine/local.
- Large datasets and trained checkpoints must be recreated or copied into expected local paths.

# Implementation Log

## 2026-06-01 Phase 1

- Started implementation from `plan.md`, scoped to "First Exact Codex Task" / Phase 1.
- Initial workspace contained only `plan.md`; `git status` failed because the folder is not a Git repository.
- Added Phase 1 repository scaffold, configuration, image pipeline, mock detector/segmenter, quantity engine, exporters, debug visualizers, report writer, and smoke test.
- Verification passed: `python run_pipeline.py --input sample_data/images/example.png --output outputs/first_test --detectors mock --segmenter mock --scale 0.01 --debug`.
- Verification passed: `python -m pytest tests/test_pipeline_smoke.py -q`.
- Verification passed: `python -m compileall run_pipeline.py src tests`.
- No current implementation failures.

## 2026-06-02 Phase 2

- Started implementation of real detection infrastructure: YOLO/RT-DETR adapters, per-tile execution, coordinate restoration, NMS deduplication, detector config loading, and missing-model fallback.
- Added `configs/detectors.yaml` and `configs/segmenters.yaml`.
- Added detector adapters: `YOLODetector`, `RTDETRDetector`, `EnsembleDetector`, shared NMS, and a weighted-boxes-fusion placeholder that currently delegates to NMS.
- Updated the pipeline to execute real detectors per tile, restore tile coordinates to full-image coordinates, and deduplicate detections with NMS.
- Added missing-model fallback behavior controlled by `detectors.fallback_to_mock`.
- Added model placement documentation in `README.md` and `models/README.md`.
- Added tests for coordinate restoration, NMS, and YOLO/RT-DETR missing-model fallback.
- Verification passed: `python run_pipeline.py --input sample_data/images/example.png --output outputs/yolo_test --detectors yolo --segmenter mock --scale 0.01 --debug`.
- Verification passed: `python run_pipeline.py --input sample_data/images/example.png --output outputs/rtdetr_test --detectors rtdetr --segmenter mock --scale 0.01 --debug`.
- Verification passed: `python -m pytest -q`.
- Verification passed: `python -m compileall run_pipeline.py src tests`.
- No current implementation failures. The configured YOLO and RT-DETR model files are absent, so acceptance commands exercised the intended mock fallback path.

## 2026-06-04 Training Setup

- Started model training preparation.
- Selected CubiCasa5K as the first public dataset target. Source: Zenodo record `2613548`, archive `cubicasa5k.zip`, 5.5 GB, MD5 `0ce0b203d1e3c125b51087b219bd23b9`.
- Added dataset downloader, CubiCasa SVG-to-YOLO door/window converter, YOLO training script, RT-DETR training script, and dataset notes.
- Updated YOLO training default to `yolo11m.pt`; small YOLO is no longer the default training model.
- Downloaded `datasets/raw/cubicasa5k.zip` from Zenodo and verified MD5.
- Prepared a disk-conscious starter YOLO dataset directly from the ZIP: 450 train images, 50 validation images, 9,728 labeled door/window objects.
- Installed `ultralytics`.
- Initial CUDA PyTorch install failed with `No space left on device`; purged pip cache and retried with `--no-cache-dir`.
- CUDA PyTorch is now enabled: `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, CUDA available on `NVIDIA GeForce RTX 3070 Laptop GPU`.
- Downloaded larger starter weights: `yolo11m.pt` and `rtdetr-l.pt`.
- Verification passed: `python -m pytest -q`.
- Verification passed: `python -m compileall src tests run_pipeline.py`.
- Training is ready to start with GPU using `yolo11m.pt` or `rtdetr-l.pt`.
- Training launch issue: direct script execution initially failed because repo root was not on `sys.path`; updated YOLO and RT-DETR training scripts to support direct file execution.
- YOLO training was started with `yolo11m.pt`, 50 epochs, image size 1280, batch 4, device 0.
- Current status check: Python training processes are active, GPU process is attached, and Ultralytics created `runs/detect/models/yolo/runs/door_window/args.yaml`. No `results.csv` epoch metrics were present yet at the time of this check.
- Later status check: no Python training process is active, GPU memory is idle, and the run directory contains only `args.yaml`; no `results.csv`, batch images, or weights were written. The run stopped before completing the first epoch, and the foreground output was not captured because the tool call was aborted.
- Training restart failed before epoch 1 with Windows `WinError 1455` page-file exhaustion while spawning dataloader/CUDA worker processes. Added `--workers` and `--amp/--no-amp` controls to YOLO and RT-DETR training scripts; default workers set to `0` for Windows stability.
- Stable YOLO training run `door_window-3` was stopped on request. Checkpoints exist at `runs/detect/models/yolo/runs/door_window-3/weights/last.pt` and `best.pt`.
- Last recorded training epoch before stop: epoch `28` in `runs/detect/models/yolo/runs/door_window-3/results.csv`.
- Resume command: `python src/training/train_yolo.py --model runs/detect/models/yolo/runs/door_window-3/weights/last.pt --resume --device 0`.
- YOLO training resumed from epoch 29 and completed epoch 50. Final wrapper exit was nonzero because Matplotlib failed while saving a final PR curve (`ValueError: object __array__ method not producing an array`), after weights and `results.csv` were written.
- Added `--plots/--no-plots` controls to YOLO and RT-DETR training scripts to avoid plot-generation failures on future runs.
- Copied trained YOLO best checkpoint to runtime model path: `models/yolo/door_window.pt`.
- Final YOLO epoch 50 metrics from `door_window-3/results.csv`: precision `0.6798`, recall `0.71252`, mAP50 `0.66246`, mAP50-95 `0.45286`.
- Verification passed: `python run_pipeline.py --input sample_data/images/example.png --output outputs/yolo_trained_test --detectors yolo --segmenter mock --scale 0.01 --debug`.
- Runtime YOLO verification used trained model with no fallback warnings; it produced 6 tiled detections, merged to 4 detections on the synthetic sample.

## 2026-06-06 Phase 3

- Started real segmentation infrastructure.
- Added SegFormer adapter, Mask2Former placeholder adapter, segmenter factory, mask-to-polygon conversion, room polygonizer, and wall mask skeletonizer.
- Updated pipeline segmentation stage to use segmenter config and missing-model fallback behavior.
- Acceptance command passed: `python run_pipeline.py --input sample_data/images/example.png --output outputs/segmentation_test --detectors mock --segmenter segformer --scale 0.01 --debug`.
- Test cleanup: YOLO fallback test was updated to account for the now-trained runtime YOLO model, and wall skeletonization now falls back when `skimage` has binary compatibility issues.

## 2026-06-06 Phase 4

- Started PDF support implementation.
- Added `PDFLoader` with first-page rendering, text extraction, and basic vector primitive extraction via PyMuPDF.
- Added `PDFVectorExtractor` wrapper.
- Updated pipeline to accept PDF inputs and preserve PDF text blocks through the OCR stage.
- Added automatic synthetic `sample_data/pdf/example.pdf` creation for the acceptance command.
- Installed PyMuPDF in the local Python environment.
- Added PDF loader and PDF pipeline tests.
- Acceptance command passed: `python run_pipeline.py --input sample_data/pdf/example.pdf --output outputs/pdf_test --detectors mock --segmenter mock --scale 0.01 --debug`.
- PDF verification output: rendered image `3500 x 2250`, 12 tiles, 7 vector primitives, 2 text blocks, 11 merged objects, no warnings.
- Verification passed: `python -m pytest -q` with 12 tests.
- Verification passed: `python -m compileall run_pipeline.py src tests`.

## 2026-06-06 RT-DETR Training

- RT-DETR-L initial training at image size 1280, batch 1 failed on the first backward pass with CUDA out-of-memory on the RTX 3070 Laptop GPU.
- Restarted RT-DETR-L at image size 1024, batch 1, workers 0, no plots. The process reached epoch 15/50 before the 4-hour command timeout stopped it.
- Resume checkpoint exists at `runs/detect/models/rtdetr/runs/door_window_1024/weights/last.pt`; best checkpoint exists at `runs/detect/models/rtdetr/runs/door_window_1024/weights/best.pt`.
- Last recorded epoch 15 metrics: precision `0.5444`, recall `0.59252`, mAP50 `0.49027`, mAP50-95 `0.31973` from `runs/detect/models/rtdetr/runs/door_window_1024/results.csv`.
- Estimated remaining RT-DETR-L time from epoch 15 to 50 is about 9 hours at the observed training pace.
- Lowered the RT-DETR training wrapper default image size to 1024 because 1280 failed on the available 8 GB CUDA GPU.
- Added `scripts/run_rtdetr_resume.bat` to launch a resumable RT-DETR training job with durable stdout/stderr logs.
- Added `scripts/launch_rtdetr_resume.cmd` to start the RT-DETR batch launcher in a separate minimized Windows command process.
- Normal sandboxed/background launches were killed with the tool session; the RT-DETR launcher had to be run with elevated execution so the long CUDA process could continue independently.
- RT-DETR training is currently active as Python PID `36576`, resumed from epoch 16/50 at image size 1024, batch 1, workers 0.
- Active training logs: `logs/rtdetr_scheduled_1024_20260606.out.log` and `logs/rtdetr_scheduled_1024_20260606.err.log`.
- Verification passed: `python -m py_compile src\training\train_rtdetr.py`.
- RT-DETR training completed epoch 50. Final epoch 50 metrics: precision `0.69118`, recall `0.67763`, mAP50 `0.62285`, mAP50-95 `0.48298`.
- Copied trained RT-DETR best checkpoint to runtime model path: `models/rtdetr/door_window.pt`.
- Verification passed: `python run_pipeline.py --input sample_data\images\example.png --output outputs\rtdetr_trained_test --detectors rtdetr --segmenter mock --scale 0.01 --debug`.
- Runtime RT-DETR verification used trained model with no fallback warnings; it produced 8 tiled detections, merged to 4 detections on the synthetic sample.

## 2026-06-11 Phase 5

- Started DXF vector branch implementation.
- Installed `ezdxf` for DXF parsing.
- Added `DXFParser` for LINE, LWPOLYLINE, POLYLINE, INSERT, TEXT, MTEXT, CIRCLE, ARC, and DIMENSION primitives.
- Added simple DXF preview rendering to `rendered.png`.
- Added layer classification with English/Turkish CAD naming hints.
- Added wall candidate extraction from wall layers and orthogonal long polylines.
- Added block symbol extraction for door/window blocks.
- Updated the pipeline to accept DXF inputs, extract vector objects, preserve DXF text blocks, and pass vector objects into fusion.
- Added automatic synthetic `sample_data/dxf/example.dxf` creation for the acceptance command.
- Added DXF parser and DXF pipeline tests.
- Acceptance command passed: `python run_pipeline.py --input sample_data\dxf\example.dxf --output outputs\dxf_test --detectors mock --segmenter mock --scale 0.01 --debug`.
- DXF verification output: rendered image `1380 x 840`, 13 vector primitives, 10 vector objects, 3 text blocks, 22 merged objects, no warnings.
- Updated the RT-DETR fallback test to account for the now-trained runtime RT-DETR model.
- Verification passed: `python -m pytest -q` with 14 tests.
- Verification passed: `python -m compileall run_pipeline.py src tests`.

## 2026-06-11 Phase 6

- Started raster/vector fusion and graph construction improvements.
- Added geometry distance helpers for point/polyline, bbox/polyline, and polygon-boundary relationships.
- Updated `VectorRasterFusion` to keep segmentation room polygons, prefer vector walls when vector wall candidates exist, merge raster/vector door-window duplicates, and add source-evidence metadata.
- Added text-to-room attachment during fusion; room objects now record attached text ids and text objects record attached room ids.
- Updated `DrawingGraphBuilder` to export room-text containment, room-opening containment, room-wall adjacency, wall-opening, opening-near-wall, and text-near-object relationships.
- Added focused fusion/graph tests.
- Acceptance command passed: `python run_pipeline.py --input sample_data\dxf\example.dxf --output outputs\phase6_dxf_test --detectors mock --segmenter mock --scale 0.01 --debug`.
- Phase 6 DXF verification output: 16 merged objects, 28 graph edges, 3 fused raster/vector opening objects, and source evidence on all merged objects.
- Verification passed: `python -m pytest -q` with 16 tests.
- Verification passed: `python -m compileall run_pipeline.py src tests`.

## 2026-06-11 Phase 7

- Started evaluation pipeline implementation.
- Added `evaluate_outputs.py` CLI for comparing predicted merged objects and quantities against ground truth.
- Detection evaluation supports door/window matching by IoU, precision, recall, F1, false positives, false negatives, count error per class, and average IoU.
- Quantity evaluation supports absolute and percentage errors for room area, room perimeter, wall length, total wall length, door count, and window count.
- Evaluation writes `evaluation_report.json` and `evaluation_report.md`.
- Added `sample_data/ground_truth/example_objects.json` with objects and quantities for the synthetic sample drawing.
- Added evaluation tests covering perfect matches, false-positive/false-negative behavior, and CLI report writing.
- Acceptance command passed: `python evaluate_outputs.py --pred outputs\first_test\merged_objects.json --gt sample_data\ground_truth\example_objects.json --output outputs\first_test`.
- Verification passed: `python -m pytest -q` with 19 tests.
- Verification passed: `python -m compileall run_pipeline.py evaluate_outputs.py src tests`.

## 2026-06-11 Phase 8

- Started training scripts and dataset preparation completion.
- Implemented `prepare_segmentation_dataset.py` for room/wall semantic segmentation masks from CubiCasa-style SVG annotations.
- Segmentation dataset format now writes `images/train`, `images/val`, `masks/train`, `masks/val`, `data.yaml`, and `metadata.json`.
- Segmentation class ids are documented as `0=background`, `1=room`, `2=wall`.
- Replaced the SegFormer training placeholder with a CLI training loop that uses `torch` and `transformers` when optional training dependencies are installed.
- SegFormer training output defaults to `models/segformer/room_wall`, matching `configs/segmenters.yaml`.
- Replaced the Mask2Former placeholder with a clear CLI entrypoint that validates dataset presence and explains that a project-specific training loop/checkpoint format is still required.
- Updated dataset notes and README with detection and segmentation dataset formats, training commands, and model output locations.
- Added tests for segmentation mask preparation and segmentation training config parsing.
- Verification passed: `python -m pytest -q` with 22 tests.
- Verification passed: `python -m compileall run_pipeline.py evaluate_outputs.py src tests`.

## 2026-06-11 Pipeline Hardening

- Added workspace-local runtime cache directories for Ultralytics, Matplotlib, Torch, and XDG cache state.
- Quieted noisy third-party loggers so debug pipeline output remains focused on pipeline stages.
- Suppressed noisy `ezdxf` import-time Windows font-cache access warnings while keeping real DXF import failures explicit.
- Acceptance passed for image input: `python run_pipeline.py --input sample_data\images\example.png --output outputs\accept_image --detectors mock --segmenter mock --scale 0.01 --debug`.
- Acceptance passed for PDF input: `python run_pipeline.py --input sample_data\pdf\example.pdf --output outputs\accept_pdf --detectors mock --segmenter mock --scale 0.01 --debug`.
- Acceptance passed for DXF input: `python run_pipeline.py --input sample_data\dxf\example.dxf --output outputs\accept_dxf --detectors mock --segmenter mock --scale 0.01 --debug`.
- Acceptance passed for evaluation output: `python evaluate_outputs.py --pred outputs\first_test\merged_objects.json --gt sample_data\ground_truth\example_objects.json --output outputs\accept_eval`.
- Acceptance passed for combined trained detectors: `python run_pipeline.py --input sample_data\images\example.png --output outputs\accept_yolo_rtdetr --detectors yolo,rtdetr --segmenter mock --scale 0.01 --debug`.
- Combined detector verification produced 6 YOLO tile detections, 8 RT-DETR tile detections, 10 merged objects, and 29 graph edges with no pipeline warnings or errors.
- Environment note: Ultralytics emits a non-blocking SciPy warning because the installed NumPy version is newer than SciPy's declared compatible range; inference still completed successfully.
- Verification passed: `python -m pytest -q` with 22 tests.
- Verification passed: `python -m compileall run_pipeline.py evaluate_outputs.py src tests`.

## 2026-06-12 SegFormer Fine-Tuning

- Started room/wall SegFormer fine-tuning setup from local `datasets/raw/cubicasa5k.zip`.
- Prepared a usable segmentation dataset at `datasets/room_wall` with 3,889 train image/mask pairs and 500 validation pairs before the full conversion command timed out.
- Added missing `datasets/room_wall/data.yaml` for the prepared room/wall dataset.
- Repaired the Anaconda training environment by cleanly reinstalling `numpy==1.26.4`; this fixed SciPy, pandas, scikit-learn, and Transformers imports.
- Updated runtime setup to force Transformers into PyTorch-only mode with `USE_TF=0`, `USE_FLAX=0`, and `TRANSFORMERS_NO_TF=1`.
- Verified CUDA training environment: Anaconda Python sees `NVIDIA GeForce RTX 3070 Laptop GPU`.
- Ran SegFormer smoke training successfully and saved `models/segformer/room_wall_smoke`.
- Added resumable SegFormer training support with checkpoint saving and `training_state.json`.
- Added SegFormer launch/resume scripts: `scripts/run_segformer_train.bat`, `scripts/resume_segformer_train.bat`, `scripts/launch_segformer_train.cmd`, and `scripts/launch_segformer_resume.cmd`.
- Started full SegFormer training in the background as Anaconda Python PID `25116`.
- Active logs: `logs/segformer_room_wall.out.log` and `logs/segformer_room_wall.err.log`.
- Expected training time for 20 epochs at image size 512 and batch 2 is about 5-7 hours.
- Verification passed: `C:\Users\arals\anaconda3\python.exe -m py_compile src\training\train_segformer.py src\utils\runtime_paths.py`.

## 2026-06-13 Full Trained Pipeline Completion

- SegFormer training completed successfully at epoch 20/20 with final train loss `0.234877`.
- Runtime SegFormer checkpoint is available at `models/segformer/room_wall`.
- Fixed the SegFormer runtime adapter confidence-map batch dimension bug.
- Improved wall mask post-processing by extracting longer Hough centerlines from skeletonized wall masks and filtering tiny fragments.
- Made SegFormer processor loading explicit with `use_fast=False` to avoid runtime warning noise.
- Updated pipeline config loading so explicit `--config` files override default detector/segmenter model paths.
- Updated the segmenter fallback test so it explicitly uses a missing temporary SegFormer model path instead of depending on repository model state.
- Full trained acceptance passed: `C:\Users\arals\anaconda3\python.exe run_pipeline.py --input sample_data\images\example.png --output outputs\accept_full_trained --detectors yolo,rtdetr --segmenter segformer --scale 0.01 --debug`.
- Full trained acceptance output: 8 raster detections, 35 segmentation objects, 40 merged objects, and 190 graph edges with no pipeline warnings or errors.
- Verification passed: `C:\Users\arals\anaconda3\python.exe -m pytest -q` with 22 tests.
- Verification passed: `C:\Users\arals\anaconda3\python.exe -m compileall run_pipeline.py evaluate_outputs.py src tests`.

## 2026-06-13 Basic Validation UI

- Added `ui_app.py`, a temporary standard-library local UI server for uploading `.png`, `.jpg`, `.jpeg`, `.pdf`, and `.dxf` files.
- UI runs the existing pipeline in a background thread and stores run state under `outputs/ui_runs`.
- UI supports detector/segmenter/scale selection, run polling, debug overlay preview, processing report display, quantity table display, object counts, graph edge count, and validation decisions.
- Validation decisions are saved to each run as `validation.json` with `approved`, `needs_review`, or `rejected` status and notes.
- Started the local UI server as Anaconda Python PID `26528` at `http://127.0.0.1:8765`.
- UI smoke upload passed using `sample_data/images/example.png` with mock detector/segmenter.
- UI smoke run completed with overlay/report/Excel artifacts and validation save succeeded.
- Updated `README.md` with the temporary UI start command.
- Verification passed: `C:\Users\arals\anaconda3\python.exe -m pytest -q` with 22 tests.
- Verification passed: `C:\Users\arals\anaconda3\python.exe -m compileall ui_app.py run_pipeline.py evaluate_outputs.py src tests`.

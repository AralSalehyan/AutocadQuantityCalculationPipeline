# Dataset Notes

## CubiCasa5K

CubiCasa5K is the first training target for this repo because it is public, includes 5,000 floor-plan samples, and has SVG polygon annotations for floor-plan categories including doors and windows.

Source: https://zenodo.org/records/2613548

License: Creative Commons Attribution Non Commercial Share Alike 4.0 International.

Use:

```bash
python src/training/download_dataset.py --dataset cubicasa5k --output-dir datasets/raw --extract
python src/training/prepare_detection_dataset.py --source datasets/raw/cubicasa5k --output datasets/door_window
python src/training/train_yolo.py --data datasets/door_window/data.yaml --model yolo11m.pt --epochs 50 --imgsz 1280 --batch 2 --workers 0 --device 0
python src/training/train_rtdetr.py --data datasets/door_window/data.yaml --epochs 50 --imgsz 1024 --batch 1 --workers 0 --device 0 --no-plots
python src/training/prepare_segmentation_dataset.py --source datasets/raw/cubicasa5k --output datasets/room_wall
python src/training/train_segformer.py --data datasets/room_wall/data.yaml --output models/segformer/room_wall --epochs 20 --imgsz 512 --batch 2
```

## Dataset Formats

Detection dataset:

```text
datasets/door_window/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

Detection labels use YOLO rows:

```text
class_id center_x center_y width height
```

All coordinates are normalized to `[0, 1]`. Classes are `0=door`, `1=window`.

Segmentation dataset:

```text
datasets/room_wall/
  images/train/*.png
  images/val/*.png
  masks/train/*.png
  masks/val/*.png
  data.yaml
  metadata.json
```

Segmentation masks are single-channel PNG files with class ids:

```text
0 background
1 room
2 wall
```

Runtime model output locations match config defaults:

```text
models/yolo/door_window.pt
models/rtdetr/door_window.pt
models/segformer/room_wall/
models/mask2former/room_wall/
```

## Other Candidate Datasets

- CVC-FP for structural parsing.
- FloorPlanCAD for CAD symbol spotting.
- ArchCAD-400K for larger-scale CAD symbol detection if access is available.
- CubiCasa5K for room/wall/opening segmentation pretraining.

Open datasets are useful for pretraining and prototype validation, but production quality requires Turkish architectural drawings with local symbols, Turkish room labels, layer names, scales, legends, and drawing conventions.

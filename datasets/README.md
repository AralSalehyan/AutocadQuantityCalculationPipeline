# Datasets

Training data is downloaded and prepared here.

## CubiCasa5K Detection Training

```bash
python src/training/download_dataset.py --dataset cubicasa5k --output-dir datasets/raw --extract
python src/training/prepare_detection_dataset.py --source datasets/raw/cubicasa5k --output datasets/door_window
python src/training/train_yolo.py --data datasets/door_window/data.yaml --model yolo11m.pt --epochs 50 --imgsz 1280 --batch 2 --workers 0 --device 0
python src/training/train_rtdetr.py --data datasets/door_window/data.yaml --epochs 50 --imgsz 1280
```

The prepared detector dataset has this shape:

```text
datasets/door_window/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

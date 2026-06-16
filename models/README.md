# Runtime Models

This directory contains the current runtime checkpoints needed for the best pipeline path.

Tracked runtime files:

- `models/yolo/door_window.pt`
- `models/rtdetr/door_window.pt`
- `models/segformer/room_wall/model.safetensors`
- `models/segformer/room_wall/config.json`
- `models/segformer/room_wall/preprocessor_config.json`
- `models/segformer/room_wall/training_state.json`

These are enough to run:

```powershell
python run_pipeline.py --input sample_data/images/example.png --output outputs/full_trained --detectors yolo,rtdetr --segmenter segformer --scale 0.01 --debug
```

Detection model classes:

- `door`
- `window`

Segmentation class ids:

- `0`: background
- `1`: room
- `2`: wall

Not tracked:

- raw datasets
- training run directories
- base pretrained weights such as `yolo11m.pt` and `rtdetr-l.pt`
- SegFormer smoke/benchmark checkpoints
- generated outputs and logs

If a runtime checkpoint is replaced later, keep the same path so the default configs continue to work.

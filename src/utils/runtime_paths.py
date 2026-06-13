import os
from pathlib import Path


def configure_workspace_runtime_dirs() -> None:
    runtime_dir = Path(".runtime")
    ultralytics_dir = runtime_dir / "ultralytics"
    matplotlib_dir = runtime_dir / "matplotlib"
    torch_dir = runtime_dir / "torch"
    cache_dir = runtime_dir / "cache"
    for path in (ultralytics_dir, matplotlib_dir, torch_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    os.environ.setdefault("TORCH_HOME", str(torch_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

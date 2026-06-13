import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_paths import configure_workspace_runtime_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RT-DETR door/window detector.")
    parser.add_argument("--data", type=Path, default=Path("datasets/door_window/data.yaml"))
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", type=Path, default=Path("models/rtdetr/runs"))
    parser.add_argument("--name", default="door_window")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}. Run prepare_detection_dataset.py first.")

    configure_workspace_runtime_dirs()
    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install requirements before training.") from exc

    model = RTDETR(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "amp": args.amp,
        "plots": args.plots,
        "resume": args.resume,
        "project": str(args.project),
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    result = model.train(**train_kwargs)
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    target_path = Path("models/rtdetr/door_window.pt")
    if best_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(best_path.read_bytes())
        print(f"Copied best weights to {target_path}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

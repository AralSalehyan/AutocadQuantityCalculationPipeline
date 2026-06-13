import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask2Former room/wall training entrypoint placeholder.")
    parser.add_argument("--data", type=Path, default=Path("datasets/room_wall/data.yaml"))
    parser.add_argument("--model", default="facebook/mask2former-swin-tiny-ade-semantic")
    parser.add_argument("--output", type=Path, default=Path("models/mask2former/room_wall"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}. Run prepare_segmentation_dataset.py first.")
    raise SystemExit(
        "Mask2Former training is intentionally not implemented yet. "
        "Use train_segformer.py for the first room/wall segmentation checkpoint, or implement a project-specific "
        f"Mask2Former training loop that saves to {args.output}."
    )


if __name__ == "__main__":
    raise SystemExit(main())

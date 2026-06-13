import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_paths import configure_workspace_runtime_dirs


ID_TO_LABEL = {0: "background", 1: "room", 2: "wall"}
LABEL_TO_ID = {value: key for key, value in ID_TO_LABEL.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SegFormer room/wall semantic segmenter.")
    parser.add_argument("--data", type=Path, default=Path("datasets/room_wall/data.yaml"))
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--output", type=Path, default=Path("models/segformer/room_wall"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke training.")
    parser.add_argument("--resume", type=Path, default=None, help="Existing SegFormer checkpoint directory to continue from.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = read_data_yaml(args.data)
    train_images, train_masks = collect_pairs(config, "train", "train_masks")
    val_images, val_masks = collect_pairs(config, "val", "val_masks")
    if args.limit:
        train_images, train_masks = train_images[: args.limit], train_masks[: args.limit]
        val_images, val_masks = val_images[: max(1, args.limit // 5)], val_masks[: max(1, args.limit // 5)]
    if not train_images:
        raise SystemExit(f"No training image/mask pairs found for {args.data}. Run prepare_segmentation_dataset.py first.")

    configure_workspace_runtime_dirs()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as exc:
        raise SystemExit(
            "SegFormer training requires torch and transformers. Install optional training dependencies first."
        ) from exc

    model_source = args.resume or args.model
    if args.resume and not args.resume.exists():
        raise SystemExit(f"Resume checkpoint not found: {args.resume}")

    processor = AutoImageProcessor.from_pretrained(model_source, do_reduce_labels=False)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_source,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(device)

    train_dataset = SegmentationDataset(train_images, train_masks, processor, args.imgsz)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = read_start_epoch(args.resume, args.epochs)
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"epoch={epoch} train_loss={avg_loss:.6f}", flush=True)
        if epoch == args.epochs or args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.output, model, processor, epoch, avg_loss, args)

    print(f"Saved SegFormer room/wall model to {args.output}")
    if val_images:
        print(f"Validation pairs available: {len(val_images)}")
    return 0


class SegmentationDataset:
    def __init__(self, images: list[Path], masks: list[Path], processor, image_size: int):
        self.images = images
        self.masks = masks
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        image = Image.open(self.images[index]).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(self.masks[index]).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.from_numpy(np.array(mask, dtype=np.int64)),
        }


def read_start_epoch(resume: Path | None, target_epochs: int) -> int:
    if not resume:
        return 1
    state_path = resume / "training_state.json"
    if not state_path.exists():
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    last_epoch = int(state.get("last_epoch") or 0)
    if last_epoch >= target_epochs:
        raise SystemExit(f"Checkpoint already reached epoch {last_epoch}; target epochs is {target_epochs}.")
    return last_epoch + 1


def save_checkpoint(output: Path, model, processor, epoch: int, loss: float, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    state = {
        "last_epoch": epoch,
        "last_train_loss": loss,
        "target_epochs": args.epochs,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "learning_rate": args.lr,
    }
    (output / "training_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def collect_pairs(config: dict, image_key: str, mask_key: str) -> tuple[list[Path], list[Path]]:
    root = Path(config["path"])
    image_dir = root / config[image_key]
    mask_dir = root / config[mask_key]
    images = sorted(image_dir.glob("*.png"))
    pairs = [(image, mask_dir / image.name) for image in images if (mask_dir / image.name).exists()]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def read_data_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Dataset config not found: {path}. Run prepare_segmentation_dataset.py first.")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("pyyaml is required to read dataset configs.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

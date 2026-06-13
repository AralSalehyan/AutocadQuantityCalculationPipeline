import argparse
import io
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image


CLASS_TO_ID = {"door": 0, "window": 1}
IMAGE_NAMES = ("F1_original.png", "F1_scaled.png", "original.png", "image.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CubiCasa5K SVG labels to YOLO door/window labels.")
    parser.add_argument("--source", type=Path, required=True, help="Extracted CubiCasa5K root or nested data directory.")
    parser.add_argument("--output", type=Path, default=Path("datasets/door_window"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source.suffix.lower() == ".zip":
        samples = find_zip_samples(args.source)
        if args.limit:
            samples = samples[: args.limit]
        return prepare_zip_samples(args.source, samples, args.output, args.val_ratio, args.seed)

    samples = find_samples(args.source)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No CubiCasa samples with model.svg and image files found under {args.source}")

    random.Random(args.seed).shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio)) if len(samples) > 1 else 0
    val_samples = set(samples[:val_count])

    reset_yolo_dirs(args.output)
    counts = {"train": 0, "val": 0, "objects": 0}
    for sample in samples:
        split = "val" if sample in val_samples else "train"
        image_path, svg_path = sample
        boxes = extract_door_window_boxes(svg_path)
        if not boxes:
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        output_stem = safe_stem(image_path.parent)
        target_image = args.output / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
        target_label = args.output / "labels" / split / f"{output_stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(to_yolo_rows(boxes, width, height)) + "\n", encoding="utf-8")
        counts[split] += 1
        counts["objects"] += len(boxes)

    write_data_yaml(args.output)
    print(f"Prepared {counts['train']} train images, {counts['val']} val images, {counts['objects']} objects at {args.output}")
    return 0


def prepare_zip_samples(source: Path, samples: list[tuple[str, str]], output: Path, val_ratio: float, seed: int) -> int:
    if not samples:
        raise SystemExit(f"No CubiCasa samples with model.svg and image files found in {source}")
    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val_samples = set(samples[:val_count])

    reset_yolo_dirs(output)
    counts = {"train": 0, "val": 0, "objects": 0}
    with zipfile.ZipFile(source) as archive:
        for image_name, svg_name in samples:
            split = "val" if (image_name, svg_name) in val_samples else "train"
            boxes = extract_door_window_boxes_from_text(archive.read(svg_name).decode("utf-8", errors="replace"))
            if not boxes:
                continue
            image_bytes = archive.read(image_name)
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                suffix = Path(image_name).suffix.lower() or ".png"
                output_stem = safe_stem(Path(image_name).parent)
                target_image = output / "images" / split / f"{output_stem}{suffix}"
                target_label = output / "labels" / split / f"{output_stem}.txt"
                image.convert("RGB").save(target_image)
            rows = to_yolo_rows(boxes, width, height)
            if not rows:
                target_image.unlink(missing_ok=True)
                continue
            target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
            counts[split] += 1
            counts["objects"] += len(rows)

    write_data_yaml(output)
    print(f"Prepared {counts['train']} train images, {counts['val']} val images, {counts['objects']} objects at {output}")
    return 0


def find_samples(source: Path) -> list[tuple[Path, Path]]:
    samples = []
    for svg_path in source.rglob("model.svg"):
        image_path = find_image(svg_path.parent)
        if image_path:
            samples.append((image_path, svg_path))
    return samples


def find_zip_samples(source: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
    by_dir: dict[str, dict[str, str]] = {}
    for name in names:
        path = Path(name)
        folder = path.parent.as_posix()
        if path.name == "model.svg":
            by_dir.setdefault(folder, {})["svg"] = name
        elif path.name in IMAGE_NAMES or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            by_dir.setdefault(folder, {}).setdefault("image", name)
    samples = []
    for item in by_dir.values():
        if "image" in item and "svg" in item:
            samples.append((item["image"], item["svg"]))
    return samples


def find_image(directory: Path) -> Path | None:
    for name in IMAGE_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    images = [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")]
    return images[0] if images else None


def extract_door_window_boxes(svg_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.parse(svg_path).getroot()
    return extract_door_window_boxes_from_root(root)


def extract_door_window_boxes_from_text(svg_text: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.fromstring(svg_text)
    return extract_door_window_boxes_from_root(root)


def extract_door_window_boxes_from_root(root: ET.Element) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes = []
    for element in root.iter():
        label = classify_element(element)
        if label is None:
            continue
        points = collect_points(element)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
            continue
        boxes.append((label, (min(xs), min(ys), max(xs), max(ys))))
    return boxes


def classify_element(element: ET.Element) -> str | None:
    text = " ".join([element.attrib.get("id", ""), element.attrib.get("class", "")]).lower()
    if "door" in text:
        return "door"
    if "window" in text:
        return "window"
    return None


def collect_points(element: ET.Element) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for child in element.iter():
        tag = strip_namespace(child.tag)
        if tag == "polygon" or tag == "polyline":
            points.extend(parse_points(child.attrib.get("points", "")))
        elif tag == "rect":
            points.extend(rect_points(child))
        elif tag == "path":
            points.extend(path_points(child.attrib.get("d", "")))
    return points


def parse_points(value: str) -> list[tuple[float, float]]:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]


def rect_points(element: ET.Element) -> list[tuple[float, float]]:
    x = float(element.attrib.get("x") or 0)
    y = float(element.attrib.get("y") or 0)
    width = float(element.attrib.get("width") or 0)
    height = float(element.attrib.get("height") or 0)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def path_points(value: str) -> list[tuple[float, float]]:
    return parse_points(value)


def to_yolo_rows(boxes: list[tuple[str, tuple[float, float, float, float]]], width: int, height: int) -> list[str]:
    rows = []
    for label, (x1, y1, x2, y2) in boxes:
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 1 or box_h <= 1:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        rows.append(f"{CLASS_TO_ID[label]} {cx:.6f} {cy:.6f} {box_w / width:.6f} {box_h / height:.6f}")
    return rows


def reset_yolo_dirs(output: Path) -> None:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for folder in (output / "images" / split, output / "labels" / split):
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.name != ".gitkeep":
                    file_path.unlink()


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: door",
                "  1: window",
                "",
            ]
        ),
        encoding="utf-8",
    )


def safe_stem(directory: Path) -> str:
    parts = [part for part in directory.parts[-3:] if part not in {"/", "\\"}]
    return "_".join(re.sub(r"[^A-Za-z0-9_-]+", "_", part) for part in parts)


def strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import io
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_TO_ID = {"background": 0, "room": 1, "wall": 2}
IMAGE_NAMES = ("F1_original.png", "F1_scaled.png", "original.png", "image.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert floor-plan SVG labels to room/wall segmentation masks.")
    parser.add_argument("--source", type=Path, required=True, help="Extracted CubiCasa5K root, nested data directory, or CubiCasa ZIP.")
    parser.add_argument("--output", type=Path, default=Path("datasets/room_wall"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-width", type=int, default=8, help="Fallback draw width for line/polyline wall labels.")
    args = parser.parse_args()

    if args.source.suffix.lower() == ".zip":
        samples = find_zip_samples(args.source)
        if args.limit:
            samples = samples[: args.limit]
        return prepare_zip_samples(args.source, samples, args.output, args.val_ratio, args.seed, args.wall_width)

    samples = find_samples(args.source)
    if args.limit:
        samples = samples[: args.limit]
    return prepare_file_samples(samples, args.output, args.val_ratio, args.seed, args.wall_width, args.source)


def prepare_file_samples(
    samples: list[tuple[Path, Path]],
    output: Path,
    val_ratio: float,
    seed: int,
    wall_width: int,
    source: Path,
) -> int:
    if not samples:
        raise SystemExit(f"No samples with model.svg and image files found under {source}")
    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val_samples = set(samples[:val_count])
    reset_segmentation_dirs(output)

    counts = {"train": 0, "val": 0, "room_pixels": 0, "wall_pixels": 0}
    for image_path, svg_path in samples:
        split = "val" if (image_path, svg_path) in val_samples else "train"
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            mask = svg_to_mask(svg_path.read_text(encoding="utf-8", errors="replace"), rgb.size, wall_width)
        if not _mask_has_labels(mask):
            continue
        output_stem = safe_stem(image_path.parent)
        target_image = output / "images" / split / f"{output_stem}.png"
        target_mask = output / "masks" / split / f"{output_stem}.png"
        rgb.save(target_image)
        mask.save(target_mask)
        counts[split] += 1
        counts["room_pixels"] += _count_class(mask, CLASS_TO_ID["room"])
        counts["wall_pixels"] += _count_class(mask, CLASS_TO_ID["wall"])

    write_data_yaml(output)
    write_metadata(output, counts)
    print(f"Prepared {counts['train']} train masks, {counts['val']} val masks at {output}")
    return 0


def prepare_zip_samples(source: Path, samples: list[tuple[str, str]], output: Path, val_ratio: float, seed: int, wall_width: int) -> int:
    if not samples:
        raise SystemExit(f"No samples with model.svg and image files found in {source}")
    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val_samples = set(samples[:val_count])
    reset_segmentation_dirs(output)

    counts = {"train": 0, "val": 0, "room_pixels": 0, "wall_pixels": 0}
    with zipfile.ZipFile(source) as archive:
        for image_name, svg_name in samples:
            split = "val" if (image_name, svg_name) in val_samples else "train"
            image_bytes = archive.read(image_name)
            svg_text = archive.read(svg_name).decode("utf-8", errors="replace")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = image.convert("RGB")
                mask = svg_to_mask(svg_text, rgb.size, wall_width)
            if not _mask_has_labels(mask):
                continue
            output_stem = safe_stem(Path(image_name).parent)
            target_image = output / "images" / split / f"{output_stem}.png"
            target_mask = output / "masks" / split / f"{output_stem}.png"
            rgb.save(target_image)
            mask.save(target_mask)
            counts[split] += 1
            counts["room_pixels"] += _count_class(mask, CLASS_TO_ID["room"])
            counts["wall_pixels"] += _count_class(mask, CLASS_TO_ID["wall"])

    write_data_yaml(output)
    write_metadata(output, counts)
    print(f"Prepared {counts['train']} train masks, {counts['val']} val masks at {output}")
    return 0


def svg_to_mask(svg_text: str, size: tuple[int, int], wall_width: int = 8) -> Image.Image:
    root = ET.fromstring(svg_text)
    mask = Image.new("L", size, CLASS_TO_ID["background"])
    draw = ImageDraw.Draw(mask)
    # Draw rooms first, then walls on top so thin wall labels remain visible.
    for target_type in ("room", "wall"):
        for element in root.iter():
            label = classify_element(element)
            if label != target_type:
                continue
            draw_element(draw, element, CLASS_TO_ID[target_type], wall_width if target_type == "wall" else 1)
    return mask


def draw_element(draw: ImageDraw.ImageDraw, element: ET.Element, value: int, wall_width: int) -> None:
    tag = strip_namespace(element.tag)
    if tag in {"polygon", "polyline"}:
        points = parse_points(element.attrib.get("points", ""))
        if len(points) >= 3 and tag == "polygon":
            draw.polygon(points, fill=value)
        elif len(points) >= 2:
            draw.line(points, fill=value, width=wall_width, joint="curve")
    elif tag == "rect":
        points = rect_points(element)
        draw.polygon(points, fill=value)
    elif tag == "path":
        points = path_points(element.attrib.get("d", ""))
        if len(points) >= 3:
            draw.polygon(points, fill=value)
        elif len(points) >= 2:
            draw.line(points, fill=value, width=wall_width, joint="curve")
    for child in element:
        draw_element(draw, child, value, wall_width)


def classify_element(element: ET.Element) -> str | None:
    text = " ".join([element.attrib.get("id", ""), element.attrib.get("class", ""), element.attrib.get("data-name", "")])
    normalized = text.lower().replace("_", " ").replace("-", " ")
    if any(token in normalized for token in ("wall", "walls", "duvar")):
        return "wall"
    if any(token in normalized for token in ("room", "space", "living", "bedroom", "kitchen", "bath", "toilet", "oda", "salon", "mutfak", "banyo", "wc", "hol")):
        return "room"
    return None


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
    return [(item["image"], item["svg"]) for item in by_dir.values() if "image" in item and "svg" in item]


def find_image(directory: Path) -> Path | None:
    for name in IMAGE_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    images = [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")]
    return images[0] if images else None


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


def reset_segmentation_dirs(output: Path) -> None:
    for split in ("train", "val"):
        for folder in (output / "images" / split, output / "masks" / split):
            folder.mkdir(parents=True, exist_ok=True)
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
                "train_masks: masks/train",
                "val_masks: masks/val",
                "names:",
                "  0: background",
                "  1: room",
                "  2: wall",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_metadata(output: Path, counts: dict[str, int]) -> None:
    (output / "metadata.json").write_text(
        json.dumps({"class_to_id": CLASS_TO_ID, "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_stem(directory: Path) -> str:
    parts = [part for part in directory.parts[-3:] if part not in {"/", "\\"}]
    return "_".join(re.sub(r"[^A-Za-z0-9_-]+", "_", part) for part in parts)


def strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _mask_has_labels(mask: Image.Image) -> bool:
    extrema = mask.getextrema()
    return extrema[1] > 0


def _count_class(mask: Image.Image, class_id: int) -> int:
    histogram = mask.histogram()
    return histogram[class_id] if class_id < len(histogram) else 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.geometry.primitives import DetectionObject, TileInfo


class DebugVisualizer:
    COLORS = {
        "room": (46, 125, 50),
        "wall": (33, 33, 33),
        "door": (198, 40, 40),
        "window": (21, 101, 192),
        "room_text": (123, 31, 162),
        "dimension_text": (245, 124, 0),
    }

    def draw_overlay(self, image_path: Path, output_path: Path, objects: list[DetectionObject]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = ImageFont.load_default()
        for item in objects:
            color = self.COLORS.get(item.type, (0, 0, 0))
            label = f"{item.type} {item.confidence:.2f}" if item.confidence is not None else item.type
            if item.geometry_type == "bbox":
                x1, y1, x2, y2 = item.geometry["x1"], item.geometry["y1"], item.geometry["x2"], item.geometry["y2"]
                draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
                draw.text((x1, max(0, y1 - 12)), label, fill=color + (255,), font=font)
            elif item.geometry_type == "polygon":
                points = [tuple(point) for point in item.geometry.get("points", [])]
                if points:
                    draw.polygon(points, outline=color + (255,), fill=color + (35,))
                    draw.text(points[0], label, fill=color + (255,), font=font)
            elif item.geometry_type == "polyline":
                points = [tuple(point) for point in item.geometry.get("points", [])]
                if len(points) >= 2:
                    draw.line(points, fill=color + (255,), width=4)
                    draw.text(points[0], label, fill=color + (255,), font=font)
        canvas.save(output_path)
        return output_path

    def draw_tiles(self, image_path: Path, output_path: Path, tiles: list[TileInfo]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = ImageFont.load_default()
        for tile in tiles:
            x1, y1 = tile.x_offset, tile.y_offset
            x2, y2 = tile.x_offset + tile.width, tile.y_offset + tile.height
            draw.rectangle((x1, y1, x2, y2), outline=(245, 124, 0, 220), width=2)
            draw.text((x1 + 4, y1 + 4), tile.id, fill=(245, 124, 0, 255), font=font)
        canvas.save(output_path)
        return output_path


from pathlib import Path

from PIL import Image

from src.geometry.primitives import TileInfo


class ImageTiler:
    def create_tiles(
        self,
        image_path: Path,
        output_dir: Path,
        tile_size: int = 1280,
        overlap: float = 0.25,
    ) -> list[TileInfo]:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            width, height = image.size
            step = max(1, int(tile_size * (1.0 - overlap)))
            xs = _axis_offsets(width, tile_size, step)
            ys = _axis_offsets(height, tile_size, step)
            tiles: list[TileInfo] = []
            index = 0
            for y in ys:
                for x in xs:
                    right = min(x + tile_size, width)
                    bottom = min(y + tile_size, height)
                    tile = image.crop((x, y, right, bottom))
                    tile_path = output_dir / f"tile_{index:04d}.png"
                    tile.save(tile_path)
                    tiles.append(
                        TileInfo(
                            id=f"tile_{index:04d}",
                            image_path=str(tile_path),
                            x_offset=x,
                            y_offset=y,
                            width=right - x,
                            height=bottom - y,
                        )
                    )
                    index += 1
        return tiles


def _axis_offsets(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    offsets = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


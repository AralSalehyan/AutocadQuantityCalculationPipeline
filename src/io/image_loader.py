from pathlib import Path

from PIL import Image


class ImageLoader:
    def load(self, input_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_path) as image:
            image.convert("RGB").save(output_path)
        return output_path


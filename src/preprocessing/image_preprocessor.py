from pathlib import Path

from PIL import Image, ImageOps


class ImagePreprocessor:
    def preprocess(self, image_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            processed = ImageOps.autocontrast(image.convert("RGB"))
            processed.save(output_path)
        return output_path


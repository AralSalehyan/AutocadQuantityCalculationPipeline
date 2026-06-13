import json
from pathlib import Path
from typing import Any

from src.geometry.primitives import to_plain


class JSONExporter:
    def write(self, output_path: Path, data: Any) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(to_plain(data), handle, indent=2, ensure_ascii=False)
        return output_path


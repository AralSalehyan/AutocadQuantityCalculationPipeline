from pathlib import Path


def detect_file_type(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".dxf":
        return "dxf"
    if suffix == ".dwg":
        raise ValueError("DWG is not supported directly in this prototype. Please convert DWG to DXF first.")
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "unsupported"


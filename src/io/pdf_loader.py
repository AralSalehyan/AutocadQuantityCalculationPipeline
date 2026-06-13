from pathlib import Path

from src.geometry.primitives import DetectionObject, VectorPrimitive
from src.utils.ids import new_id


class PDFLoader:
    def render_first_page(self, input_path: Path, output_path: Path, dpi: int = 300) -> Path:
        fitz = _fitz()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(input_path) as document:
            if document.page_count == 0:
                raise ValueError(f"PDF has no pages: {input_path}")
            page = document.load_page(0)
            zoom = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.save(output_path)
        return output_path

    def extract_text_blocks(self, input_path: Path) -> list[DetectionObject]:
        fitz = _fitz()
        objects: list[DetectionObject] = []
        with fitz.open(input_path) as document:
            if document.page_count == 0:
                return []
            page = document.load_page(0)
            for block in page.get_text("blocks"):
                x1, y1, x2, y2, text, *_ = block
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                objects.append(
                    DetectionObject(
                        id=new_id("text"),
                        type=_classify_text(cleaned),
                        source="pdf_vector",
                        geometry_type="bbox",
                        geometry={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        label=cleaned,
                        confidence=0.9,
                        metadata={"page": 0},
                    )
                )
        return objects

    def extract_vector_primitives(self, input_path: Path) -> list[VectorPrimitive]:
        fitz = _fitz()
        primitives: list[VectorPrimitive] = []
        with fitz.open(input_path) as document:
            if document.page_count == 0:
                return []
            page = document.load_page(0)
            for drawing in page.get_drawings():
                layer = drawing.get("layer") or None
                for item in drawing.get("items", []):
                    primitive = _drawing_item_to_primitive(item, layer)
                    if primitive is not None:
                        primitives.append(primitive)
        return primitives


def _drawing_item_to_primitive(item, layer: str | None) -> VectorPrimitive | None:
    op = item[0]
    if op == "l":
        start, end = item[1], item[2]
        return VectorPrimitive(
            id=new_id("pdf_line"),
            type="line",
            points=[[float(start.x), float(start.y)], [float(end.x), float(end.y)]],
            layer=layer,
            raw={"operator": op},
        )
    if op == "re":
        rect = item[1]
        return VectorPrimitive(
            id=new_id("pdf_polygon"),
            type="polygon",
            points=[
                [float(rect.x0), float(rect.y0)],
                [float(rect.x1), float(rect.y0)],
                [float(rect.x1), float(rect.y1)],
                [float(rect.x0), float(rect.y1)],
            ],
            layer=layer,
            raw={"operator": op},
        )
    if op == "qu":
        quad = item[1]
        points = [[float(point.x), float(point.y)] for point in [quad.ul, quad.ur, quad.lr, quad.ll]]
        return VectorPrimitive(id=new_id("pdf_polygon"), type="polygon", points=points, layer=layer, raw={"operator": op})
    if op == "c":
        points = [[float(point.x), float(point.y)] for point in item[1:]]
        return VectorPrimitive(id=new_id("pdf_polyline"), type="polyline", points=points, layer=layer, raw={"operator": op})
    return None


def _classify_text(text: str) -> str:
    lowered = text.lower()
    if any(unit in lowered for unit in ("m2", "m²", "m^2", "room", "oda", "salon", "mutfak", "banyo", "wc", "hol")):
        return "room_text"
    if any(unit in lowered for unit in ("cm", "mm", " m")):
        return "dimension_text"
    return "room_text"


def _fitz():
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF support. Install it with `python -m pip install PyMuPDF`.") from exc
    return fitz


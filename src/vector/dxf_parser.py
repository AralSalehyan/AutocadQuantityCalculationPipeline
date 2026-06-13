from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from PIL import Image, ImageDraw, ImageFont

from src.geometry.primitives import VectorPrimitive
from src.utils.ids import new_id


class DXFParser:
    def parse(self, input_path: Path) -> list[VectorPrimitive]:
        ezdxf = _ezdxf()
        document = ezdxf.readfile(input_path)
        primitives: list[VectorPrimitive] = []
        for entity in document.modelspace():
            primitive = _entity_to_primitive(entity)
            if primitive is not None:
                primitives.append(primitive)
        return primitives

    def render_preview(self, input_path: Path, output_path: Path) -> Path:
        primitives = self.parse(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = _preview_size(primitives)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        transform = _coordinate_transform(primitives, width, height)

        for primitive in primitives:
            color = _color_for_primitive(primitive)
            points = [transform(point) for point in primitive.points]
            if primitive.type in {"line", "polyline", "polygon"} and len(points) >= 2:
                draw.line(points + ([points[0]] if primitive.type == "polygon" else []), fill=color, width=4 if _is_wall_layer(primitive.layer) else 2)
            elif primitive.type == "block" and points:
                x, y = points[0]
                size = 24
                draw.rectangle((x - size, y - size, x + size, y + size), outline=color, width=3)
                draw.text((x + size + 3, y - size), primitive.label or "BLOCK", fill=color, font=font)
            elif primitive.type == "text" and points:
                draw.text(tuple(points[0]), primitive.label or "", fill=color, font=font)
            elif primitive.type == "circle" and points:
                x, y = points[0]
                radius = float((primitive.raw or {}).get("radius") or 10.0)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        image.save(output_path)
        return output_path


def _entity_to_primitive(entity) -> VectorPrimitive | None:
    kind = entity.dxftype()
    layer = getattr(entity.dxf, "layer", None)
    if kind == "LINE":
        return VectorPrimitive(
            id=new_id("dxf_line"),
            type="line",
            points=[_xy(entity.dxf.start), _xy(entity.dxf.end)],
            layer=layer,
            raw={"entity": kind},
        )
    if kind == "LWPOLYLINE":
        points = [[float(x), float(y)] for x, y, *_ in entity.get_points()]
        return VectorPrimitive(
            id=new_id("dxf_polyline"),
            type="polygon" if entity.closed else "polyline",
            points=points,
            layer=layer,
            raw={"entity": kind, "closed": bool(entity.closed)},
        )
    if kind == "POLYLINE":
        points = [_xy(vertex.dxf.location) for vertex in entity.vertices]
        return VectorPrimitive(
            id=new_id("dxf_polyline"),
            type="polygon" if entity.is_closed else "polyline",
            points=points,
            layer=layer,
            raw={"entity": kind, "closed": bool(entity.is_closed)},
        )
    if kind == "INSERT":
        return VectorPrimitive(
            id=new_id("dxf_block"),
            type="block",
            points=[_xy(entity.dxf.insert)],
            layer=layer,
            label=str(entity.dxf.name),
            raw={"entity": kind, "name": str(entity.dxf.name), "xscale": float(entity.dxf.xscale), "yscale": float(entity.dxf.yscale)},
        )
    if kind in {"TEXT", "MTEXT"}:
        text = entity.dxf.text if kind == "TEXT" else entity.text
        return VectorPrimitive(
            id=new_id("dxf_text"),
            type="text",
            points=[_xy(entity.dxf.insert)],
            layer=layer,
            label=str(text).strip(),
            raw={"entity": kind},
        )
    if kind == "CIRCLE":
        return VectorPrimitive(
            id=new_id("dxf_circle"),
            type="circle",
            points=[_xy(entity.dxf.center)],
            layer=layer,
            raw={"entity": kind, "radius": float(entity.dxf.radius)},
        )
    if kind == "ARC":
        return VectorPrimitive(
            id=new_id("dxf_arc"),
            type="arc",
            points=[_xy(entity.dxf.center)],
            layer=layer,
            raw={"entity": kind, "radius": float(entity.dxf.radius), "start_angle": float(entity.dxf.start_angle), "end_angle": float(entity.dxf.end_angle)},
        )
    if kind == "DIMENSION":
        point = getattr(entity.dxf, "defpoint", None)
        return VectorPrimitive(
            id=new_id("dxf_dimension"),
            type="dimension",
            points=[_xy(point)] if point is not None else [],
            layer=layer,
            raw={"entity": kind},
        )
    return None


def _xy(value) -> list[float]:
    return [float(value[0]), float(value[1])]


def _preview_size(primitives: list[VectorPrimitive]) -> tuple[int, int]:
    points = [point for primitive in primitives for point in primitive.points]
    if not points:
        return 1400, 900
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    if 0 <= min(point[0] for point in points) and 0 <= min(point[1] for point in points) and max_x <= 2400 and max_y <= 1800:
        return max(800, int(max_x + 120)), max(600, int(max_y + 120))
    return 1400, 900


def _coordinate_transform(primitives: list[VectorPrimitive], width: int, height: int):
    points = [point for primitive in primitives for point in primitive.points]
    if not points:
        return lambda point: (point[0], point[1])
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    if min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height:
        return lambda point: (point[0], point[1])
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((width - 80) / span_x, (height - 80) / span_y)
    return lambda point: ((point[0] - min_x) * scale + 40, (point[1] - min_y) * scale + 40)


def _color_for_primitive(primitive: VectorPrimitive) -> tuple[int, int, int]:
    layer = (primitive.layer or "").lower()
    label = (primitive.label or "").lower()
    if _is_wall_layer(primitive.layer):
        return (30, 30, 30)
    if "door" in label or "kapi" in label:
        return (180, 40, 40)
    if "window" in label or "pencere" in label:
        return (40, 100, 190)
    if "text" in layer:
        return (80, 80, 80)
    return (90, 90, 90)


def _is_wall_layer(layer: str | None) -> bool:
    return bool(layer) and any(token in layer.lower() for token in ("wall", "duvar", "a-wall", "a_walls"))


def _ezdxf():
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            import ezdxf
    except ImportError as exc:
        raise RuntimeError("ezdxf is required for DXF support. Install it with `python -m pip install ezdxf`.") from exc
    return ezdxf

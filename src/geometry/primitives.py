from typing import Any

from pydantic import BaseModel, Field


class DetectionObject(BaseModel):
    id: str
    type: str
    source: str
    geometry_type: str
    geometry: dict[str, Any]
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


class QuantityItem(BaseModel):
    id: str
    category: str
    name: str
    quantity: float
    unit: str
    source_object_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    calculation: dict[str, Any] | None = None


class VectorPrimitive(BaseModel):
    id: str
    type: str
    points: list[list[float]]
    layer: str | None = None
    label: str | None = None
    raw: dict[str, Any] | None = None


class TileInfo(BaseModel):
    id: str
    image_path: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    scale: float = 1.0


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value.dict()
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return value


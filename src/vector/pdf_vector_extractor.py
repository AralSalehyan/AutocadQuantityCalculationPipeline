from pathlib import Path

from src.geometry.primitives import VectorPrimitive
from src.io.pdf_loader import PDFLoader


class PDFVectorExtractor:
    def extract(self, input_path: Path) -> list[VectorPrimitive]:
        return PDFLoader().extract_vector_primitives(input_path)


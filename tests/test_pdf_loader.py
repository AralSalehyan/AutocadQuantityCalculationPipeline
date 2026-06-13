from pathlib import Path

from run_pipeline import create_synthetic_pdf
from src.io.pdf_loader import PDFLoader


def test_pdf_loader_renders_and_extracts_vectors(tmp_path: Path) -> None:
    pdf_path = tmp_path / "example.pdf"
    rendered_path = tmp_path / "rendered.png"
    create_synthetic_pdf(pdf_path)

    loader = PDFLoader()
    loader.render_first_page(pdf_path, rendered_path, dpi=150)
    primitives = loader.extract_vector_primitives(pdf_path)
    text_blocks = loader.extract_text_blocks(pdf_path)

    assert rendered_path.exists()
    assert primitives
    assert any(item.type == "room_text" for item in text_blocks)


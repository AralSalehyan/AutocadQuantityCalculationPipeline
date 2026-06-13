from pathlib import Path

from run_pipeline import create_synthetic_dxf
from src.vector.block_symbol_extractor import BlockSymbolExtractor
from src.vector.dxf_parser import DXFParser
from src.vector.wall_candidate_extractor import WallCandidateExtractor


def test_dxf_parser_extracts_primitives_and_vector_objects(tmp_path: Path) -> None:
    dxf_path = tmp_path / "example.dxf"
    preview_path = tmp_path / "rendered.png"
    create_synthetic_dxf(dxf_path)

    parser = DXFParser()
    primitives = parser.parse(dxf_path)
    parser.render_preview(dxf_path, preview_path)
    walls = WallCandidateExtractor().extract(primitives)
    symbols = BlockSymbolExtractor().extract(primitives)

    assert preview_path.exists()
    assert len(primitives) >= 10
    assert len(walls) >= 6
    assert {item.type for item in symbols} == {"door", "window"}

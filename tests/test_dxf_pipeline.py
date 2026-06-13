import subprocess
import sys
from pathlib import Path


def test_dxf_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "dxf_test"
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--input",
            "sample_data/dxf/example.dxf",
            "--output",
            str(output_dir),
            "--detectors",
            "mock",
            "--segmenter",
            "mock",
            "--scale",
            "0.01",
            "--debug",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "rendered.png").exists()
    assert (output_dir / "vector_primitives.json").exists()
    assert (output_dir / "drawing_graph.json").exists()
    report = (output_dir / "processing_report.md").read_text(encoding="utf-8")
    assert "File type: `dxf`" in report
    assert "dxf_vector_objects" in report

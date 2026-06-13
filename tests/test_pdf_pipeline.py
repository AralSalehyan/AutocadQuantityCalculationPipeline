import subprocess
import sys
from pathlib import Path


def test_pdf_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdf_test"
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--input",
            "sample_data/pdf/example.pdf",
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
    assert (output_dir / "processing_report.md").exists()
    report = (output_dir / "processing_report.md").read_text(encoding="utf-8")
    assert "File type: `pdf`" in report


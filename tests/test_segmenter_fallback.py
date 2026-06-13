import subprocess
import sys
from pathlib import Path


def test_segformer_missing_model_falls_back_to_mock(tmp_path: Path) -> None:
    output_dir = tmp_path / "segmentation_test"
    config_path = tmp_path / "missing_segformer.yaml"
    config_path.write_text(
        "\n".join(
            [
                "segformer:",
                f"  model_path: {str(tmp_path / 'missing_model').replace(chr(92), '/')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--input",
            "sample_data/images/example.png",
            "--output",
            str(output_dir),
            "--detectors",
            "mock",
            "--segmenter",
            "segformer",
            "--scale",
            "0.01",
            "--debug",
            "--config",
            str(config_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = (output_dir / "processing_report.md").read_text(encoding="utf-8")
    assert "using mock segmenter fallback" in report
    assert (output_dir / "segmentation_objects.json").exists()

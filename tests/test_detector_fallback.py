import json
import subprocess
import sys
from pathlib import Path


def test_yolo_missing_model_falls_back_to_mock(tmp_path: Path) -> None:
    output_dir = tmp_path / "yolo_test"
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--input",
            "sample_data/images/example.png",
            "--output",
            str(output_dir),
            "--detectors",
            "yolo",
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
    quantities = json.loads((output_dir / "quantities.json").read_text(encoding="utf-8"))
    assert any(item["category"] == "door_count" for item in quantities)
    report = (output_dir / "processing_report.md").read_text(encoding="utf-8")
    if Path("models/yolo/door_window.pt").exists():
        assert "using mock detector fallback" not in report
        assert "yolo_tile_detections" in report
    else:
        assert "using mock detector fallback" in report


def test_rtdetr_missing_model_falls_back_to_mock(tmp_path: Path) -> None:
    output_dir = tmp_path / "rtdetr_test"
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--input",
            "sample_data/images/example.png",
            "--output",
            str(output_dir),
            "--detectors",
            "rtdetr",
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
    report = (output_dir / "processing_report.md").read_text(encoding="utf-8")
    assert "Detector `rtdetr`" not in report
    if Path("models/rtdetr/door_window.pt").exists():
        assert "using mock detector fallback" not in report
        assert "rtdetr_tile_detections" in report
    else:
        assert "using mock detector fallback" in report

import json
import subprocess
import sys
from pathlib import Path


def test_phase1_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "first_test"
    command = [
        sys.executable,
        "run_pipeline.py",
        "--input",
        "sample_data/images/example.png",
        "--output",
        str(output_dir),
        "--detectors",
        "mock",
        "--segmenter",
        "mock",
        "--scale",
        "0.01",
        "--debug",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    expected_files = [
        "rendered.png",
        "preprocessed.png",
        "raster_detections.json",
        "segmentation_objects.json",
        "merged_objects.json",
        "quantities.json",
        "quantities.xlsx",
        "debug_overlay.png",
        "processing_report.md",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists(), filename
    assert any((output_dir / "tiles").glob("tile_*.png"))

    quantities = json.loads((output_dir / "quantities.json").read_text(encoding="utf-8"))
    categories = {item["category"] for item in quantities}
    assert {"room_area", "room_perimeter", "wall_length_total", "door_count", "window_count"} <= categories


import json
import subprocess
import sys
from pathlib import Path

from evaluate_outputs import evaluate


def test_evaluate_perfect_bbox_and_quantity_match() -> None:
    objects = [
        _bbox("door_1", "door", 0, 0, 10, 10),
        _bbox("window_1", "window", 20, 20, 40, 40),
    ]
    quantities = [
        {"id": "q1", "category": "door_count", "name": "Door count", "quantity": 1.0, "unit": "count"},
        {"id": "q2", "category": "window_count", "name": "Window count", "quantity": 1.0, "unit": "count"},
    ]

    report = evaluate(objects, objects, quantities, quantities)

    assert report["detection"]["overall"]["precision"] == 1.0
    assert report["detection"]["overall"]["recall"] == 1.0
    assert report["detection"]["overall"]["f1"] == 1.0
    assert all(item["absolute_error"] == 0.0 for item in report["quantities"]["items"])


def test_evaluate_counts_false_positive_and_false_negative() -> None:
    pred = [_bbox("door_pred", "door", 0, 0, 10, 10), _bbox("window_extra", "window", 50, 50, 70, 70)]
    gt = [_bbox("door_gt", "door", 0, 0, 10, 10), _bbox("window_gt", "window", 100, 100, 120, 120)]

    report = evaluate(pred, gt, [], [])

    assert report["detection"]["overall"]["true_positive"] == 1
    assert report["detection"]["overall"]["false_positive"] == 1
    assert report["detection"]["overall"]["false_negative"] == 1
    assert report["detection"]["by_class"]["window"]["count_error"] == 0
    assert report["detection"]["by_class"]["window"]["f1"] == 0.0


def test_evaluate_outputs_cli_writes_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    result = subprocess.run(
        [
            sys.executable,
            "evaluate_outputs.py",
            "--pred",
            "outputs/first_test/merged_objects.json",
            "--gt",
            "sample_data/ground_truth/example_objects.json",
            "--output",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["detection"]["overall"]["f1"] == 1.0
    assert (output_dir / "evaluation_report.md").exists()


def _bbox(object_id: str, object_type: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "id": object_id,
        "type": object_type,
        "source": "test",
        "geometry_type": "bbox",
        "geometry": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": 1.0,
    }

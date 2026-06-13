from pathlib import Path

from PIL import Image

from src.training.prepare_detection_dataset import main


def test_prepare_detection_dataset_from_cubicasa_like_svg(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "raw" / "sample_001"
    source.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(source / "F1_original.png")
    (source / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="Door"><polygon points="10,20 40,20 40,60 10,60"/></g>
          <g id="Window"><rect x="100" y="10" width="50" height="20"/></g>
        </svg>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "door_window"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_detection_dataset.py",
            "--source",
            str(tmp_path / "raw"),
            "--output",
            str(output),
        ],
    )

    assert main() == 0

    labels = [*output.glob("labels/**/*.txt")]
    assert len(labels) == 1
    rows = labels[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith("0 ")
    assert rows[1].startswith("1 ")
    assert (output / "data.yaml").exists()


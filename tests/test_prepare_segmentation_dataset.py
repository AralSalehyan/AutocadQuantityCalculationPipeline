from pathlib import Path

from PIL import Image

from src.training.prepare_segmentation_dataset import CLASS_TO_ID, main, svg_to_mask


def test_svg_to_room_wall_mask() -> None:
    svg_text = """
    <svg xmlns="http://www.w3.org/2000/svg">
      <g id="Room"><polygon points="10,10 90,10 90,90 10,90"/></g>
      <g id="Wall"><polyline points="0,50 100,50"/></g>
    </svg>
    """

    mask = svg_to_mask(svg_text, (120, 120), wall_width=6)

    assert mask.getpixel((20, 20)) == CLASS_TO_ID["room"]
    assert mask.getpixel((50, 50)) == CLASS_TO_ID["wall"]
    assert mask.getpixel((110, 110)) == CLASS_TO_ID["background"]


def test_prepare_segmentation_dataset_from_cubicasa_like_svg(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "raw" / "sample_001"
    source.mkdir(parents=True)
    Image.new("RGB", (120, 120), "white").save(source / "F1_original.png")
    (source / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="Space Room"><polygon points="10,10 90,10 90,90 10,90"/></g>
          <g id="Wall"><rect x="0" y="48" width="120" height="8"/></g>
        </svg>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "room_wall"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_segmentation_dataset.py",
            "--source",
            str(tmp_path / "raw"),
            "--output",
            str(output),
        ],
    )

    assert main() == 0

    masks = [*output.glob("masks/**/*.png")]
    assert len(masks) == 1
    with Image.open(masks[0]) as mask:
        assert mask.getpixel((20, 20)) == CLASS_TO_ID["room"]
        assert mask.getpixel((20, 50)) == CLASS_TO_ID["wall"]
    assert (output / "data.yaml").exists()
    assert (output / "metadata.json").exists()

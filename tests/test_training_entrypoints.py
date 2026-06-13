from pathlib import Path

from src.training.train_segformer import read_data_yaml


def test_segformer_reads_room_wall_data_yaml(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images/train",
                "val: images/val",
                "train_masks: masks/train",
                "val_masks: masks/val",
                "names:",
                "  0: background",
                "  1: room",
                "  2: wall",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = read_data_yaml(data_yaml)

    assert config["train_masks"] == "masks/train"
    assert config["names"][1] == "room"

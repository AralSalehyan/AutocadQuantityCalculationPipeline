import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from src.pipeline.context import PipelineContext
from src.pipeline.pipeline import CADQuantityPipeline
from src.utils.config import load_pipeline_config
from src.utils.logger import configure_logger
from src.utils.runtime_paths import configure_workspace_runtime_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CAD quantity calculation pipeline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detectors", default="mock", help="Comma-separated detector names.")
    parser.add_argument("--segmenter", default="mock")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    configure_workspace_runtime_dirs()
    logger = configure_logger(args.debug)
    if _should_create_sample(args.input):
        create_synthetic_floorplan(args.input)
        logger.info("Created synthetic sample image at %s", args.input)
    if _should_create_sample_pdf(args.input):
        create_synthetic_pdf(args.input)
        logger.info("Created synthetic sample PDF at %s", args.input)
    if _should_create_sample_dxf(args.input):
        create_synthetic_dxf(args.input)
        logger.info("Created synthetic sample DXF at %s", args.input)

    if not args.input.exists():
        logger.error("Input file does not exist: %s", args.input)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    context = PipelineContext(
        input_path=args.input,
        output_dir=args.output,
        scale_ratio=args.scale,
        config=load_pipeline_config(args.config),
        detector_names=[item.strip() for item in args.detectors.split(",") if item.strip()],
        segmenter_name=args.segmenter,
        debug=args.debug,
    )
    result = CADQuantityPipeline(logger).run(context)
    if result.errors:
        for error in result.errors:
            logger.error(error)
        return 1
    logger.info("Pipeline complete: %s", args.output)
    return 0


def _should_create_sample(input_path: Path) -> bool:
    normalized = input_path.as_posix().lower()
    return normalized.endswith("sample_data/images/example.png") and not input_path.exists()


def _should_create_sample_pdf(input_path: Path) -> bool:
    normalized = input_path.as_posix().lower()
    return normalized.endswith("sample_data/pdf/example.pdf") and not input_path.exists()


def _should_create_sample_dxf(input_path: Path) -> bool:
    normalized = input_path.as_posix().lower()
    return normalized.endswith("sample_data/dxf/example.dxf") and not input_path.exists()


def create_synthetic_floorplan(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    wall = (30, 30, 30)
    thin = (80, 80, 80)
    draw.rectangle((140, 125, 1260, 720), outline=wall, width=12)
    draw.line((700, 125, 700, 720), fill=wall, width=10)
    draw.line((140, 430, 700, 430), fill=wall, width=8)
    draw.rectangle((650, 242, 760, 297), fill="white", outline=(180, 40, 40), width=4)
    draw.arc((655, 245, 770, 360), start=180, end=270, fill=(180, 40, 40), width=3)
    draw.rectangle((250, 558, 350, 612), fill="white", outline=(180, 40, 40), width=4)
    draw.arc((250, 500, 365, 615), start=90, end=180, fill=(180, 40, 40), width=3)
    draw.rectangle((980, 82, 1175, 128), fill="white", outline=(40, 100, 190), width=4)
    draw.line((980, 105, 1175, 105), fill=(40, 100, 190), width=2)
    draw.rectangle((70, 370, 145, 500), fill="white", outline=(40, 100, 190), width=4)
    draw.line((108, 370, 108, 500), fill=(40, 100, 190), width=2)
    draw.text((310, 260), "ROOM 1", fill=thin)
    draw.text((900, 260), "ROOM 2", fill=thin)
    draw.text((310, 470), "ROOM 3", fill=thin)
    image.save(output_path)


def create_synthetic_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to create the synthetic sample PDF.") from exc

    document = fitz.open()
    page = document.new_page(width=840, height=540)
    wall = (0.1, 0.1, 0.1)
    door = (0.7, 0.1, 0.1)
    window = (0.1, 0.35, 0.75)
    page.draw_rect(fitz.Rect(84, 75, 756, 432), color=wall, width=4)
    page.draw_line(fitz.Point(420, 75), fitz.Point(420, 432), color=wall, width=3)
    page.draw_line(fitz.Point(84, 258), fitz.Point(420, 258), color=wall, width=3)
    page.draw_rect(fitz.Rect(390, 145, 456, 178), color=door, width=2)
    page.draw_rect(fitz.Rect(150, 335, 210, 368), color=door, width=2)
    page.draw_rect(fitz.Rect(588, 50, 705, 77), color=window, width=2)
    page.draw_rect(fitz.Rect(42, 222, 87, 300), color=window, width=2)
    page.insert_text(fitz.Point(185, 155), "ROOM 1", fontsize=16, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(540, 155), "ROOM 2", fontsize=16, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(185, 290), "ROOM 3", fontsize=16, color=(0.2, 0.2, 0.2))
    document.save(output_path)
    document.close()


def create_synthetic_dxf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("ezdxf is required to create the synthetic sample DXF.") from exc

    document = ezdxf.new("R2010")
    for layer, color in {
        "A-WALL": 7,
        "A-DOOR": 1,
        "A-WINDOW": 5,
        "A-TEXT": 8,
    }.items():
        if layer not in document.layers:
            document.layers.add(layer, color=color)

    door_block = document.blocks.new("KAPI_DOOR")
    door_block.add_lwpolyline([(-40, -18), (40, -18), (40, 18), (-40, 18), (-40, -18)], dxfattribs={"layer": "A-DOOR"})
    window_block = document.blocks.new("PENCERE_WINDOW")
    window_block.add_lwpolyline([(-55, -12), (55, -12), (55, 12), (-55, 12), (-55, -12)], dxfattribs={"layer": "A-WINDOW"})

    msp = document.modelspace()
    walls = [
        ((140, 125), (1260, 125)),
        ((1260, 125), (1260, 720)),
        ((1260, 720), (140, 720)),
        ((140, 720), (140, 125)),
        ((700, 125), (700, 720)),
        ((140, 430), (700, 430)),
    ]
    for start, end in walls:
        msp.add_line(start, end, dxfattribs={"layer": "A-WALL"})
    msp.add_blockref("KAPI_DOOR", (705, 270), dxfattribs={"layer": "A-DOOR"})
    msp.add_blockref("KAPI_DOOR", (300, 585), dxfattribs={"layer": "A-DOOR"})
    msp.add_blockref("PENCERE_WINDOW", (1080, 105), dxfattribs={"layer": "A-WINDOW"})
    msp.add_blockref("PENCERE_WINDOW", (108, 435), dxfattribs={"layer": "A-WINDOW"})
    msp.add_text("ROOM 1", dxfattribs={"layer": "A-TEXT", "height": 24}).set_placement((310, 260))
    msp.add_text("ROOM 2", dxfattribs={"layer": "A-TEXT", "height": 24}).set_placement((900, 260))
    msp.add_text("ROOM 3", dxfattribs={"layer": "A-TEXT", "height": 24}).set_placement((310, 470))
    document.saveas(output_path)


if __name__ == "__main__":
    sys.exit(main())

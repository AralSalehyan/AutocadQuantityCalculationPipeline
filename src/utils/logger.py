import logging
import warnings


def configure_logger(debug: bool = False) -> logging.Logger:
    warnings.filterwarnings("ignore", message=".*TripleDES.*")
    warnings.filterwarnings("ignore", message=".*Blowfish.*")
    warnings.filterwarnings("ignore", message=".*use_fast.*")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy_logger in ("PIL", "fontTools", "matplotlib", "ezdxf", "numexpr", "paramiko"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return logging.getLogger("cad_quantity_pipeline")

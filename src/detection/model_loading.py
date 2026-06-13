from pathlib import Path


class ModelUnavailableError(RuntimeError):
    pass


def require_model_file(model_path: Path) -> None:
    if not model_path.exists():
        raise ModelUnavailableError(f"Model file does not exist: {model_path}")


import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path


CUBICASA_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
CUBICASA_MD5 = "0ce0b203d1e3c125b51087b219bd23b9"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public floor-plan datasets.")
    parser.add_argument("--dataset", choices=["cubicasa5k"], default="cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "cubicasa5k.zip"
    if archive_path.exists() and args.skip_existing:
        print(f"Archive already exists: {archive_path}")
    else:
        download_file(CUBICASA_URL, archive_path)

    digest = md5sum(archive_path)
    if digest != CUBICASA_MD5:
        print(f"MD5 mismatch for {archive_path}: expected {CUBICASA_MD5}, got {digest}", file=sys.stderr)
        return 2
    print(f"MD5 verified: {archive_path}")

    if args.extract:
        extract_dir = args.output_dir / "cubicasa5k"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
    return 0


def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading {url}")
    print(f"Output: {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r{downloaded / (1024 ** 3):.2f} / {total / (1024 ** 3):.2f} GB ({pct:.1f}%)", end="")
        print()


def md5sum(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())


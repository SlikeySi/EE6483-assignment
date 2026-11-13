from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from ee6483.utils import ensure_new_file


def verify_directory(directory: Path) -> dict:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    corrupt: list[str] = []
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                modes[image.mode] += 1
                sizes[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError, SyntaxError) as exc:
            corrupt.append(f"{path}: {exc}")
    return {
        "count": len(files),
        "corrupt": corrupt,
        "modes": dict(modes),
        "unique_sizes": len(sizes),
        "manifest_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/logs/dataset_verification.json")
    )
    args = parser.parse_args()
    ensure_new_file(args.output)
    result = {
        name: verify_directory(args.data_dir / name)
        for name in ["train/cat", "train/dog", "val/cat", "val/dog", "test"]
    }
    expected = {
        "train/cat": 10000,
        "train/dog": 10000,
        "val/cat": 2500,
        "val/dog": 2500,
        "test": 500,
    }
    problems = []
    for name, expected_count in expected.items():
        if result[name]["count"] != expected_count:
            problems.append(f"{name}: expected {expected_count}, found {result[name]['count']}")
        if result[name]["corrupt"]:
            problems.append(f"{name}: {len(result[name]['corrupt'])} corrupt images")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: {"count": v["count"], "corrupt": len(v["corrupt"])} for k, v in result.items()},
            indent=2,
        )
    )
    if problems:
        raise SystemExit("Dataset verification failed: " + "; ".join(problems))
    print(f"Dataset verification passed; report: {args.output}")


if __name__ == "__main__":
    main()

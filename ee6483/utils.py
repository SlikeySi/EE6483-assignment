from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and prefer deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def numeric_image_id(path: str | Path) -> int:
    """Return numeric filename stem, raising a useful error otherwise."""
    stem = Path(path).stem
    if not stem.isdigit():
        raise ValueError(f"Test image filename must be a numeric ID: {path}")
    return int(stem)


def sorted_test_images(directory: str | Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(paths, key=numeric_image_id)


def ensure_new_file(path: str | Path) -> Path:
    """Return a new output path, refusing to overwrite any existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_run_directory(path: str | Path) -> Path:
    """Create a fresh experiment directory so runs cannot overwrite one another."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Run directory already exists; choose a new --output-dir: {path}")
    path.mkdir(parents=True)
    return path


def serializable(value: Any) -> Any:
    """Convert paths and nested argparse-style values to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_csv_rows(rows: list[dict[str, Any]], output: str | Path) -> None:
    """Atomically persist accumulated epoch rows after every completed epoch."""
    if not rows:
        raise ValueError("At least one CSV row is required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def validate_submission(
    rows: Iterable[tuple[int, int]], expected_ids: Iterable[int] | None = None
) -> list[tuple[int, int]]:
    validated = [(int(image_id), int(label)) for image_id, label in rows]
    ids = [image_id for image_id, _ in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Submission contains duplicate image IDs")
    expected = set(expected_ids) if expected_ids is not None else None
    if expected is not None and set(ids) != expected:
        raise ValueError(
            "Submission IDs do not match the expected test set; "
            f"missing={sorted(expected - set(ids))[:10]}, extra={sorted(set(ids) - expected)[:10]}"
        )
    for _, label in validated:
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 (cat) or 1 (dog), got {label}")
    return sorted(validated)


def write_submission(
    rows: Iterable[tuple[int, int]],
    output: str | Path,
    expected_ids: Iterable[int] | None = None,
) -> None:
    validated = validate_submission(rows, expected_ids)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for image_id, label in validated:
            writer.writerow([image_id, label])


def save_json(data: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable(data), indent=2, sort_keys=True), encoding="utf-8")

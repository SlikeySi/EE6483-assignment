from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ee6483.data import UnlabelledImageDataset, image_transforms
from ee6483.models import build_model
from ee6483.utils import ensure_new_file, select_device, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission.csv; 1=dog, 0=cat")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_new_file(args.output)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    _, transform = image_transforms(
        checkpoint["image_size"], checkpoint.get("augmentation", "standard")
    )
    dataset = UnlabelledImageDataset(args.data_dir / "test", transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    rows: list[tuple[int, int]] = []
    with torch.inference_mode():
        for images, image_ids, _ in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            rows.extend(
                (int(image_id), int(label)) for image_id, label in zip(image_ids, predictions)
            )
    write_submission(rows, args.output, expected_ids=range(1, 501))
    print(f"Wrote {len(rows)} predictions to {args.output} (1=dog, 0=cat)")


if __name__ == "__main__":
    main()

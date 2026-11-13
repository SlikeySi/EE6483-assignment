from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from ee6483.data import IMAGENET_MEAN, IMAGENET_STD, image_transforms
from ee6483.models import build_model
from ee6483.utils import ensure_new_file, save_json, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise correct and incorrect validation examples"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/error_analysis.png"))
    parser.add_argument("--examples-per-type", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_output = args.output.with_suffix(".json")
    ensure_new_file(args.output)
    ensure_new_file(metadata_output)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    _, transform = image_transforms(
        checkpoint["image_size"], checkpoint.get("augmentation", "standard")
    )
    dataset = datasets.ImageFolder(args.data_dir / "val", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = build_model(checkpoint["model_name"], 2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    correct, incorrect = [], []
    dataset_index = 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(1).cpu()
            predictions = probabilities.argmax(1)
            for image, actual, predicted, probability in zip(
                images, labels, predictions, probabilities
            ):
                source_path = dataset.samples[dataset_index][0]
                dataset_index += 1
                item = (
                    image,
                    int(actual),
                    int(predicted),
                    float(probability[predicted]),
                    source_path,
                )
                (correct if actual == predicted else incorrect).append(item)
            if len(correct) >= args.examples_per_type and len(incorrect) >= args.examples_per_type:
                break
    if len(correct) < args.examples_per_type or len(incorrect) < args.examples_per_type:
        raise RuntimeError(
            f"Requested {args.examples_per_type} examples of each type, but found "
            f"{len(correct)} correct and {len(incorrect)} incorrect"
        )
    chosen = correct[: args.examples_per_type] + incorrect[: args.examples_per_type]
    cols = args.examples_per_type
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6), squeeze=False)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    names = ["cat", "dog"]
    records = []
    for axis, (image, actual, predicted, confidence, source_path) in zip(axes.flat, chosen):
        axis.imshow((image * std + mean).permute(1, 2, 0).clamp(0, 1))
        axis.set_title(f"actual={names[actual]}\npred={names[predicted]} ({confidence:.1%})")
        axis.axis("off")
        records.append(
            {
                "source_path": source_path,
                "actual_label": actual,
                "actual_name": names[actual],
                "predicted_label": predicted,
                "predicted_name": names[predicted],
                "confidence": confidence,
                "correct": actual == predicted,
            }
        )
    for axis in axes.flat[len(chosen) :]:
        axis.axis("off")
    axes[0, 0].set_ylabel("Correct")
    axes[1, 0].set_ylabel("Incorrect")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "confusion_convention": "actual labels are ground truth from val/cat and val/dog",
            "examples": records,
        },
        metadata_output,
    )
    print(f"Saved error analysis to {args.output}")


if __name__ == "__main__":
    main()

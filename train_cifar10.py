from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from ee6483.engine import classification_metrics, confusion_matrix, parameter_counts, run_epoch
from ee6483.models import build_model
from ee6483.utils import (
    prepare_run_directory,
    save_json,
    select_device,
    serializable,
    set_seed,
    write_csv_rows,
)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class FocalLoss(nn.Module):
    """Multiclass focal loss; optional weights act as class-wise alpha values."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target, reduction="none")
        true_probability = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - true_probability) ** self.gamma) * cross_entropy
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def long_tailed_indices(targets: list[int], ratio: float, seed: int) -> list[int]:
    """Keep exponentially fewer examples in later classes; ratio=min/max class count."""
    if not 0 < ratio <= 1:
        raise ValueError("imbalance-ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    class_ids = sorted(set(targets))
    class_count = len(class_ids)
    if class_ids != list(range(class_count)):
        raise ValueError("Targets must use contiguous class IDs starting at zero")
    maximum = min(targets.count(class_id) for class_id in class_ids)
    if class_count == 1:
        desired = [maximum]
    else:
        desired = [
            int(maximum * (ratio ** (class_id / (class_count - 1)))) for class_id in class_ids
        ]
    selected: list[int] = []
    target_array = np.asarray(targets)
    for class_id, count in enumerate(desired):
        indices = np.flatnonzero(target_array == class_id)
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def stratified_split(
    targets: list[int], val_per_class: int, seed: int
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    target_array = np.asarray(targets)
    for class_id in sorted(set(targets)):
        indices = np.flatnonzero(target_array == class_id)
        if val_per_class >= len(indices):
            raise ValueError(
                f"val_per_class={val_per_class} leaves no training data for class {class_id}"
            )
        rng.shuffle(indices)
        val_indices.extend(indices[:val_per_class].tolist())
        train_indices.extend(indices[val_per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 classification and imbalance remedies")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cifar10/resnet18_balanced_seed6483"),
        help="Must be a new experiment directory; existing paths are never overwritten.",
    )
    parser.add_argument("--model", choices=["smallcnn", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6483)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--imbalance-ratio", type=float, default=1.0)
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "weighted_sampler", "class_weighted_loss", "focal_loss"],
        default="none",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs, args.batch_size, args.workers, args.model = 1, 16, 0, "smallcnn"
    set_seed(args.seed)
    device = select_device(args.device)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    base_train = datasets.CIFAR10(
        args.data_dir, train=True, download=args.download, transform=train_transform
    )
    base_val = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=test_transform)
    test_data = datasets.CIFAR10(
        args.data_dir, train=False, download=args.download, transform=test_transform
    )
    train_pool, val_indices = stratified_split(
        base_train.targets, val_per_class=500, seed=args.seed
    )
    pool_targets = [base_train.targets[index] for index in train_pool]
    selected_local = long_tailed_indices(pool_targets, args.imbalance_ratio, args.seed)
    indices = [train_pool[index] for index in selected_local]
    if args.smoke_test:
        indices = indices[:64]
        val_indices = val_indices[:32]
        test_data = Subset(test_data, range(32))
    train_data = Subset(base_train, indices)
    val_data = Subset(base_val, val_indices)
    selected_targets = [base_train.targets[index] for index in indices]
    counts = torch.bincount(torch.tensor(selected_targets), minlength=10).float()
    inverse = counts.sum() / counts.clamp_min(1)
    sampler = None
    shuffle = True
    if args.imbalance_strategy == "weighted_sampler":
        sample_weights = inverse[torch.tensor(selected_targets)]
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True, generator=generator
        )
        shuffle = False
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    run_dir = prepare_run_directory(args.output_dir)
    (run_dir / "checkpoints").mkdir()
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    history_path = run_dir / "history.csv"
    model = build_model(args.model, 10, pretrained=False).to(device)
    if args.model == "resnet18":
        # CIFAR-10 images are 32x32: preserve spatial detail instead of ImageNet downsampling.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        model.maxpool = nn.Identity()
    parameter_summary = parameter_counts(model)
    class_weights = inverse.to(device) / inverse.mean()
    if args.imbalance_strategy == "class_weighted_loss":
        train_criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
    elif args.imbalance_strategy == "focal_loss":
        train_criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        train_criterion = nn.CrossEntropyLoss()
    # Unweighted evaluation loss makes runs directly comparable across remedies.
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    config = {
        "task": "cifar10" if args.imbalance_ratio == 1 else "class_imbalance",
        "arguments": serializable(vars(args)),
        "classes": CIFAR10_CLASSES,
        "dataset": {
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "class_counts": counts.int().tolist(),
        },
        "parameters": parameter_summary,
        "training": {
            "loss": type(train_criterion).__name__,
            "evaluation_loss": "CrossEntropyLoss(unweighted)",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "checkpoint_selection": "highest validation accuracy",
        },
        "preprocessing": {
            "train": "RandomCrop(32,padding=4), RandomHorizontalFlip, CIFAR-10 normalization",
            "validation_test": "CIFAR-10 normalization only",
            "resnet18_stem": "3x3 stride-1 convolution, no initial max-pool",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": str(device),
        },
    }
    save_json(config, run_dir / "config.json")

    rows: list[dict] = []
    best_accuracy, best_epoch = -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, train_criterion, device, optimizer, f"train {epoch}"
        )
        val_metrics = run_epoch(model, val_loader, eval_criterion, device, None, f"val {epoch}")
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(time.perf_counter() - epoch_started, 4),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
        write_csv_rows(rows, history_path)
        print(
            f"epoch={epoch:02d} train_acc={train_metrics.accuracy:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f}"
        )
        if val_metrics.accuracy > best_accuracy:
            best_accuracy, best_epoch = val_metrics.accuracy, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_name": args.model,
                    "num_classes": 10,
                    "classes": CIFAR10_CLASSES,
                    "image_size": 32,
                    "pretrained_initialization": False,
                    "cifar_stem": args.model == "resnet18",
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_accuracy": best_accuracy,
                    "parameters": parameter_summary,
                    "config": config,
                },
                checkpoint_path,
            )
        scheduler.step()

    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, eval_criterion, device, None, "final test")
    matrix = confusion_matrix(model, test_loader, device, 10)
    derived = classification_metrics(matrix)
    summary = {
        "task": config["task"],
        "model": args.model,
        "best_epoch": best_epoch,
        "epochs_completed": len(rows),
        "best_val_accuracy": best_accuracy,
        "final_test_loss": test_metrics.loss,
        "final_test_accuracy": test_metrics.accuracy,
        "classification_metrics": derived,
        "class_counts": counts.int().tolist(),
        "class_names": CIFAR10_CLASSES,
        "imbalance_ratio": args.imbalance_ratio,
        "imbalance_strategy": args.imbalance_strategy,
        "confusion_matrix_rows_actual_cols_predicted": matrix.tolist(),
        "training_seconds": round(training_seconds, 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "parameters": parameter_summary,
        "checkpoint": str(checkpoint_path),
    }
    save_json(summary, run_dir / "metrics.json")
    print(summary)


if __name__ == "__main__":
    main()

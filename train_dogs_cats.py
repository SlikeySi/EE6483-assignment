from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import torch
import torchvision
from torch import nn

from ee6483.data import dogs_cats_loaders, resolve_leakage_exclusions
from ee6483.engine import classification_metrics, confusion_matrix, parameter_counts, run_epoch
from ee6483.models import build_model, freeze_backbone
from ee6483.utils import (
    prepare_run_directory,
    save_json,
    select_device,
    serializable,
    set_seed,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reproducible Dogs-vs-Cats classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483"),
        help="Must be a new experiment directory; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--model",
        choices=["smallcnn", "resnet18", "vgg16", "mobilenet_v3_small"],
        default="resnet18",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-weights", type=Path)
    parser.add_argument("--augmentation", choices=["none", "standard", "vgg"], default="standard")
    parser.add_argument(
        "--leakage-manifest",
        type=Path,
        default=Path("configs/dogs_cats_leakage_exclusions.json"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--patience", type=int, default=0, help="0 disables early stopping so all epochs run."
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6483)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs, args.image_size, args.batch_size = 1, 64, 4
        args.max_train_samples, args.max_val_samples = 8, 4
        args.pretrained, args.workers, args.patience = False, 0, 0
        args.local_weights = None

    set_seed(args.seed)
    device = select_device(args.device)
    excluded = resolve_leakage_exclusions(args.data_dir, args.leakage_manifest)
    train_loader, val_loader, class_to_idx = dogs_cats_loaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.workers,
        args.seed,
        args.max_train_samples,
        args.max_val_samples,
        args.augmentation,
        args.leakage_manifest,
    )
    run_dir = prepare_run_directory(args.output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    history_path = run_dir / "history.csv"

    model = build_model(args.model, 2, args.pretrained, args.local_weights).to(device)
    if args.freeze_backbone and args.model != "smallcnn":
        freeze_backbone(model, args.model)
    counts = parameter_counts(model)
    config = {
        "task": "dogs_vs_cats",
        "arguments": serializable(vars(args)),
        "class_to_idx": class_to_idx,
        "dataset": {
            "train_samples_after_exclusion_and_limit": len(train_loader.dataset),
            "validation_samples_after_limit": len(val_loader.dataset),
            "excluded_training_files": [str(path) for path in excluded],
        },
        "parameters": counts,
        "training": {
            "loss": f"CrossEntropyLoss(label_smoothing={args.label_smoothing})",
            "optimizer": "AdamW" if args.optimizer == "adamw" else "Adam",
            "scheduler": "CosineAnnealingLR" if args.scheduler == "cosine" else "none",
            "checkpoint_selection": "highest validation accuracy",
        },
        "preprocessing": {
            "image_size": args.image_size,
            "normalization": "ImageNet mean/std",
            "augmentation": args.augmentation,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": str(device),
        },
    }
    save_json(config, run_dir / "config.json")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == "cosine"
        else None
    )
    best_accuracy, best_epoch, stale_epochs = -1.0, 0, 0
    started = time.perf_counter()
    rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, f"train {epoch}"
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, f"val {epoch}")
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 4),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
        rows.append(row)
        write_csv_rows(rows, history_path)
        print(
            f"epoch={epoch:02d} train_acc={train_metrics.accuracy:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f}"
        )
        if val_metrics.accuracy > best_accuracy:
            best_accuracy, best_epoch, stale_epochs = val_metrics.accuracy, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "model_name": args.model,
                    "num_classes": 2,
                    "class_to_idx": class_to_idx,
                    "image_size": args.image_size,
                    "pretrained_initialization": bool(args.pretrained),
                    "freeze_backbone": bool(args.freeze_backbone),
                    "augmentation": args.augmentation,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_accuracy": best_accuracy,
                    "parameters": counts,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if scheduler:
            scheduler.step()
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_val_metrics = run_epoch(model, val_loader, criterion, device, None, "best validation")
    matrix = confusion_matrix(model, val_loader, device, 2)
    derived = classification_metrics(matrix)
    summary = {
        "task": "dogs_vs_cats",
        "model": args.model,
        "best_epoch": best_epoch,
        "epochs_completed": len(rows),
        "best_val_accuracy": best_accuracy,
        "best_val_loss": best_val_metrics.loss,
        "classification_metrics": derived,
        "confusion_matrix_rows_actual_cols_predicted": matrix.tolist(),
        "training_seconds": round(training_seconds, 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "parameters": counts,
        "checkpoint": str(checkpoint_path),
    }
    save_json(summary, run_dir / "metrics.json")
    print(summary)


if __name__ == "__main__":
    main()

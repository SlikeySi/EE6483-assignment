from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    description: str = "epoch",
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, desc=description, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += batch_size
    return EpochMetrics(total_loss / total_samples, total_correct / total_samples, total_samples)


def confusion_matrix(model: nn.Module, loader, device: torch.device, classes: int) -> torch.Tensor:
    matrix = torch.zeros((classes, classes), dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for actual, predicted in zip(labels, predictions):
                matrix[int(actual), int(predicted)] += 1
    return matrix


def classification_metrics(matrix: torch.Tensor) -> dict[str, Any]:
    """Derive reproducible aggregate/per-class metrics from an actual×predicted matrix."""
    matrix = matrix.to(torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    true_positive = matrix.diag()
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator)
    )
    total = matrix.sum()
    accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": [
            {
                "class_id": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(matrix.shape[0])
        ],
    }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    """Compact baseline used for offline smoke tests and model comparison."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    local_weights: str | Path | None = None,
) -> nn.Module:
    if local_weights is not None and name != "vgg16":
        raise ValueError("--local-weights is supported only for VGG16")
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "vgg16":
        if local_weights is not None:
            if not pretrained:
                raise ValueError("Local VGG16 ImageNet weights require pretrained=True")
            local_weights = Path(local_weights)
            if not local_weights.is_file():
                raise FileNotFoundError(f"VGG16 weight file does not exist: {local_weights}")
            model = models.vgg16(weights=None)
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state)
        else:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model: nn.Module, name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if name == "resnet18" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True

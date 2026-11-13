import pytest
import torch

from ee6483.engine import classification_metrics
from ee6483.models import build_model, freeze_backbone


def test_smallcnn_output_shape():
    model = build_model("smallcnn", num_classes=2, pretrained=False)
    output = model(torch.randn(3, 3, 64, 64))
    assert output.shape == (3, 2)


def test_resnet_output_shape():
    model = build_model("resnet18", num_classes=10, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 10)


def test_vgg16_head_and_feature_freezing():
    model = build_model("vgg16", num_classes=2, pretrained=False)
    freeze_backbone(model, "vgg16")
    assert model.classifier[-1].out_features == 2
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_classification_metrics_from_confusion_matrix():
    metrics = classification_metrics(torch.tensor([[8, 2], [1, 9]]))
    assert metrics["accuracy"] == 0.85
    assert metrics["balanced_accuracy"] == pytest.approx(0.85)
    assert metrics["per_class"][0]["recall"] == 0.8

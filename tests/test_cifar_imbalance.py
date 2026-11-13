from collections import Counter

import torch

from train_cifar10 import FocalLoss, long_tailed_indices, stratified_split


def test_long_tailed_indices_are_reproducible_and_imbalanced():
    targets = [class_id for class_id in range(10) for _ in range(100)]
    first = long_tailed_indices(targets, ratio=0.1, seed=6483)
    second = long_tailed_indices(targets, ratio=0.1, seed=6483)
    counts = Counter(targets[index] for index in first)
    assert first == second
    assert counts[0] == 100
    assert counts[9] == 10
    assert all(counts[i] >= counts[i + 1] for i in range(9))


def test_stratified_split_has_no_overlap_and_equal_validation_counts():
    targets = [class_id for class_id in range(10) for _ in range(20)]
    train_indices, val_indices = stratified_split(targets, val_per_class=5, seed=6483)
    counts = Counter(targets[index] for index in val_indices)
    assert len(train_indices) == 150
    assert len(val_indices) == 50
    assert set(train_indices).isdisjoint(val_indices)
    assert set(counts.values()) == {5}


def test_focal_loss_matches_probability_definition():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    targets = torch.tensor([0, 1])
    probabilities = torch.softmax(logits, dim=1)[torch.arange(2), targets]
    expected = (((1 - probabilities) ** 2) * (-torch.log(probabilities))).mean()
    assert torch.allclose(FocalLoss(gamma=2.0)(logits, targets), expected)

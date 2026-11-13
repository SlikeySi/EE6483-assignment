from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ee6483.utils import numeric_image_id, sorted_test_images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transforms(
    image_size: int = 224, augmentation: str = "standard"
) -> tuple[transforms.Compose, transforms.Compose]:
    if augmentation == "vgg":
        # Preserve the teammate VGG16 protocol while reusing the shared loader.
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return train_transform, eval_transform

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if augmentation == "none":
        train_transform = eval_transform
    elif augmentation == "standard":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation profile: {augmentation}")
    return train_transform, eval_transform


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_leakage_exclusions(
    data_dir: str | Path, manifest_path: str | Path | None
) -> list[Path]:
    """Validate and return training files excluded due to cross-split duplicates."""
    if manifest_path is None:
        return []
    data_dir = Path(data_dir).resolve()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded: list[Path] = []
    for record in manifest.get("cross_split_duplicates", []):
        training = (data_dir / record["exclude_from_training"]).resolve()
        validation = (data_dir / record["retained_for_validation"]).resolve()
        expected_hash = record["sha256"]
        if not training.is_relative_to(data_dir) or not validation.is_relative_to(data_dir):
            raise ValueError("Leakage manifest paths must remain inside --data-dir")
        for path in (training, validation):
            if not path.is_file():
                raise FileNotFoundError(f"Leakage manifest file is missing: {path}")
            actual_hash = file_sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Leakage manifest hash mismatch for {path}: {actual_hash} != {expected_hash}"
                )
        excluded.append(training)
    if len(excluded) != len(set(excluded)):
        raise ValueError("Leakage manifest contains duplicate exclusions")
    return excluded


class UnlabelledImageDataset(Dataset):
    def __init__(self, directory: str | Path, transform=None):
        self.paths = sorted_test_images(directory)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, numeric_image_id(path), str(path)


def _limited(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=generator)[:limit].tolist())


def dogs_cats_loaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    augmentation: str = "standard",
    leakage_manifest: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    data_dir = Path(data_dir)
    train_transform, eval_transform = image_transforms(image_size, augmentation)
    train_data = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_data = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    expected = {"cat": 0, "dog": 1}
    if train_data.class_to_idx != expected or val_data.class_to_idx != expected:
        raise ValueError(f"Expected class mapping {expected}; got {train_data.class_to_idx}")
    excluded = set(resolve_leakage_exclusions(data_dir, leakage_manifest))
    if excluded:
        keep_indices = [
            index
            for index, (path, _) in enumerate(train_data.samples)
            if Path(path).resolve() not in excluded
        ]
        found = len(train_data) - len(keep_indices)
        if found != len(excluded):
            raise ValueError(
                f"Expected to exclude {len(excluded)} training files, but matched {found}"
            )
        train_data = Subset(train_data, keep_indices)
    train_data = _limited(train_data, max_train_samples, seed)
    val_data = _limited(val_data, max_val_samples, seed + 1)
    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, expected

import json
from pathlib import Path

import pytest
from PIL import Image

from ee6483.data import file_sha256, image_transforms, resolve_leakage_exclusions


def test_augmentation_profiles_are_explicit():
    no_augmentation, evaluation = image_transforms(64, "none")
    standard, _ = image_transforms(64, "standard")
    vgg, vgg_evaluation = image_transforms(64, "vgg")
    assert no_augmentation is evaluation
    assert "RandomResizedCrop" in repr(standard)
    assert "RandomRotation(degrees=[-15.0, 15.0]" in repr(vgg)
    assert "Resize(size=(64, 64)" in repr(vgg_evaluation)
    with pytest.raises(ValueError, match="Unknown augmentation"):
        image_transforms(64, "invented")


def test_leakage_manifest_validates_hash_and_preserves_files(tmp_path: Path):
    training = tmp_path / "train" / "cat" / "train.jpg"
    validation = tmp_path / "val" / "cat" / "validation.jpg"
    training.parent.mkdir(parents=True)
    validation.parent.mkdir(parents=True)
    image = Image.new("RGB", (4, 4), "red")
    image.save(training)
    image.save(validation)
    digest = file_sha256(training)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cross_split_duplicates": [
                    {
                        "exclude_from_training": "train/cat/train.jpg",
                        "retained_for_validation": "val/cat/validation.jpg",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert resolve_leakage_exclusions(tmp_path, manifest) == [training.resolve()]
    assert training.is_file() and validation.is_file()

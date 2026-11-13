import csv
from pathlib import Path

import pytest

from ee6483.utils import (
    numeric_image_id,
    prepare_run_directory,
    sorted_test_images,
    validate_submission,
    write_submission,
)


def test_numeric_image_sorting(tmp_path: Path):
    for name in ["10.jpg", "2.jpg", "1.jpg", "ignore.txt"]:
        (tmp_path / name).touch()
    assert [path.name for path in sorted_test_images(tmp_path)] == ["1.jpg", "2.jpg", "10.jpg"]


def test_non_numeric_test_id_rejected():
    with pytest.raises(ValueError):
        numeric_image_id("dog.jpg")


def test_submission_schema_and_labels(tmp_path: Path):
    output = tmp_path / "submission.csv"
    write_submission([(1, 0), (2, 1)], output)
    with output.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["id", "label"], ["1", "0"], ["2", "1"]]


def test_invalid_submission_label_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        write_submission([(1, 2)], tmp_path / "bad.csv")


def test_submission_ids_must_be_complete_and_unique():
    assert validate_submission([(2, 1), (1, 0)], expected_ids=[1, 2]) == [(1, 0), (2, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission([(1, 0), (1, 1)])
    with pytest.raises(ValueError, match="expected test set"):
        validate_submission([(1, 0)], expected_ids=[1, 2])


def test_run_directory_refuses_overwrite(tmp_path: Path):
    run = prepare_run_directory(tmp_path / "run")
    assert run.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run_directory(run)

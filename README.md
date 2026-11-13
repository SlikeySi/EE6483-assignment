# EE6483 Mini Project

Compact PyTorch codebase for the EE6483 image-classification experiments:

1. **Dogs vs Cats** — SmallCNN, ImageNet-pretrained ResNet18, and the integrated VGG16 protocol.
2. **CIFAR-10** — CIFAR-adapted ResNet18 ten-class classification.
3. **Class imbalance** — untreated baseline, Weighted Random Sampler, Class-Weighted Loss, and Focal Loss.

Dogs-vs-Cats labels are fixed as `0=cat` and `1=dog`. All training entry points share the same
data, model, engine, logging, checkpoint, and reproducibility utilities in `ee6483/`.

## Repository structure

```text
datasets/
├── configs/
│   └── dogs_cats_leakage_exclusions.json
├── ee6483/
│   ├── __init__.py
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   └── utils.py
├── scripts/
│   ├── run_three_seeds.sh
│   ├── setup.sh
│   └── verify_dataset.py
├── tests/
├── docs/
├── train_dogs_cats.py
├── train_cifar10.py
├── predict.py
├── analyze_errors.py
├── requirements.txt
└── pyproject.toml
```

Datasets, checkpoints, logs, predictions, figures, reports, virtual environments, and archives are
local artifacts and are excluded by `.gitignore`. They are not required in the GitHub source tree.

## Setup and lightweight verification

```bash
cd /path/to/datasets
bash scripts/setup.sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Expected local Dogs-vs-Cats layout:

```text
datasets/
├── train/cat/  train/dog/
├── val/cat/    val/dog/
└── test/
```

The leakage-exclusion manifest is applied by the shared loader without deleting any source image.

## Training

### Dogs vs Cats — SmallCNN

```bash
.venv/bin/python train_dogs_cats.py --data-dir . --model smallcnn --no-pretrained \
  --epochs 12 --batch-size 32 --learning-rate 3e-4 --augmentation standard --seed 6483 \
  --output-dir outputs/dogs_vs_cats/smallcnn_standard_lr3e-4_seed6483
```

### Dogs vs Cats — pretrained ResNet18

```bash
.venv/bin/python train_dogs_cats.py --data-dir . --model resnet18 --pretrained \
  --no-freeze-backbone --epochs 12 --batch-size 32 --learning-rate 3e-4 \
  --augmentation standard --seed 6483 \
  --output-dir outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483
```

### Dogs vs Cats — integrated VGG16 protocol

Place the local ImageNet weight file outside this repository, then pass its path explicitly:

```bash
.venv/bin/python train_dogs_cats.py --data-dir . --model vgg16 --pretrained \
  --local-weights /path/to/vgg16-397923af.pth --freeze-backbone \
  --epochs 12 --batch-size 32 --image-size 224 --learning-rate 1e-4 \
  --weight-decay 0 --optimizer adam --scheduler none --label-smoothing 0 \
  --augmentation vgg --seed 42 \
  --output-dir outputs/dogs_vs_cats/vgg16_frozen_adam_lr1e-4_seed42
```

### Balanced CIFAR-10

```bash
.venv/bin/python train_cifar10.py --data-dir data/cifar10 --download --model resnet18 \
  --epochs 30 --batch-size 128 --learning-rate 1e-3 --imbalance-ratio 1 \
  --imbalance-strategy none --seed 6483 \
  --output-dir outputs/cifar10/resnet18_balanced_seed6483
```

The full ablation and class-imbalance command matrix is in
[`docs/experiment_plan.md`](docs/experiment_plan.md).

## Prediction and labelled error analysis

```bash
.venv/bin/python predict.py --data-dir . \
  --checkpoint outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483/checkpoints/best.pt \
  --output outputs/dogs_vs_cats/formal_submission/submission.csv

.venv/bin/python analyze_errors.py --data-dir . \
  --checkpoint outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483/checkpoints/best.pt \
  --output outputs/dogs_vs_cats/formal_analysis/correct_incorrect_examples.png
```

`predict.py` validates IDs 1–500 and writes exactly the columns `id,label`.
`analyze_errors.py` uses the true labels in `val/cat` and `val/dog`.

## Output protection

Formal run directories are immutable: the training scripts reject an existing `--output-dir`, and
prediction/analysis scripts reject an existing output file. A completed run records configuration,
per-epoch history, best checkpoint, final metrics, timing, class metrics, and confusion matrices.

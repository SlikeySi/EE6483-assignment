# Formal experiment plan

Run from `/Users/qqqsss/6483project/datasets`. A run directory is immutable: if a command needs to
be repeated, use a new seed or suffix instead of reusing its output path.

## 1. Dogs vs Cats main comparison

Both models use the same split, image size, normalization, augmentation, 12 epochs, batch size,
learning rate, optimizer, scheduler, seed, and validation protocol. SmallCNN trains from scratch;
ResNet18 starts from ImageNet weights and fine-tunes its full backbone.

```bash
.venv/bin/python train_dogs_cats.py --data-dir . --model smallcnn --no-pretrained \
  --epochs 12 --batch-size 32 --learning-rate 3e-4 --augmentation standard --seed 6483 \
  --output-dir outputs/dogs_vs_cats/smallcnn_standard_lr3e-4_seed6483

.venv/bin/python train_dogs_cats.py --data-dir . --model resnet18 --pretrained \
  --no-freeze-backbone --epochs 12 --batch-size 32 --learning-rate 3e-4 \
  --augmentation standard --seed 6483 \
  --output-dir outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483
```

Optional three-seed repeats: `bash scripts/run_three_seeds.sh`. Do not repeat a command whose run
directory already exists; completed seed-6483 main runs already count toward the seed study.

## 2. Learning-rate and augmentation ablations

Only the named factor changes. Reuse the ResNet18 `3e-4 + standard` main run in both comparisons.

```bash
# Learning rate: 1e-4, 3e-4 (main run), 1e-3
for LR_TAG in 1e-4 1e-3; do
  .venv/bin/python train_dogs_cats.py --data-dir . --model resnet18 --pretrained \
    --no-freeze-backbone --epochs 12 --batch-size 32 --learning-rate "$LR_TAG" \
    --augmentation standard --seed 6483 \
    --output-dir "outputs/dogs_vs_cats/resnet18_standard_lr${LR_TAG}_seed6483"
done

# Augmentation: none versus standard (main run)
.venv/bin/python train_dogs_cats.py --data-dir . --model resnet18 --pretrained \
  --no-freeze-backbone --epochs 12 --batch-size 32 --learning-rate 3e-4 \
  --augmentation none --seed 6483 \
  --output-dir outputs/dogs_vs_cats/resnet18_none_lr3e-4_seed6483
```

## 3. Teammate VGG16 experiment

This command retains the teammate's model-specific choices while using the shared leakage-safe
loader, logging, checkpoint, and prediction infrastructure. The convolutional feature extractor is
frozen; all layers in the VGG classifier remain trainable, matching the original code. Use a fresh
output directory because formal runs are immutable.

```bash
.venv/bin/python train_dogs_cats.py --data-dir . --model vgg16 --pretrained \
  --local-weights ../VGG/vgg16-397923af.pth --freeze-backbone \
  --epochs 12 --batch-size 32 --image-size 224 --learning-rate 1e-4 \
  --weight-decay 0 --optimizer adam --scheduler none --label-smoothing 0 \
  --augmentation vgg --seed 42 \
  --output-dir outputs/dogs_vs_cats/vgg16_frozen_adam_lr1e-4_seed42
```

The historical teammate log reports 98.96% validation accuracy at epochs 3 and 7, but its best
checkpoint file is not present. Do not treat the ImageNet file `vgg16-397923af.pth` as that trained
checkpoint. A merged-code rerun may also differ because the shared loader excludes the verified
train/validation duplicate and the retained checkpoint rule does not overwrite an earlier equal
score.

## 4. Dogs-vs-Cats prediction and case analysis

```bash
.venv/bin/python predict.py --data-dir . \
  --checkpoint outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483/checkpoints/best.pt \
  --output outputs/dogs_vs_cats/formal_submission/submission.csv

.venv/bin/python analyze_errors.py --data-dir . \
  --checkpoint outputs/dogs_vs_cats/resnet18_standard_lr3e-4_seed6483/checkpoints/best.pt \
  --examples-per-type 4 \
  --output outputs/dogs_vs_cats/formal_analysis/correct_incorrect_examples.png
```

## 5. Balanced CIFAR-10

Use `--download` only on the first run when `data/cifar10` is absent. Later commands omit it.

```bash
.venv/bin/python train_cifar10.py --data-dir data/cifar10 --download --model resnet18 \
  --epochs 30 --batch-size 128 --learning-rate 1e-3 --imbalance-ratio 1 \
  --imbalance-strategy none --seed 6483 \
  --output-dir outputs/cifar10/resnet18_balanced_seed6483
```

## 6. Class-imbalance comparison

All runs use identical long-tailed indices (`ratio=0.1`, `seed=6483`), architecture, epochs,
optimizer, schedule, validation split, and test set. Only the remedy changes.

```bash
for METHOD in none weighted_sampler class_weighted_loss focal_loss; do
  .venv/bin/python train_cifar10.py --data-dir data/cifar10 --no-download --model resnet18 \
    --epochs 30 --batch-size 128 --learning-rate 1e-3 --imbalance-ratio 0.1 \
    --imbalance-strategy "$METHOD" --seed 6483 \
    --output-dir "outputs/class_imbalance/${METHOD}_ratio0.1_seed6483"
done
```

The required comparison is `none`, `weighted_sampler`, and `class_weighted_loss`; `focal_loss` is
the resource-permitting extension. Every metrics file reports Accuracy, Macro-F1, Balanced
Accuracy, each class's Recall/Precision/F1, and the actual×predicted confusion matrix.

Generated report assets are intentionally excluded from the source-only GitHub repository. Each
training run still writes machine-readable `history.csv`, `metrics.json`, and `config.json` files
under its immutable output directory for downstream analysis.

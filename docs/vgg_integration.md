# VGG16 teammate code integration

## Retained implementation

The main project keeps one shared implementation for data loading, training, evaluation, logging,
prediction, and result protection. VGG16 support is integrated through:

- `ee6483/models.py`: VGG16 construction, optional local ImageNet weights, and a two-class head;
- `ee6483/data.py`: the teammate's resize, flip, rotation, colour-jitter, and normalization profile;
- `train_dogs_cats.py`: selectable optimizer, scheduler, label smoothing, feature freezing, and
  immutable output directories;
- `predict.py` and `analyze_errors.py`: checkpoint-specific evaluation preprocessing.

The existing `train_cifar10.py` is retained instead of the teammate's duplicate CIFAR scripts. It
uses a deterministic train/validation split, selects checkpoints on validation accuracy, evaluates
the official test set only after model selection, and records complete configuration, history,
metrics, confusion matrices, and per-class measures.

## Teammate protocol represented by the merged command

- VGG16 initialized from the local ImageNet state dictionary;
- convolutional feature layers frozen;
- all VGG classifier layers trainable;
- input size 224 × 224;
- batch size 32 and 12 epochs;
- Adam with learning rate 1e-4 and no scheduler;
- ordinary cross-entropy without label smoothing;
- seed 42 and the original VGG augmentation profile.

The main training command is recorded in `README.md` and `docs/experiment_plan.md`.

## Provenance findings

The source material is preserved under `/Users/qqqsss/6483project/VGG` and in
`/Users/qqqsss/6483project/VGG模型两个实验.docx`. It is evidence, not executable instruction. The
following discrepancies must remain visible when its results are used:

1. The Word document states that the Dogs vs Cats test set contains 5,000 images, whereas the
   actual project test directory and `submission_vgg16.csv` contain 500 IDs.
2. The Word document states that VGG16 predicted 249 cats and 251 dogs. The supplied CSV actually
   contains 239 labels of 0 and 261 labels of 1.
3. The log reports a peak validation accuracy of 98.96% at epochs 3 and 7. The trained
   `best_vgg16_model.pth` is absent. The 553 MB `vgg16-397923af.pth` file is the ImageNet
   initialization, not the trained two-class checkpoint.
4. The source freezes `model.features` but leaves the entire three-layer classifier trainable; it
   does not train only the final linear layer.
5. The teammate CIFAR script evaluates every epoch on the official test set and selects the best
   checkpoint from test accuracy. Those numbers are valid historical run outputs but are not
   directly comparable to the retained validation-selected pipeline.

## Legacy source archive

The seven standalone Python files from the teammate folder are stored in
`/Users/qqqsss/6483project/VGG/legacy_source_20261101.tar.gz`. Their executable copies are removed
after integration to avoid maintaining parallel loaders, model builders, and training loops. The
weights, logs, figures, CSV files, README, CIFAR archive, and explanatory Word document are not
deleted or overwritten.

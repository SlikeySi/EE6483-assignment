#!/usr/bin/env bash
set -euo pipefail

for model in smallcnn resnet18; do
  for seed in 6483 6484 6485; do
    if [[ "$model" == "resnet18" ]]; then
      pretraining_flag="--pretrained"
    else
      pretraining_flag="--no-pretrained"
    fi
    .venv/bin/python train_dogs_cats.py \
      --data-dir . --model "$model" "$pretraining_flag" --epochs 12 --batch-size 32 \
      --learning-rate 3e-4 --augmentation standard --seed "$seed" \
      --output-dir "outputs/dogs_vs_cats/${model}_standard_lr3e-4_seed${seed}"
  done
done


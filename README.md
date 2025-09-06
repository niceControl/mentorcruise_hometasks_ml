# MentorCruise Projects

This repository contains homeworks and exercises completed as part of my MentorCruise learning program.

## 📂 Structure

- **single_layer_pytorch/** — assignments related to PyTorch (single-layer model training).
- **.flake8** — linting configuration.
- **.gitignore** — git ignore rules.
- **requirements.txt** — Python dependencies.

## 🚀 Setup

First, create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Run pytorch implementation with arguments:
Arguments

--file (default: wine-white.pt)
Path to the dataset .pt file.

--task (regression | classification, default: regression)
Select task type.

--epochs (int, default: 1200)
Number of training epochs.

--batch_size (int, default: 128)
Training batch size.

--lr (float, default: 5.5e-2)
Learning rate.

--weight_decay (float, default: 1e-4)
L2 regularization (weight decay).

--hidden (int, default: 5)
Number of units in the hidden layer.

--p_drop (float, default: 0.3)
Dropout probability after hidden layer.

## Test Run on pytorch implemetation

Example: train a classification model on the White Wine dataset.

```bash
python single_layer_pytorch/model_pytorch.py \
  --file ./single_layer_pytorch/data/wine-white.pt \
  --task classification \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.01 \
  --weight_decay 1e-4 \
  --hidden 10 \
  --p_drop 0.3
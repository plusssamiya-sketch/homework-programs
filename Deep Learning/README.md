# Deep Learning Assignments

This folder contains homework programs for neural network modeling. The scripts are written as clean training templates with explicit data loading, model definition, training loops, evaluation, and reproducibility controls.

## Programs

- `mlp_tabular_classification.py`: multilayer perceptron for tabular binary classification with validation tracking and early stopping.
- `cnn_image_classification.py`: compact convolutional neural network for image classification. It uses `torchvision.datasets.FakeData` by default so the program runs without downloading external data.

## Learning Objectives

- Understand the full PyTorch workflow: dataset, dataloader, module, optimizer, loss, train/evaluate loop.
- Track training and validation performance over epochs.
- Apply regularization through dropout, weight decay, and early stopping.
- Keep code modular enough for replacing synthetic data with real course datasets.

## Run

```bash
python mlp_tabular_classification.py
python cnn_image_classification.py
```

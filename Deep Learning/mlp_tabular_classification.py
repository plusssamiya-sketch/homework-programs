"""
Deep Learning Homework 1: MLP for Tabular Classification

Task:
    Train a neural network on standardized tabular data and report validation
    performance with early stopping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 2026


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def make_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader, int]:
    x, y = make_classification(
        n_samples=1_200,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        class_sep=1.2,
        random_state=SEED,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size), x.shape[1]


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            losses.append(float(loss.item()))
    return float(np.mean(losses)), correct / total


def train(max_epochs: int = 40, patience: int = 5) -> list[EpochMetrics]:
    set_seed()
    train_loader, val_loader, input_dim = make_loaders()
    model = MLPClassifier(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        metrics = EpochMetrics(epoch, float(np.mean(train_losses)), val_loss, val_accuracy)
        history.append(metrics)
        print(
            f"epoch={metrics.epoch:02d} train_loss={metrics.train_loss:.4f} "
            f"val_loss={metrics.val_loss:.4f} val_acc={metrics.val_accuracy:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return history


if __name__ == "__main__":
    train()

"""
Deep Learning Homework 3: Transformer Encoder for Sequence Classification

Task:
    Train a compact transformer encoder on synthetic sequence data. The design
    demonstrates embeddings, positional encodings, attention masks, and a
    pooled classification head.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


SEED = 77
VOCAB_SIZE = 40
SEQ_LEN = 24


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = SEQ_LEN) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 48, n_heads: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.15,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.position(self.embedding(tokens))
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def make_sequence_dataset(n_samples: int = 1_500) -> TensorDataset:
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(1, VOCAB_SIZE, size=(n_samples, SEQ_LEN))
    early_signal = tokens[:, :8].mean(axis=1)
    late_signal = tokens[:, -8:].mean(axis=1)
    motif_signal = ((tokens[:, 5] % 3 == 0) & (tokens[:, 12] > 20)).astype(float)
    logits = 0.12 * (late_signal - early_signal) + 1.1 * motif_signal + rng.normal(0, 0.6, n_samples)
    labels = (logits > np.median(logits)).astype(np.int64)
    return TensorDataset(torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            predictions = model(tokens).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / total


def main() -> None:
    set_seed()
    config = TrainConfig()
    dataset = make_sequence_dataset()
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [1_200, 300], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_accuracy = evaluate(model, val_loader, device)
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_accuracy={val_accuracy:.3f} device={device}")


if __name__ == "__main__":
    main()

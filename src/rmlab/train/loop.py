"""Minimal training loop for supervised classification.

This is a smoke-test version. The full Trainer (early stopping, checkpoints, scheduler,
mixed precision) is PLAN item 3.5, where the training step will be rewritten by hand.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def build_optimizer(
    params, name: str, lr: float, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}; available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str
) -> float:
    """One pass over the training data; returns the mean cross-entropy loss."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"training loss is {loss.item()}; stopping the run")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Mean cross-entropy loss and accuracy over a whole split."""
    model.eval()
    total, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)
    return {"loss": total / count, "accuracy": correct / count}

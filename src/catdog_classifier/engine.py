"""实现训练循环、验证、最佳模型保存和曲线绘制。"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class TrainingCheckpoint(TypedDict):
    model_state: dict[str, Tensor]
    class_names: tuple[str, str]
    image_size: int
    epoch: int
    validation_accuracy: float


class HistoryPayload(TypedDict):
    train_loss: list[float]
    validation_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)

    def to_payload(self) -> HistoryPayload:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "train_accuracy": self.train_accuracy,
            "validation_accuracy": self.validation_accuracy,
        }


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("数据加载器为空，无法计算训练指标")
    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    class_names: tuple[str, str],
    image_size: int,
    logger: logging.Logger,
) -> TrainingHistory:
    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()
    best_accuracy = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, device, None
        )
        history.train_loss.append(train_metrics.loss)
        history.validation_loss.append(validation_metrics.loss)
        history.train_accuracy.append(train_metrics.accuracy)
        history.validation_accuracy.append(validation_metrics.accuracy)
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%",
            epoch,
            epochs,
            train_metrics.loss,
            train_metrics.accuracy * 100,
            validation_metrics.loss,
            validation_metrics.accuracy * 100,
        )
        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            checkpoint: TrainingCheckpoint = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "image_size": image_size,
                "epoch": epoch,
                "validation_accuracy": validation_metrics.accuracy,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")

    _save_history(history, output_dir)
    logger.info("训练完成，最佳验证准确率 %.2f%%", best_accuracy * 100)
    return history


def _save_history(history: TrainingHistory, output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    epochs = range(1, len(history.train_loss) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.validation_loss, label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history.train_accuracy, label="train")
    axes[1].plot(epochs, history.validation_accuracy, label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


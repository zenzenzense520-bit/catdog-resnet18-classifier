"""验证训练循环会保存最佳模型、指标历史和曲线。"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from catdog_classifier.engine import train_model


def test_train_model_saves_all_artifacts(tmp_path: Path) -> None:
    images = torch.rand(8, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = train_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        optimizer=optimizer,
        epochs=1,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        class_names=("Cat", "Dog"),
        image_size=224,
        logger=logging.getLogger("test"),
    )

    assert len(history.train_loss) == 1
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "training_curves.png").is_file()


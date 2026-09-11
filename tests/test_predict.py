"""验证单图推理可以读取检查点并返回合法结果。"""

from pathlib import Path

import torch
from PIL import Image

from catdog_classifier.model import build_resnet18
from catdog_classifier.predict import predict_image


def test_predict_image_returns_class_and_confidence(tmp_path: Path) -> None:
    model = build_resnet18(num_classes=2, pretrained=False)
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": ("Cat", "Dog"),
            "image_size": 64,
            "epoch": 1,
            "validation_accuracy": 0.5,
        },
        model_path,
    )
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 80), color=(100, 120, 140)).save(image_path)

    prediction = predict_image(image_path, model_path)

    assert prediction.label in {"Cat", "Dog"}
    assert 0.0 <= prediction.confidence <= 1.0

"""加载训练检查点并预测单张猫狗图片。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from PIL import Image, UnidentifiedImageError

from catdog_classifier.data import build_transforms
from catdog_classifier.engine import TrainingCheckpoint, select_device
from catdog_classifier.logging_utils import configure_logging
from catdog_classifier.model import build_resnet18


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float

    def to_json(self) -> str:
        return json.dumps(
            {"label": self.label, "confidence": round(self.confidence, 6)},
            ensure_ascii=False,
        )


def load_checkpoint(model_path: Path, device: torch.device) -> TrainingCheckpoint:
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    raw_checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("模型文件格式无效")
    required_keys = {
        "model_state",
        "class_names",
        "image_size",
        "epoch",
        "validation_accuracy",
    }
    if not required_keys.issubset(raw_checkpoint):
        raise ValueError("模型文件缺少必要字段")
    return cast(TrainingCheckpoint, raw_checkpoint)


def predict_image(image_path: Path, model_path: Path) -> Prediction:
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    device = select_device()
    checkpoint = load_checkpoint(model_path, device)
    model = build_resnet18(len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    _, validation_transform = build_transforms(checkpoint["image_size"])
    try:
        with Image.open(image_path) as image:
            image_tensor = validation_transform(image.convert("RGB")).unsqueeze(0)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{image_path}") from exc
    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor.to(device)), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    return Prediction(
        label=checkpoint["class_names"][class_index],
        confidence=float(probabilities[class_index].item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="预测单张图片是猫还是狗")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.pt"))
    arguments = parser.parse_args()
    logger = configure_logging("predict")
    try:
        prediction = predict_image(arguments.image, arguments.model)
        logger.info(
            "预测结果：%s，置信度 %.2f%%", prediction.label, prediction.confidence * 100
        )
        print(prediction.to_json())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("推理失败：%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


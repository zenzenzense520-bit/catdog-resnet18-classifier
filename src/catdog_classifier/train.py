"""解析命令行参数并启动 ResNet18 迁移学习训练。"""

import argparse
from pathlib import Path

import torch

from catdog_classifier.config import TrainConfig
from catdog_classifier.data import create_data_loaders
from catdog_classifier.engine import seed_everything, select_device, train_model
from catdog_classifier.logging_utils import configure_logging
from catdog_classifier.model import build_resnet18


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练猫狗二分类 ResNet18")
    parser.add_argument("--data-dir", type=Path, required=True, help="数据集根目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    return TrainConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        validation_ratio=arguments.validation_ratio,
        image_size=arguments.image_size,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )


def main() -> None:
    logger = configure_logging("train")
    try:
        config = parse_args()
        config.validate()
        seed_everything(config.seed)
        device = select_device()
        logger.info("训练设备：%s", device)
        data_loaders = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            validation_ratio=config.validation_ratio,
            image_size=config.image_size,
            num_workers=config.num_workers,
            seed=config.seed,
            logger=logger,
        )
        model = build_resnet18(num_classes=len(data_loaders.class_names))
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=config.learning_rate, weight_decay=1e-4
        )
        train_model(
            model=model,
            train_loader=data_loaders.train,
            validation_loader=data_loaders.validation,
            optimizer=optimizer,
            epochs=config.epochs,
            device=device,
            output_dir=config.output_dir,
            class_names=data_loaders.class_names,
            image_size=config.image_size,
            logger=logger,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("训练失败：%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


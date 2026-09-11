"""读取 Kaggle Cats vs Dogs 数据并执行分层训练/验证划分。"""

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES: tuple[str, str] = ("Cat", "Dog")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    class_names: tuple[str, str]


class CatDogDataset(Dataset[tuple[Tensor, Tensor]]):
    """以强类型样本列表为输入的猫狗图片数据集。"""

    def __init__(self, samples: Sequence[ImageSample], transform: ImageTransform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"读取图片失败：{sample.path}") from exc
        return tensor, torch.tensor(sample.label, dtype=torch.long)


def _find_class_root(data_dir: Path) -> Path:
    candidates = (data_dir, data_dir / "PetImages", data_dir / "train")
    for candidate in candidates:
        if all((candidate / class_name).is_dir() for class_name in CLASS_NAMES):
            return candidate
    expected = "、".join(str(data_dir / name) for name in CLASS_NAMES)
    raise FileNotFoundError(f"未找到 Cat/Dog 子目录，期望类似：{expected}")


def _is_valid_image(path: Path, logger: logging.Logger) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning("跳过损坏图片 %s：%s", path, exc)
        return False


def discover_samples(data_dir: Path, logger: logging.Logger) -> list[ImageSample]:
    """发现并校验两个类别的所有图片。"""
    class_root = _find_class_root(data_dir)
    samples: list[ImageSample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        paths = sorted(
            path
            for path in (class_root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        valid_paths = [path for path in paths if _is_valid_image(path, logger)]
        if len(valid_paths) < 2:
            raise ValueError(f"类别 {class_name} 至少需要 2 张有效图片")
        samples.extend(ImageSample(path=path, label=label) for path in valid_paths)
        logger.info("类别 %s：发现 %d 张有效图片", class_name, len(valid_paths))
    return samples


def stratified_split(
    samples: Sequence[ImageSample], validation_ratio: float, seed: int
) -> tuple[list[ImageSample], list[ImageSample]]:
    """按类别分别打乱并划分，保证训练集和验证集均含两个类别。"""
    random_generator = random.Random(seed)
    train_samples: list[ImageSample] = []
    validation_samples: list[ImageSample] = []
    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample.label == label]
        random_generator.shuffle(class_samples)
        validation_count = max(1, round(len(class_samples) * validation_ratio))
        validation_count = min(validation_count, len(class_samples) - 1)
        validation_samples.extend(class_samples[:validation_count])
        train_samples.extend(class_samples[validation_count:])
    random_generator.shuffle(train_samples)
    random_generator.shuffle(validation_samples)
    return train_samples, validation_samples


def build_transforms(image_size: int) -> tuple[ImageTransform, ImageTransform]:
    """构造训练增强和确定性的验证预处理。"""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    validation_ratio: float,
    image_size: int,
    num_workers: int,
    seed: int,
    logger: logging.Logger,
) -> DataLoaders:
    samples = discover_samples(data_dir, logger)
    train_samples, validation_samples = stratified_split(
        samples, validation_ratio, seed
    )
    train_transform, validation_transform = build_transforms(image_size)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CatDogDataset(train_samples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        CatDogDataset(validation_samples, validation_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(
        "数据划分完成：训练 %d 张，验证 %d 张",
        len(train_samples),
        len(validation_samples),
    )
    return DataLoaders(train_loader, validation_loader, CLASS_NAMES)


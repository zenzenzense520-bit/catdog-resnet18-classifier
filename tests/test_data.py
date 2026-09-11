"""验证数据发现、损坏图片过滤与分层划分。"""

import logging
from pathlib import Path

from PIL import Image

from catdog_classifier.data import discover_samples, stratified_split


def _create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(path)


def test_discover_and_stratified_split(tmp_path: Path) -> None:
    for class_name in ("Cat", "Dog"):
        for index in range(5):
            _create_image(tmp_path / "PetImages" / class_name / f"{index}.jpg")
    samples = discover_samples(tmp_path, logging.getLogger("test"))
    train_samples, validation_samples = stratified_split(samples, 0.2, seed=42)
    assert len(samples) == 10
    assert len(train_samples) == 8
    assert len(validation_samples) == 2
    assert {sample.label for sample in validation_samples} == {0, 1}


def test_discover_skips_corrupt_image(tmp_path: Path) -> None:
    for class_name in ("Cat", "Dog"):
        _create_image(tmp_path / class_name / "1.jpg")
        _create_image(tmp_path / class_name / "2.jpg")
    (tmp_path / "Cat" / "broken.jpg").write_text("not an image", encoding="utf-8")
    samples = discover_samples(tmp_path, logging.getLogger("test"))
    assert len(samples) == 4


"""基于本地 ViT 模型的 NSFW 内容识别，用作推理前置门禁。"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from catdog_classifier.config import NsfwConfig
from catdog_classifier.engine import select_device

NSFW_LABEL = "nsfw"


@dataclass(frozen=True)
class NsfwVerdict:
    """单张图片的不适宜内容判定结果。"""

    label: str
    score: float
    nsfw_score: float
    threshold: float

    @property
    def is_nsfw(self) -> bool:
        return self.nsfw_score >= self.threshold

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "score": round(self.score, 6),
                "nsfw_score": round(self.nsfw_score, 6),
                "is_nsfw": self.is_nsfw,
            },
            ensure_ascii=False,
        )


class NsfwRejectedError(RuntimeError):
    """图片被判定为不适宜内容而拒绝进入分类。"""


class NsfwChecker(Protocol):
    """门禁所需的最小检测接口，便于测试注入替身。"""

    def detect(self, image_path: Path) -> NsfwVerdict: ...


def build_verdict(label_scores: Mapping[str, float], threshold: float) -> NsfwVerdict:
    """把各标签概率解释为判定结果（纯函数，便于单测）。"""
    if not label_scores:
        raise ValueError("模型未返回任何标签概率")
    label = max(label_scores, key=lambda key: label_scores[key])
    nsfw_score = float(label_scores.get(NSFW_LABEL, 0.0))
    return NsfwVerdict(
        label=label,
        score=float(label_scores[label]),
        nsfw_score=nsfw_score,
        threshold=threshold,
    )


class NsfwDetector:
    """加载一次本地 NSFW 模型，可复用于多张图片检测。"""

    def __init__(
        self,
        config: NsfwConfig,
        device: torch.device,
        processor: AutoImageProcessor,
        model: AutoModelForImageClassification,
    ) -> None:
        self._config = config
        self._device = device
        self._processor = processor
        self._model = model

    @classmethod
    def load(cls, config: NsfwConfig, logger: logging.Logger) -> "NsfwDetector":
        """从本地缓存或 HuggingFace 下载并加载模型。"""
        config.validate()
        device = select_device()
        try:
            processor = AutoImageProcessor.from_pretrained(config.model_id)
            model = AutoModelForImageClassification.from_pretrained(config.model_id)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 NSFW 模型 {config.model_id}，首次运行需联网下载权重"
            ) from exc
        model.to(device).eval()
        logger.info("NSFW 模型已加载：%s（设备 %s）", config.model_id, device)
        return cls(config, device, processor, model)

    def detect(self, image_path: Path) -> NsfwVerdict:
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"无法读取图片：{image_path}") from exc
        inputs = self._processor(images=rgb_image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(**inputs).logits[0], dim=-1)
        id_to_label = self._model.config.id2label
        label_scores = {
            str(name): float(probabilities[int(index)])
            for index, name in id_to_label.items()
        }
        return build_verdict(label_scores, self._config.threshold)


def enforce_nsfw_gate(image_path: Path, checker: NsfwChecker) -> NsfwVerdict:
    """执行前置门禁：命中不适宜内容时抛出 NsfwRejectedError。"""
    verdict = checker.detect(image_path)
    if verdict.is_nsfw:
        raise NsfwRejectedError(
            f"图片疑似不适宜内容（NSFW 概率 {verdict.nsfw_score:.2%}），已拒绝分类"
        )
    return verdict
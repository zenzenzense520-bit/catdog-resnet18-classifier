"""验证 NSFW 门禁的判定逻辑与拒绝行为。"""

from pathlib import Path

import pytest
from PIL import Image

from catdog_classifier.nsfw import (
    NsfwChecker,
    NsfwRejectedError,
    NsfwVerdict,
    build_verdict,
    enforce_nsfw_gate,
)


class StubChecker:
    """实现 NsfwChecker 协议的固定结果替身，避免测试下载真实模型。"""

    def __init__(self, verdict: NsfwVerdict) -> None:
        self._verdict = verdict

    def detect(self, image_path: Path) -> NsfwVerdict:
        return self._verdict


def _make_image(tmp_path: Path, name: str) -> Path:
    image_path = tmp_path / name
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(image_path)
    return image_path


def test_stub_checker_satisfies_protocol() -> None:
    checker: NsfwChecker = StubChecker(build_verdict({"normal": 1.0}, 0.5))
    assert isinstance(checker, StubChecker)


def test_build_verdict_flags_nsfw_above_threshold() -> None:
    verdict = build_verdict({"normal": 0.2, "nsfw": 0.8}, threshold=0.5)

    assert verdict.label == "nsfw"
    assert verdict.is_nsfw is True
    assert verdict.nsfw_score == pytest.approx(0.8)


def test_build_verdict_keeps_safe_below_threshold() -> None:
    verdict = build_verdict({"normal": 0.95, "nsfw": 0.05}, threshold=0.5)

    assert verdict.label == "normal"
    assert verdict.is_nsfw is False


def test_build_verdict_without_nsfw_label_is_safe() -> None:
    verdict = build_verdict({"normal": 1.0}, threshold=0.5)

    assert verdict.is_nsfw is False
    assert verdict.nsfw_score == 0.0


def test_build_verdict_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        build_verdict({}, threshold=0.5)


def test_enforce_gate_allows_safe_image(tmp_path: Path) -> None:
    checker = StubChecker(build_verdict({"normal": 0.9, "nsfw": 0.1}, 0.5))

    verdict = enforce_nsfw_gate(_make_image(tmp_path, "safe.jpg"), checker)

    assert verdict.is_nsfw is False


def test_enforce_gate_rejects_nsfw_image(tmp_path: Path) -> None:
    checker = StubChecker(build_verdict({"normal": 0.1, "nsfw": 0.9}, 0.5))

    with pytest.raises(NsfwRejectedError):
        enforce_nsfw_gate(_make_image(tmp_path, "nsfw.jpg"), checker)
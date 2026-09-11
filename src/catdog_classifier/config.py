"""集中定义训练配置，避免未结构化配置在模块间传递。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """一次训练运行所需的完整配置。"""

    data_dir: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_ratio: float = 0.2
    image_size: int = 224
    num_workers: int = 0
    seed: int = 42

    def validate(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir}")
        if self.epochs < 1:
            raise ValueError("epochs 必须大于 0")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio 必须位于 0 和 1 之间")
        if self.image_size < 32:
            raise ValueError("image_size 不能小于 32")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")


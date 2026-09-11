"""配置同时写入控制台与 logs 目录的日志。"""

import logging
from pathlib import Path


def configure_logging(log_name: str) -> logging.Logger:
    """创建项目级日志器，并覆盖同名日志器的旧处理器。"""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_dir / f"{log_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


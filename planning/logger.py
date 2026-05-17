import logging
import sys
from pathlib import Path


def setup_logger(
        name: str = "path_planning",
        level: int = logging.DEBUG,
        log_file: Path = None
) -> logging.Logger:
    """
    Настраивает и возвращает логгер для проекта.
    Args:
        name: имя логгера
        level: уровень логирования
        log_file: путь к файлу для записи логов (если None — только консоль)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



logger = setup_logger()
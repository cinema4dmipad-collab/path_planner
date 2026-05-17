from pathlib import Path
from typing import Union
import numpy as np

from logger import logger
from exceptions import (EmptyFileError,
                                 InvalidFormatError
                                 )


def load_txt(filepath: Union[str, Path]) -> np.ndarray:
    """
        Загружает контур из текстового файла с координатами.
        Формат файла: каждая строка содержит x и y, разделённые пробелом или запятой.
        Пример:
            0.0 0.0
            10.0 0.0
            10.0 10.0
            0.0 10.0
        Args:
            filepath: путь к txt-файлу
        Returns:
            numpy array формы (n, 2) с координатами точек
        """
    logger.info(f'Загрузка файла {filepath}')
    points = []
    line_number = 0
    with open(filepath, 'r') as f:
        for line in f:
            line_number += 1
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 2:
                raise ValueError(
                    f"Строка {line_number}: ожидалось 2 числа, "
                    f"получено {len(parts)}: {line}"
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError as e:
                raise InvalidFormatError(
                    f"Строка {line_number}: некорректные данные: {line}"
                ) from e
            points.append([x, y])
    if not points:
        raise EmptyFileError(f"Файл {filepath} не содержит точек")
    logger.info('Контур успешно загружен')
    return np.array(points)


def close_contour(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """
        Замыкает контур, если последняя точка не совпадает с первой.
        Args:
            points: массив точек (n, 2)
            tolerance: допуск для проверки совпадения точек
        Returns:
            замкнутый контур (последняя точка == первой)
        """
    first = points[0]
    last = points[-1]
    if np.linalg.norm(last - first) > tolerance:
        points = np.vstack([points, first])
    return points


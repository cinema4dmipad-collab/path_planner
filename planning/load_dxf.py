import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional, Union

from logger import logger


try:
    import ezdxf
    from ezdxf.math import Vec2

    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

from exceptions import (
    DXFLoadError,
    UnsupportedEntityError,
    FileNotFoundError as DXFNotFoundError
)


class DXFLoader:
    """
    Загрузчик контуров из DXF-файлов.
    Поддерживает различные геометрические примитивы.
    """

    SUPPORTED_ENTITIES = {
        'LINE', 'LWPOLYLINE', 'POLYLINE',
        'ARC', 'CIRCLE', 'SPLINE', 'ELLIPSE'
    }

    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: точность дискретизации кривых (мм)
        """
        self.tolerance = tolerance

    def load(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Загружает контур из DXF-файла.

        Args:
            filepath: путь к DXF-файлу

        Returns:
            numpy array с точками контура формы (n, 2)

        Raises:
            DXFNotFoundError: если файл не существует
            DXFLoadError: если не удалось загрузить DXF
            UnsupportedEntityError: если встречен неподдерживаемый тип
        """
        if not EZDXF_AVAILABLE:
            raise ImportError(
                "Библиотека ezdxf не установлена. "
                "Установите: poetry add ezdxf"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise DXFNotFoundError(f"Файл не найден: {filepath}")

        logger.info(f"Загрузка DXF-файла: {filepath}")

        try:
            doc = ezdxf.readfile(str(filepath))
        except Exception as e:
            raise DXFLoadError(f"Не удалось прочитать DXF-файл: {e}")

        # Собираем все точки из modelspace
        points = []
        msp = doc.modelspace()

        for entity in msp:
            dxftype = entity.dxftype()

            if dxftype not in self.SUPPORTED_ENTITIES:
                logger.warning(f"Пропущен неподдерживаемый тип: {dxftype}")
                continue

            try:
                entity_points = self._process_entity(entity)
                points.extend(entity_points)
            except Exception as e:
                logger.error(f"Ошибка обработки {dxftype}: {e}")
                continue

        if not points:
            raise DXFLoadError(
                "В DXF-файле не найдено поддерживаемых примитивов")

        # Объединяем все точки в единый контур
        # (в реальном DXF может быть несколько контуров, но для простоты берём все)
        all_points = np.vstack(points)

        logger.info(f"Загружено {len(all_points)} точек из DXF")
        return all_points

    def _process_entity(self, entity) -> List[np.ndarray]:
        """
        Обрабатывает отдельный примитив DXF.

        Returns:
            Список массивов точек для данного примитива
        """
        dxftype = entity.dxftype()

        if dxftype == 'LINE':
            return self._process_line(entity)
        elif dxftype in ('LWPOLYLINE', 'POLYLINE'):
            return self._process_polyline(entity)
        elif dxftype == 'ARC':
            return self._process_arc(entity)
        elif dxftype == 'CIRCLE':
            return self._process_circle(entity)
        elif dxftype == 'SPLINE':
            return self._process_spline(entity)
        elif dxftype == 'ELLIPSE':
            return self._process_ellipse(entity)
        else:
            raise UnsupportedEntityError(f"Неподдерживаемый тип: {dxftype}")

    def _process_line(self, line) -> List[np.ndarray]:
        """Обрабатывает отрезок."""
        start = line.dxf.start
        end = line.dxf.end
        return [np.array([[start.x, start.y], [end.x, end.y]])]

    def _process_polyline(self, polyline) -> List[np.ndarray]:
        """Обрабатывает полилинию (ломаная линия)."""
        points = []

        if polyline.dxftype() == 'LWPOLYLINE':
            # Для LWPOLYLINE точки хранятся в vertices
            for vertex in polyline.vertices():
                points.append([vertex.dxf.location.x, vertex.dxf.location.y])
        else:
            # Для POLYLINE
            for vertex in polyline.vertices:
                points.append([vertex.dxf.location.x, vertex.dxf.location.y])

        return [np.array(points)]

    def _process_arc(self, arc) -> List[np.ndarray]:
        """
        Обрабатывает дугу, разбивая на отрезки с заданной точностью.
        """
        center = arc.dxf.center
        radius = arc.dxf.radius
        start_angle = arc.dxf.start_angle
        end_angle = arc.dxf.end_angle

        # Переводим углы в радианы
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)

        # Длина дуги
        arc_length = radius * abs(end_rad - start_rad)

        # Количество сегментов для аппроксимации
        num_segments = max(3, int(np.ceil(arc_length / self.tolerance)))

        # Генерируем точки на дуге
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            angle = start_rad + t * (end_rad - start_rad)
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            points.append([x, y])

        return [np.array(points)]

    def _process_circle(self, circle) -> List[np.ndarray]:
        """Обрабатывает окружность как замкнутую дугу."""
        center = circle.dxf.center
        radius = circle.dxf.radius

        # Длина окружности
        circumference = 2 * np.pi * radius
        num_segments = max(8, int(np.ceil(circumference / self.tolerance)))

        points = []
        for i in range(num_segments + 1):
            angle = 2 * np.pi * i / num_segments
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            points.append([x, y])

        return [np.array(points)]

    def _process_spline(self, spline) -> List[np.ndarray]:
        """
        Обрабатывает сплайн, аппроксимируя его отрезками.
        Использует встроенный метод flattening библиотеки ezdxf.
        """
        # Получаем аппроксимированные точки сплайна
        # (метод flattening разбивает сплайн на отрезки с заданной точностью)
        vertices = list(spline.flattening(self.tolerance))

        points = [[v.x, v.y] for v in vertices]
        return [np.array(points)]

    def _process_ellipse(self, ellipse) -> List[np.ndarray]:
        """Обрабатывает эллипс."""
        center = ellipse.dxf.center
        major_axis = ellipse.dxf.major_axis
        ratio = ellipse.dxf.ratio  # отношение малой оси к большой

        # Параметры эллипса
        a = np.linalg.norm([major_axis.x, major_axis.y])  # большая полуось
        b = a * ratio  # малая полуось

        # Угол поворота эллипса
        angle = np.arctan2(major_axis.y, major_axis.x)

        # Длина эллипса (приближённо)
        circumference = np.pi * (
                    3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        num_segments = max(8, int(np.ceil(circumference / self.tolerance)))

        points = []
        for i in range(num_segments + 1):
            t = 2 * np.pi * i / num_segments

            # Точка в локальной системе координат
            x_local = a * np.cos(t)
            y_local = b * np.sin(t)

            # Поворот
            x_rot = x_local * np.cos(angle) - y_local * np.sin(angle)
            y_rot = x_local * np.sin(angle) + y_local * np.cos(angle)

            # Сдвиг в центр
            x = center.x + x_rot
            y = center.y + y_rot

            points.append([x, y])

        return [np.array(points)]


# Функция-обёртка для простого использования
def load_dxf(filepath: Union[str, Path], tolerance: float = 0.1) -> np.ndarray:
    """
    Загружает контур из DXF-файла.

    Args:
        filepath: путь к DXF-файлу
        tolerance: точность дискретизации кривых (мм)

    Returns:
        numpy array с точками контура
    """
    loader = DXFLoader(tolerance=tolerance)
    return loader.load(filepath)
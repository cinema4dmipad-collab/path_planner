import numpy as np
from typing import List
from geometry import Contour
from logger import logger


class PathPlanner:
    """
    Планировщик траектории заливки по принципу "змейка".

    Строит оптимальный путь движения инструмента внутри заданного контура
    с учётом шага и угла заливки.
    """
    def __init__(
            self,
            contour: Contour,
            line_distance: float,
            fill_angle: float = 0,
            tolerance: float = 1e-6
    ):
        """
        Инициализация планировщика.

        Args:
            contour: исходный контур для заливки
            line_distance: расстояние между линиями (шаг заливки)
            fill_angle: угол наклона линий относительно глобальной СК
            tolerance: точность вычислений
        """
        self.original_contour = contour
        self.line_distance = line_distance
        self.fill_angle = fill_angle
        self.tolerance = tolerance
        logger.info(f"Дискретизация контура с tolerance={tolerance} мм")
        discretized = self.original_contour.discretize()
        logger.info(f"Поворот контура на угол {fill_angle}°")
        self.working_contour = discretized.rotate(fill_angle)
        _, _, self.y_min, self.y_max = self.working_contour.bounds
        logger.debug(
            f"Диапазон сканирования по Y: [{self.y_min:.2f}, {self.y_max:.2f}]")

    def generate_path(self) -> np.ndarray:
        """
        Генерирует траекторию змейки.

        Returns:
            Массив точек формы (n, 2) с координатами пути
        """
        logger.info("Начинаем генерацию траектории")
        lines = self._generate_fill_lines()
        logger.debug(f"Сгенерировано {len(lines)} линий")
        if not lines:
            logger.warning("Не удалось сгенерировать ни одной линии")
            return np.array([])
        path = self._connect_lines(lines)
        logger.info(f"Построена траектория из {len(path)} точек")
        if abs(self.fill_angle) > self.tolerance:
            logger.debug(f"Обратный поворот траектории на {-self.fill_angle}°")
            path = self._rotate_back(path)
        return path

    def _generate_fill_lines(self) -> List[np.ndarray]:
        """
        Генерирует горизонтальные линии заливки внутри контура.
        Автоматически подбирает начальный Y для оптимального покрытия.

        Returns:
            Список отрезков [ [точка1, точка2], ... ]
        """
        lines = []

        height = self.y_max - self.y_min
        step = self.line_distance

        print(f"\n🔍 Генерация линий заливки:")
        print(f"   Высота фигуры: {height:.2f} мм")
        print(f"   Шаг: {step:.2f} мм")

        theoretical_lines = max(1, int(height / step) + 1)
        print(f"   Теоретически линий: {theoretical_lines}")

        best_lines = []
        best_count = 0
        best_offset = step / 2

        for offset_factor in [0.3, 0.5, 0.7]:
            y_start = self.y_min + step * offset_factor
            current_lines = []
            y = y_start
            count = 0

            print(
                f"\n   Попытка со смещением {offset_factor:.1f} (y_start={y_start:.2f}):")

            while y <= self.y_max + self.tolerance:
                intersections = self.working_contour.get_intersections(y)

                if len(intersections) >= 2:
                    x_start, x_end = intersections[0], intersections[-1]
                    line = np.array([[x_start, y], [x_end, y]])
                    current_lines.append(line)
                    count += 1
                    print(
                        f"      ✅ y={y:6.2f}: линия от {x_start:6.2f} до {x_end:6.2f}")
                else:
                    print(f"      ❌ y={y:6.2f}: нет пересечений")

                y += step

            print(f"      Получено линий: {count}")

            if count > best_count:
                best_count = count
                best_lines = current_lines
                best_offset = offset_factor

        print(f"\n📊 Итог:")
        print(f"   Лучшее смещение: {best_offset:.1f}")
        print(f"   Сгенерировано линий: {best_count}")
        print(f"   Ожидалось: {theoretical_lines}")

        if best_count == 0 and height > 0:
            print("⚠️ Не удалось получить линии, пробуем центральную")
            y_center = (self.y_min + self.y_max) / 2
            intersections = self.working_contour.get_intersections(y_center)

            if len(intersections) >= 2:
                x_start, x_end = intersections[0], intersections[-1]
                line = np.array([[x_start, y_center], [x_end, y_center]])
                best_lines = [line]
                print(
                    f"   ✅ Центральная линия: y={y_center:.2f}, X от {x_start:.2f} до {x_end:.2f}")

        return best_lines

    def _connect_lines(self, lines: List[np.ndarray]) -> np.ndarray:
        """
        Соединяет линии в непрерывную змейку.

        Args:
            lines: список отрезков [ [точка1, точка2], ... ]

        Returns:
            Массив точек траектории
        """
        if not lines:
            return np.array([])

        path_points = []

        for i, line in enumerate(lines):
            if i % 2 == 0:
                path_points.append(line[0])
                path_points.append(line[1])
                logger.debug(f"Линия {i}: направление →")
            else:
                path_points.append(line[1])
                path_points.append(line[0])
                logger.debug(f"Линия {i}: направление ←")
        return np.array(path_points)

    def _rotate_back(self, path: np.ndarray) -> np.ndarray:
        """Поворачивает траекторию обратно в исходную СК."""
        angle_rad = np.radians(-self.fill_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        return path @ rotation_matrix.T

    def save_path(self, filepath: str, format: str = 'txt') -> None:
        """
        Сохраняет траекторию в файл.

        Args:
            filepath: путь для сохранения
            format: формат ('txt' или 'csv')
        """
        path = self.generate_path()
        if format == 'txt':
            np.savetxt(filepath, path, fmt='%.6f', delimiter=' ')
            logger.info(f"Траектория сохранена в {filepath}")
        elif format == 'csv':
            np.savetxt(filepath, path, fmt='%.6f', delimiter=',')
            logger.info(f"Траектория сохранена в {filepath}")
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

    def get_statistics(self) -> dict:
        """
        Возвращает статистику по построенной траектории.

        Returns:
            Словарь с метриками траектории
        """
        path = self.generate_path()

        if len(path) == 0:
            return {
                'total_points': 0,
                'total_length': 0.0,
                'num_lines': 0,
                'y_range': (self.y_min, self.y_max)
            }
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(path[i + 1] - path[i])
        return {
            'total_points': len(path),
            'total_length': total_length,
            'num_lines': len(self._generate_fill_lines()),
            'y_range': (self.y_min, self.y_max),
            'line_distance': self.line_distance,
            'fill_angle': self.fill_angle
        }
import numpy as np
from typing import List, Tuple, Optional
from logger import logger


class Contour:
    """
    Замкнутый контур, заданный массивом точек.
    Поддерживает дискретизацию кривых с контролем точности.
    """

    MODE_AS_IS = "as_is"
    MODE_LESS = "less"
    MODE_MORE = "more"

    def __init__(self, points: np.ndarray,
                 approximation_tolerance: float = 0.1):
        """
        Args:
            points: массив точек формы (n, 2)
            approximation_tolerance: максимальное отклонение аппроксимации (мм)
                                   Меньше значение = точнее, но больше точек
        """

        self.points = points
        self.approximation_tolerance = approximation_tolerance
        self.points = self.ensure_closed().points
        self._validate()
        logger.info(
            f"Создан контур с {len(points)} точками, tolerance={approximation_tolerance}")

    def _validate(self):
        """Проверяет корректность контура."""
        if self.points.shape[1] != 2:
            raise ValueError("Контур должен содержать 2D точки")
        if len(self.points) < 3:
            raise ValueError("Контур должен содержать минимум 3 точки")


    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Возвращает границы контура (xmin, xmax, ymin, ymax)."""
        x = self.points[:, 0]
        y = self.points[:, 1]
        result = (x.min(), x.max(), y.min(), y.max())
        logger.debug(
            "bounds: x[%.2f, %.2f], y[%.2f, %.2f]",
            result[0], result[1], result[2], result[3],
        )
        return result

    def perimeter(self) -> float:
        """Вычисляет периметр контура."""
        perim = 0.0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            perim += np.linalg.norm(p2 - p1)
        return perim

    def discretize(self, mode: str = "as_is") -> 'Contour':
        """
        Дискретизирует контур с заданным режимом.

        Args:
            mode: режим дискретизации
                - "as_is": без изменений
                - "less": уменьшить количество точек (упростить)
                - "more": увеличить количество точек (уплотнить)

        Returns:
            Новый контур с дискретизированными точками
        """
        logger.info(f"Дискретизация контура в режиме: {mode}")

        if mode == self.MODE_AS_IS:
            return Contour(self.points.copy(), self.approximation_tolerance)

        elif mode == self.MODE_LESS:
            # Упрощаем с меньшим epsilon для сохранения формы
            simplified = self._simplify_rdp(
                epsilon=self.approximation_tolerance * 0.3)
            logger.info(
                f"Упрощение: {len(self.points)} -> {len(simplified)} точек")
            return Contour(simplified, self.approximation_tolerance)

        elif mode == self.MODE_MORE:
            # Уплотняем до целевого расстояния
            densified = self._densify(
                target_distance=self.approximation_tolerance * 2)
            logger.info(
                f"Уплотнение: {len(self.points)} -> {len(densified)} точек")
            return Contour(densified, self.approximation_tolerance)

        else:
            raise ValueError(f"Неизвестный режим дискретизации: {mode}")

    def _densify(self, target_distance: float) -> np.ndarray:
        """
        Уплотняет контур, добавляя точки.

        Args:
            target_distance: максимальное расстояние между точками

        Returns:
            новый массив с добавленными точками
        """
        if len(self.points) < 2:
            return self.points.copy()

        new_points = []

        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]

            new_points.append(p1)

            distance = np.linalg.norm(p2 - p1)

            if distance > target_distance:
                num_new = int(np.ceil(distance / target_distance))

                for j in range(1, num_new):
                    t = j / num_new
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    new_points.append([x, y])

        return np.array(new_points)

    def _simplify_rdp(self, epsilon: float) -> np.ndarray:
        """
        Упрощает контур алгоритмом Рамера-Дугласа-Пекера.

        Args:
            epsilon: максимальное отклонение

        Returns:
            упрощённый массив точек
        """
        if len(self.points) < 3:
            return self.points.copy()

        def perpendicular_distance(point, line_start, line_end):
            """Расстояние от точки до прямой."""
            line_vec = line_end - line_start
            line_len = np.linalg.norm(line_vec)

            if line_len == 0:
                return np.linalg.norm(point - line_start)

            point_vec = point - line_start
            projection = np.dot(point_vec, line_vec) / line_len

            if projection < 0:
                return np.linalg.norm(point - line_start)
            elif projection > line_len:
                return np.linalg.norm(point - line_end)
            else:
                proj_point = line_start + (projection / line_len) * line_vec
                return np.linalg.norm(point - proj_point)

        def rdp_recursive(pts, start_idx, end_idx):
            """Рекурсивная часть алгоритма."""
            if start_idx >= end_idx - 1:
                return [start_idx, end_idx]

            start_point = pts[start_idx]
            end_point = pts[end_idx]

            max_dist = 0
            max_idx = start_idx

            for i in range(start_idx + 1, end_idx):
                dist = perpendicular_distance(pts[i], start_point, end_point)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            if max_dist > epsilon:
                left = rdp_recursive(pts, start_idx, max_idx)
                right = rdp_recursive(pts, max_idx, end_idx)
                return left[:-1] + right
            else:
                return [start_idx, end_idx]

        indices = sorted(
            set(rdp_recursive(self.points, 0, len(self.points) - 1)))
        return self.points[indices]

    def get_intersections(self, y_level: float) -> List[float]:
        """
        Находит пересечения горизонтальной линии с контуром.
        """
        intersections = []

        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]

            x = self._segment_intersection(p1, p2, y_level)
            if x is not None:
                intersections.append(x)

        intersections.sort()
        return self._remove_close(intersections)

    @staticmethod
    def _segment_intersection(p1: np.ndarray, p2: np.ndarray, y: float) -> \
    Optional[float]:
        """Находит пересечение отрезка с горизонтальной линией."""
        y1, y2 = p1[1], p2[1]

        if (y1 - y) * (y2 - y) > 0:
            return None
        if abs(y2 - y1) < 1e-10:
            return None

        t = (y - y1) / (y2 - y1)
        return p1[0] + t * (p2[0] - p1[0])

    @staticmethod
    def _line_line_intersection(
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Пересечение двух прямых (бесконечных), заданных отрезками."""
        x1, y1 = float(a1[0]), float(a1[1])
        x2, y2 = float(a2[0]), float(a2[1])
        x3, y3 = float(b1[0]), float(b1[1])
        x4, y4 = float(b2[0]), float(b2[1])
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return None
        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denom
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denom
        return np.array([px, py], dtype=float)

    def offset_outward(self, distance: float) -> "Contour":
        """
        Расширяет контур наружу на distance мм.

        Используется для зоны безопасности вокруг внутренних контуров (отверстий).
        """
        if distance <= 0:
            return Contour(self.points.copy(), self.approximation_tolerance)

        pts = self.points.copy()
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return Contour(self.points.copy(), self.approximation_tolerance)

        centroid = pts.mean(axis=0)
        offset_edges: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            edge = p2 - p1
            length = float(np.linalg.norm(edge))
            if length <= 1e-12:
                continue
            direction = edge / length
            normal = np.array([-direction[1], direction[0]])
            mid = (p1 + p2) / 2.0
            if np.dot(normal, centroid - mid) > 0:
                normal = -normal
            offset_edges.append((p1 + normal * distance, p2 + normal * distance))

        if len(offset_edges) < 3:
            return Contour(self.points.copy(), self.approximation_tolerance)

        new_points: List[np.ndarray] = []
        edge_count = len(offset_edges)
        for i in range(edge_count):
            a1, a2 = offset_edges[i]
            b1, b2 = offset_edges[(i + 1) % edge_count]
            point = self._line_line_intersection(a1, a2, b1, b2)
            if point is not None:
                new_points.append(point)

        if len(new_points) < 3:
            return Contour(self.points.copy(), self.approximation_tolerance)

        logger.debug(
            "offset_outward %.3f мм: %d -> %d точек",
            distance,
            len(pts),
            len(new_points),
        )
        return Contour(np.array(new_points), self.approximation_tolerance)

    def rotate(self, angle_deg: float) -> 'Contour':
        """
        Поворачивает контур на заданный угол.

        Args:
            angle_deg: угол поворота в градусах

        Returns:
            Новый повёрнутый контур
        """
        angle_rad = np.radians(angle_deg)

        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        rotated_points = self.points @ rotation_matrix.T

        return Contour(rotated_points, self.approximation_tolerance)

    def ensure_closed(self, tolerance: float = 1e-6):
        """
        Замыкает контур, изменяя текущий объект (не создаёт новый)
        """
        first = self.points[0]
        last = self.points[-1]
        distance = np.linalg.norm(last - first)

        if distance > tolerance:
            logger.debug("Контур не замкнут, добавляем первую точку")
            self.points = np.vstack([self.points, first])
        else:
            logger.debug("Контур уже замкнут")

        return self

    @staticmethod
    def _remove_close(xs: List[float], eps: float = 1e-6) -> List[float]:
        """Удаляет слишком близкие точки."""
        if not xs:
            return xs
        result = [xs[0]]
        for x in xs[1:]:
            if abs(x - result[-1]) > eps:
                result.append(x)
        return result
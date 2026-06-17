import numpy as np
import logging
from typing import List, Optional, Sequence, Tuple

from contour_nesting import point_in_polygon
from exceptions import BridgePlanningError
from geometry import Contour
from logger import logger

# Геометрический epsilon для проверок отверстий и интервалов (не tolerance дискретизации).
GEOM_EPSILON = 1e-3


def segments_from_intersections(intersections: List[float]) -> List[Tuple[float, float]]:
    """Преобразует отсортированные пересечения в пары внутренних отрезков."""
    segments: List[Tuple[float, float]] = []
    for i in range(0, len(intersections) - 1, 2):
        if i + 1 < len(intersections):
            segments.append((intersections[i], intersections[i + 1]))
    return segments


def subtract_interval(
    start: float,
    end: float,
    cut_start: float,
    cut_end: float,
    tolerance: float = GEOM_EPSILON,
) -> List[Tuple[float, float]]:
    """Вычитает [cut_start, cut_end] из [start, end]."""
    if cut_end <= start + tolerance or cut_start >= end - tolerance:
        return [(start, end)]
    if cut_start <= start + tolerance and cut_end >= end - tolerance:
        return []

    parts: List[Tuple[float, float]] = []
    if start + tolerance < cut_start:
        parts.append((start, cut_start))
    if cut_end < end - tolerance:
        parts.append((cut_end, end))
    return parts


def fill_segments_at_y(
    contour: Contour,
    y_level: float,
    holes: Optional[List[Contour]] = None,
    tolerance: float = GEOM_EPSILON,
    *,
    endpoint_inset: float = 0.0,
    log_details: bool = False,
) -> List[Tuple[float, float]]:
    """
    Возвращает отрезки заливки на уровне Y с учётом отверстий.

    Чистая геометрическая операция: outer intersections минус hole intervals.
    """
    _ = endpoint_inset  # Deprecated compatibility arg: fill reaches real boundaries.
    outer_segments = segments_from_intersections(contour.get_intersections(y_level))
    if log_details:
        logger.debug("y=%.2f outer intervals: %s", y_level, outer_segments)

    if not holes:
        return outer_segments

    result = outer_segments
    for hole_idx, hole in enumerate(holes):
        hole_segments = segments_from_intersections(hole.get_intersections(y_level))
        if log_details and hole_segments:
            logger.debug(
                "y=%.2f hole #%d intervals: %s",
                y_level,
                hole_idx + 1,
                hole_segments,
            )
        updated: List[Tuple[float, float]] = []
        for seg_start, seg_end in result:
            parts = [(seg_start, seg_end)]
            for hole_start, hole_end in hole_segments:
                next_parts: List[Tuple[float, float]] = []
                for part_start, part_end in parts:
                    next_parts.extend(
                        subtract_interval(
                            part_start,
                            part_end,
                            hole_start,
                            hole_end,
                            tolerance=tolerance,
                        )
                    )
                parts = next_parts
            updated.extend(parts)
        result = updated

    final = [
        (start, end)
        for start, end in result
        if end - start > tolerance
    ]
    if log_details:
        logger.debug("y=%.2f fill segments: %s", y_level, final)
    return final


class PathPlanner:
    """
    Планировщик траектории заливки по принципу "змейка".

    Строит оптимальный путь движения инструмента внутри заданного контура
    с учётом шага, угла заливки и отверстий.
    """

    def __init__(
        self,
        contour: Contour,
        line_distance: float,
        fill_angle: float = 0,
        tolerance: float = 1e-6,
        holes: Optional[List[Contour]] = None,
        hole_clearance: float = 0.0,
        allow_clearance_contact: bool = True,
    ):
        self.original_contour = contour
        self.line_distance = line_distance
        self.fill_angle = fill_angle
        self.tolerance = tolerance
        _ = (hole_clearance, allow_clearance_contact)
        self.hole_clearance = 0.0
        self.allow_clearance_contact = True
        self.geom_epsilon = GEOM_EPSILON
        self.holes = holes or []
        self.path_segments: List[Tuple[int, int, str]] = []
        self.skipped_primary_segments = 0
        self.skipped_deferred_segments = 0
        self.disconnected_deferred_segments = 0
        self._fill_lines_cache: Optional[List[np.ndarray]] = None
        self._path_cache: Optional[np.ndarray] = None
        self._bridge_route_cache: dict[
            Tuple[float, float, float, float, Optional[float], Optional[float]],
            np.ndarray,
        ] = {}
        self._bridge_failed_cache: set[
            Tuple[float, float, float, float, Optional[float], Optional[float]]
        ] = set()
        logger.info(f"Поворот контура на угол {fill_angle}°")
        self.working_contour = self.original_contour.rotate(fill_angle)
        self.working_holes = [
            hole.rotate(fill_angle) for hole in self.holes
        ]
        self.planning_holes = self.working_holes
        if self.working_holes:
            logger.info("Учтено отверстий: %d", len(self.working_holes))
        self.outer_bounds = self.working_contour.bounds
        _, _, self.y_min, self.y_max = self.outer_bounds
        self.hole_bounds = [self._hole_bounds(hole) for hole in self.planning_holes]
        logger.debug(
            "Диапазон сканирования по Y: [%.2f, %.2f]",
            self.y_min,
            self.y_max,
        )

    def _row_y_tolerance(self) -> float:
        """Допуск группировки строк заливки — от шага, не от tolerance дискретизации."""
        return max(self.geom_epsilon, self.line_distance * 0.25)

    def _fill_endpoint_inset(self) -> float:
        """Deprecated compatibility hook; clearance endpoint insets are disabled."""
        return 0.0

    def generate_path(self) -> np.ndarray:
        """Генерирует траекторию змейки."""
        if self._path_cache is not None:
            return self._path_cache.copy()

        logger.info("Начинаем генерацию траектории")
        self.skipped_primary_segments = 0
        self.skipped_deferred_segments = 0
        self.disconnected_deferred_segments = 0
        self._bridge_route_cache.clear()
        self._bridge_failed_cache.clear()
        lines = self._generate_fill_lines()
        logger.debug("Сгенерировано %d линий", len(lines))
        if not lines:
            logger.warning("Не удалось сгенерировать ни одной линии")
            self.path_segments = []
            return np.array([])

        if self.working_holes:
            primary_lines, deferred_lines = self._split_primary_deferred(lines)
            logger.info(
                "Двухфазная заливка: основная змейка %d линий, "
                "дозаполнение %d пропусков",
                len(primary_lines),
                len(deferred_lines),
            )
            primary_path: np.ndarray = np.array([])
            primary_segments: List[Tuple[int, int, str]] = []
            skipped_primary = 0
            try:
                primary_path, primary_segments = self._connect_lines_primary(
                    primary_lines
                )
            except BridgePlanningError as exc:
                logger.warning(
                    "Основная змейка прервана (%s), продолжаем через best-effort",
                    exc,
                )
                (
                    primary_path,
                    primary_segments,
                    skipped_primary,
                    skipped_primary_lines,
                ) = self._connect_lines_primary_best_effort(primary_lines)
                deferred_lines = deferred_lines + skipped_primary_lines

            self.skipped_primary_segments = skipped_primary
            if skipped_primary:
                logger.warning(
                    "Пропущено %d основных сегментов: перенесены в дозаполнение",
                    skipped_primary,
                )

            path = primary_path
            self.path_segments = primary_segments
            if deferred_lines:
                path, self.path_segments = self._append_deferred_pass(
                    path, self.path_segments, deferred_lines
                )
                if self.skipped_deferred_segments:
                    logger.warning(
                        "Пропущено %d сегментов дозаполнения: "
                        "безопасный переход не найден",
                        self.skipped_deferred_segments,
                    )
        else:
            path, self.path_segments = self._connect_lines(lines)

        logger.info("Построена траектория из %d точек", len(path))
        if abs(self.fill_angle) > self.tolerance:
            logger.debug("Обратный поворот траектории на %s°", -self.fill_angle)
            path = self._rotate_back(path)
        self._path_cache = path.copy()
        return path

    def get_path_segments(self) -> Tuple[Tuple[int, int, str], ...]:
        """Индексы точек траектории: fill или travel."""
        return tuple(self.path_segments)

    def _generate_fill_lines(self) -> List[np.ndarray]:
        """Генерирует горизонтальные линии заливки внутри контура."""
        if self._fill_lines_cache is not None:
            return self._fill_lines_cache

        height = self.y_max - self.y_min
        step = self.line_distance
        log_details = bool(self.working_holes) and logger.isEnabledFor(logging.DEBUG)

        logger.debug("Генерация линий заливки: height=%.2f step=%.2f", height, step)

        theoretical_lines = max(1, int(height / step) + 1)
        best_lines: List[np.ndarray] = []
        best_count = 0
        best_offset = step / 2

        for offset_factor in [0.3, 0.5, 0.7]:
            y_start = self.y_min + step * offset_factor
            current_lines: List[np.ndarray] = []
            y = y_start
            count = 0

            while y <= self.y_max + self.tolerance:
                segments = fill_segments_at_y(
                    self.working_contour,
                    y,
                    self.planning_holes,
                    tolerance=self.geom_epsilon,
                    log_details=log_details,
                )

                for x_start, x_end in segments:
                    current_lines.append(np.array([[x_start, y], [x_end, y]]))
                    count += 1

                y += step

            if count > best_count:
                best_count = count
                best_lines = current_lines
                best_offset = offset_factor

        logger.debug(
            "Итог: offset=%.1f lines=%d expected=%d",
            best_offset,
            best_count,
            theoretical_lines,
        )

        if best_count == 0 and height > 0:
            logger.warning("Не удалось получить линии, пробуем центральную")
            y_center = (self.y_min + self.y_max) / 2
            segments = fill_segments_at_y(
                self.working_contour,
                y_center,
                self.planning_holes,
                tolerance=self.geom_epsilon,
                log_details=log_details,
            )
            for x_start, x_end in segments:
                best_lines.append(np.array([[x_start, y_center], [x_end, y_center]]))

        self._fill_lines_cache = best_lines
        return best_lines

    def _group_lines_by_y(
        self, lines: List[np.ndarray]
    ) -> List[Tuple[float, List[np.ndarray]]]:
        """Группирует отрезки заливки по уровню Y."""
        if not lines:
            return []

        sorted_lines = sorted(lines, key=lambda line: (line[0, 1], line[0, 0]))
        rows: List[Tuple[float, List[np.ndarray]]] = []
        current_y: Optional[float] = None
        current_row: List[np.ndarray] = []

        for line in sorted_lines:
            y = float(line[0, 1])
            if current_y is None or abs(y - current_y) > self._row_y_tolerance():
                if current_row:
                    rows.append((current_y, current_row))
                current_row = [line]
                current_y = y
            else:
                current_row.append(line)

        if current_row and current_y is not None:
            rows.append((current_y, current_row))

        return rows

    def _segment_endpoints(self, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Левый и правый концы отрезка."""
        p1, p2 = line[0], line[1]
        if p1[0] <= p2[0]:
            return p1.copy(), p2.copy()
        return p2.copy(), p1.copy()

    def _orient_segment(
        self, line: np.ndarray, left_to_right: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает начало и конец отрезка в нужном направлении."""
        left, right = self._segment_endpoints(line)
        if left_to_right:
            return left, right
        return right, left

    def _hole_bounds(self, hole: Contour) -> Tuple[float, float, float, float]:
        x = hole.points[:, 0]
        y = hole.points[:, 1]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    def _route_offset(self) -> float:
        return max(
            self.geom_epsilon * 10,
            self.line_distance * 0.05,
        )

    def _horizontal_segment_crosses_holes(
        self, x_start: float, x_end: float, y: float
    ) -> bool:
        x_lo = min(float(x_start), float(x_end))
        x_hi = max(float(x_start), float(x_end))
        return self._polyline_crosses_hole_interior(
            np.array([[x_lo, y], [x_hi, y]])
        )

    def _horizontal_segment_inside_outer(
        self, x_start: float, x_end: float, y: float
    ) -> bool:
        """True, если вся горизонтальная хорда лежит внутри или на границе контура."""
        x_lo = min(float(x_start), float(x_end))
        x_hi = max(float(x_start), float(x_end))
        length = x_hi - x_lo
        sample_count = max(5, int(length / max(self.line_distance, self.geom_epsilon)) + 1)
        for i in range(sample_count + 1):
            t = i / sample_count
            x = x_lo + t * (x_hi - x_lo)
            if not self._point_inside_or_on_outer(np.array([x, y])):
                return False
        return True

    def _horizontal_segment_is_valid_bridge(
        self, x_start: float, x_end: float, y: float
    ) -> bool:
        if self._horizontal_segment_crosses_holes(x_start, x_end, y):
            return False
        return self._horizontal_segment_inside_outer(x_start, x_end, y)

    def _enumerate_safe_route_ys(
        self,
        x_start: float,
        x_end: float,
        *,
        extra: Optional[List[float]] = None,
    ) -> List[float]:
        """Уровни Y, где горизонтальный обход между x_start и x_end безопасен."""
        x_lo = min(float(x_start), float(x_end))
        x_hi = max(float(x_start), float(x_end))
        candidates: List[float] = list(extra or [])

        offset = self._route_offset()
        span = float(self.y_max) - float(self.y_min) + 2.0 * offset
        base = float(self.y_min) - offset
        sample_count = max(16, int(span / max(self.line_distance, self.geom_epsilon)) + 1)
        for i in range(sample_count + 1):
            candidates.append(base + span * i / sample_count)

        unique: List[float] = []
        for value in candidates:
            if any(abs(value - existing) <= self.geom_epsilon for existing in unique):
                continue
            if self._horizontal_segment_is_valid_bridge(x_lo, x_hi, value):
                unique.append(value)
        return unique

    def _fill_x_extents_at_y(self, y: float) -> List[Tuple[float, float]]:
        """Интервалы заливки на уровне Y без inset — для построения обходов."""
        return fill_segments_at_y(
            self.working_contour,
            y,
            self.planning_holes,
            tolerance=self.geom_epsilon,
        )

    def _wide_detour_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> List[np.ndarray]:
        """Широкий обход через края интервалов заливки и bbox отверстий."""
        y_start = float(start[1])
        y_end = float(end[1])
        offset = self._route_offset()
        bypass_x_values: List[float] = []

        for y in {y_start, y_end}:
            for x0, x1 in self._fill_x_extents_at_y(y):
                bypass_x_values.extend([float(x0), float(x1)])

        for hole in self.planning_holes:
            xmin, xmax, _, _ = self._hole_bounds(hole)
            bypass_x_values.extend([xmin - offset, xmax + offset])

        x_lo = min(float(start[0]), float(end[0]))
        x_hi = max(float(start[0]), float(end[0]))
        safe_route_y_values = self._enumerate_safe_route_ys(x_lo, x_hi)
        if not safe_route_y_values:
            route_y_candidates: List[float] = list(
                self._combined_route_y_levels(
                    self.planning_holes,
                    current_y=y_start,
                    next_y=y_end,
                )
            )
            span = float(self.y_max) - float(self.y_min)
            sample_count = max(12, int(span / max(self.line_distance, self.geom_epsilon)))
            for i in range(sample_count + 1):
                route_y_candidates.append(self.y_min + span * i / sample_count)
            unique_route_y: List[float] = []
            for value in route_y_candidates:
                if not any(
                    abs(value - existing) <= self.geom_epsilon
                    for existing in unique_route_y
                ):
                    unique_route_y.append(value)
            safe_route_y_values = unique_route_y

        routes: List[np.ndarray] = []
        unique_bypass: List[float] = []
        for value in bypass_x_values:
            if not any(abs(value - existing) <= self.geom_epsilon for existing in unique_bypass):
                unique_bypass.append(value)

        for bypass_x in unique_bypass:
            if abs(y_start - y_end) <= self.geom_epsilon:
                routes.append(np.array([start, [bypass_x, y_start], end]))
            else:
                routes.append(
                    np.array([
                        start,
                        [bypass_x, y_start],
                        [bypass_x, y_end],
                        end,
                    ])
                )
                routes.append(
                    np.array([
                        start,
                        [start[0], y_start],
                        [bypass_x, y_start],
                        [bypass_x, y_end],
                        [end[0], y_end],
                        end,
                    ])
                )
            for route_y in safe_route_y_values:
                routes.append(
                    np.array([
                        start,
                        [bypass_x, y_start],
                        [bypass_x, route_y],
                        [end[0], route_y],
                        end,
                    ])
                )
                if abs(y_start - y_end) > self.geom_epsilon:
                    routes.append(
                        np.array([
                            start,
                            [start[0], route_y],
                            [bypass_x, route_y],
                            [bypass_x, y_end],
                            end,
                        ])
                    )
        return routes

    def _find_boundary_safe_route_y(
        self,
        x_start: float,
        x_end: float,
        *,
        direction: str,
        reference_y: float,
    ) -> Optional[float]:
        """Минимальный безопасный Y выше/ниже reference_y для всей хорды x_start..x_end."""
        offset = self._route_offset()
        safe_y_values = self._enumerate_safe_route_ys(x_start, x_end)
        if not safe_y_values:
            return None

        ref = float(reference_y)
        if direction == "above":
            above = [
                y for y in safe_y_values if y > ref + self.geom_epsilon
            ]
            if not above:
                return None
            return min(above) + offset

        if direction == "below":
            below = [
                y for y in safe_y_values if y < ref - self.geom_epsilon
            ]
            if not below:
                return None
            return max(below) - offset

        raise ValueError(f"Неизвестное направление обхода: {direction}")

    def _boundary_route_ys(
        self,
        x_start: float,
        x_end: float,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> List[float]:
        """Безопасные уровни обхода по реальной ширине разрыва между x_start и x_end."""
        if current_y is None:
            reference_y = float(next_y or 0.0)
        else:
            reference_y = float(current_y)

        safe_y_values = self._enumerate_safe_route_ys(x_start, x_end)
        if not safe_y_values:
            return []

        above = [
            y for y in safe_y_values if y > reference_y + self.geom_epsilon
        ]
        below = [
            y for y in safe_y_values if y < reference_y - self.geom_epsilon
        ]
        candidates: List[float] = []
        if above:
            candidates.append(min(above))
        if below:
            candidates.append(max(below))

        if current_y is not None and next_y is not None:
            if next_y > current_y + self.geom_epsilon and below:
                candidates.insert(0, max(below))
            elif next_y < current_y - self.geom_epsilon and above:
                candidates.insert(0, min(above))

        unique: List[float] = []
        for value in candidates:
            if not any(abs(value - existing) <= self.geom_epsilon for existing in unique):
                unique.append(value)
        return unique

    def _point_in_hole_interior(
        self, point: np.ndarray, hole: Contour, margin: Optional[float] = None
    ) -> bool:
        margin = self.geom_epsilon if margin is None else margin
        if self._point_on_polygon_boundary(point, hole.points):
            return False
        xmin, xmax, ymin, ymax = self._hole_bounds(hole)
        if not (
            xmin + margin < point[0] < xmax - margin
            and ymin + margin < point[1] < ymax - margin
        ):
            return False
        return point_in_polygon(point, hole.points)

    def _snap_bridge_endpoint(self, point: np.ndarray) -> np.ndarray:
        """Сдвигает точку с внутренности отверстия на ближайшую вертикальную грань."""
        x = float(point[0])
        y = float(point[1])
        for hole in self.planning_holes:
            xmin, xmax, ymin, ymax = self._hole_bounds(hole)
            if not (
                xmin - self.geom_epsilon <= x <= xmax + self.geom_epsilon
                and ymin - self.geom_epsilon <= y <= ymax + self.geom_epsilon
            ):
                continue
            if self._point_on_polygon_boundary(point, hole.points):
                return point
            snap_x = xmin if abs(x - xmin) <= abs(x - xmax) else xmax
            return np.array([snap_x, y], dtype=float)
        return np.array(point, dtype=float)

    def _point_violates_clearance(
        self, point: np.ndarray, hole: Contour
    ) -> bool:
        """Deprecated compatibility hook; only real hole interiors are forbidden."""
        return self._point_in_hole_interior(point, hole)

    def _point_on_segment(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> bool:
        tolerance = self.geom_epsilon if tolerance is None else tolerance
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= tolerance * tolerance:
            return float(np.linalg.norm(point - start)) <= tolerance

        t = float(np.dot(point - start, segment) / length_sq)
        if t < -tolerance or t > 1.0 + tolerance:
            return False
        projection = start + np.clip(t, 0.0, 1.0) * segment
        return float(np.linalg.norm(point - projection)) <= tolerance

    def _point_on_polygon_boundary(
        self,
        point: np.ndarray,
        polygon: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> bool:
        tolerance = self.geom_epsilon if tolerance is None else tolerance
        for idx in range(len(polygon) - 1):
            if self._point_on_segment(point, polygon[idx], polygon[idx + 1], tolerance):
                return True
        if len(polygon) > 2:
            return self._point_on_segment(point, polygon[-1], polygon[0], tolerance)
        return False

    def _point_inside_or_on_outer(self, point: np.ndarray) -> bool:
        xmin, xmax, ymin, ymax = self.outer_bounds
        if (
            point[0] < xmin - self.geom_epsilon
            or point[0] > xmax + self.geom_epsilon
            or point[1] < ymin - self.geom_epsilon
            or point[1] > ymax + self.geom_epsilon
        ):
            return False
        if point_in_polygon(point, self.working_contour.points):
            return True
        return self._point_on_polygon_boundary(point, self.working_contour.points)

    def _segment_on_fill_line(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """True, если отрезок лежит на одном из интервалов заливки."""
        if abs(float(start[1]) - float(end[1])) > self.geom_epsilon:
            return False
        y = float(start[1])
        x_lo = min(float(start[0]), float(end[0]))
        x_hi = max(float(start[0]), float(end[0]))
        for x0, x1 in self._fill_x_extents_at_y(y):
            seg_lo = min(float(x0), float(x1))
            seg_hi = max(float(x0), float(x1))
            if (
                x_lo >= seg_lo - self.geom_epsilon
                and x_hi <= seg_hi + self.geom_epsilon
            ):
                return True
        return False

    def _point_on_travel_surface(self, point: np.ndarray) -> bool:
        """Точка на стенке контура, стенке hole или в зоне заливки."""
        if self._point_on_polygon_boundary(point, self.working_contour.points):
            return True
        for hole in self.planning_holes:
            if self._point_on_polygon_boundary(point, hole.points):
                return True
        x = float(point[0])
        y = float(point[1])
        for x0, x1 in self._fill_x_extents_at_y(y):
            seg_lo = min(float(x0), float(x1))
            seg_hi = max(float(x0), float(x1))
            if seg_lo - self.geom_epsilon <= x <= seg_hi + self.geom_epsilon:
                return True
        return False

    def _segment_in_travel_zone(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Отрезок не пересекает hole и идёт только по стенке или материалу заливки."""
        if self._segment_crosses_hole_interior(start, end):
            return False
        length = float(np.linalg.norm(end - start))
        if length <= self.geom_epsilon:
            return self._point_on_travel_surface(start)
        sample_count = max(
            5, int(length / max(self.line_distance, self.geom_epsilon)) + 1
        )
        for i in range(sample_count + 1):
            t = i / sample_count
            point = start + t * (end - start)
            if not self._point_on_travel_surface(point):
                return False
        return True

    def _crossing_holes(
        self, start: np.ndarray, end: np.ndarray
    ) -> List[int]:
        """Индексы отверстий, которые пересекает отрезок."""
        if self._segment_on_fill_line(start, end):
            return []

        crossed: List[int] = []
        for idx, hole in enumerate(self.planning_holes):
            xmin, xmax, ymin, ymax = self.hole_bounds[idx]
            seg_xmin = min(float(start[0]), float(end[0]))
            seg_xmax = max(float(start[0]), float(end[0]))
            seg_ymin = min(float(start[1]), float(end[1]))
            seg_ymax = max(float(start[1]), float(end[1]))
            if (
                seg_xmax <= xmin + self.geom_epsilon
                or seg_xmin >= xmax - self.geom_epsilon
                or seg_ymax <= ymin + self.geom_epsilon
                or seg_ymin >= ymax - self.geom_epsilon
            ):
                continue

            for i in range(1, 12):
                t = i / 12.0
                sample = start + t * (end - start)
                if self._point_in_hole_interior(sample, hole):
                    crossed.append(idx)
                    break
        return crossed

    def _segment_crosses_hole_interior(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        return bool(self._crossing_holes(start, end))

    def _combined_route_y_levels(
        self,
        holes: Sequence[Contour],
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> List[float]:
        """Общие уровни обхода для нескольких отверстий."""
        if not holes:
            return []

        offset = self._route_offset()
        ymin = min(self._hole_bounds(hole)[2] for hole in holes)
        ymax = max(self._hole_bounds(hole)[3] for hole in holes)
        outside_above = ymin - offset
        outside_below = ymax + offset
        candidates = [outside_above, outside_below, ymin - offset * 0.5, ymax + offset * 0.5]

        if current_y is not None and next_y is not None:
            if next_y > current_y + self.geom_epsilon:
                candidates.insert(0, outside_below)
            elif next_y < current_y - self.geom_epsilon:
                candidates.insert(0, outside_above)

        unique: List[float] = []
        for value in candidates:
            if not any(abs(value - existing) <= self.geom_epsilon for existing in unique):
                unique.append(value)
        return unique

    def _route_y_candidates(
        self,
        start: np.ndarray,
        end: np.ndarray,
        hole: Contour,
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> List[float]:
        """Подбирает уровни Y для обхода отверстия снаружи."""
        _, _, ymin, ymax = self._hole_bounds(hole)
        offset = self._route_offset()
        outside_above = ymin - offset
        outside_below = ymax + offset
        candidates: List[float] = list(
            self._boundary_route_ys(
                float(start[0]),
                float(end[0]),
                current_y,
                next_y,
            )
        )

        if current_y is not None:
            if next_y is not None and next_y > current_y + self.geom_epsilon:
                candidates.extend([outside_below, ymax + offset * 0.5])
            elif next_y is not None and next_y < current_y - self.geom_epsilon:
                candidates.extend([outside_above, ymin - offset * 0.5])
            elif current_y >= ymax - self.geom_epsilon:
                candidates.append(outside_below)
            elif current_y <= ymin + self.geom_epsilon:
                candidates.append(outside_above)
            elif current_y >= (ymin + ymax) / 2 - self.geom_epsilon:
                candidates.extend([outside_below, outside_above])
            else:
                candidates.extend([outside_above, outside_below])
        else:
            candidates.extend([outside_above, outside_below])

        unique: List[float] = []
        for value in candidates:
            if not any(abs(value - existing) <= self.geom_epsilon for existing in unique):
                unique.append(value)
        return unique

    def _corner_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
        hole: Contour,
    ) -> List[np.ndarray]:
        """Кандидаты обхода через углы offset-bbox отверстия."""
        xmin, xmax, ymin, ymax = self._hole_bounds(hole)
        offset = self._route_offset()
        corners = [
            np.array([xmin - offset, ymin - offset]),
            np.array([xmax + offset, ymin - offset]),
            np.array([xmin - offset, ymax + offset]),
            np.array([xmax + offset, ymax + offset]),
        ]
        routes: List[np.ndarray] = []
        for corner in corners:
            routes.append(
                np.array([
                    start,
                    [start[0], corner[1]],
                    corner,
                    [end[0], corner[1]],
                    end,
                ])
            )
            routes.append(
                np.array([
                    start,
                    [corner[0], start[1]],
                    corner,
                    [corner[0], end[1]],
                    end,
                ])
            )
        return routes

    def _side_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
        hole: Contour,
    ) -> List[np.ndarray]:
        """Локальные обходы слева/справа/сверху/снизу от bbox отверстия."""
        if abs(start[1] - end[1]) <= self.geom_epsilon:
            routes: List[np.ndarray] = []
            for route_y in self._boundary_route_ys(
                float(start[0]),
                float(end[0]),
                float(start[1]),
                float(end[1]),
            ):
                routes.append(
                    np.array([
                        start,
                        [start[0], route_y],
                        [end[0], route_y],
                        end,
                    ])
                )
            return routes

        xmin, xmax, ymin, ymax = self._hole_bounds(hole)
        offset = self._route_offset()
        side_x_values = [xmin - offset, xmax + offset]
        side_y_values = [ymin - offset, ymax + offset]

        routes: List[np.ndarray] = []
        for side_x in side_x_values:
            routes.append(
                np.array([
                    start,
                    [side_x, start[1]],
                    [side_x, end[1]],
                    end,
                ])
            )
        for side_y in side_y_values:
            routes.append(
                np.array([
                    start,
                    [start[0], side_y],
                    [end[0], side_y],
                    end,
                ])
            )
        return routes

    def _polyline_length(self, points: np.ndarray) -> float:
        length = 0.0
        for i in range(len(points) - 1):
            length += float(np.linalg.norm(points[i + 1] - points[i]))
        return length

    def _polyline_crosses_hole_interior(self, points: np.ndarray) -> bool:
        for i in range(len(points) - 1):
            if self._segment_crosses_hole_interior(points[i], points[i + 1]):
                return True
        return False

    def _polyline_leaves_outer(self, points: np.ndarray) -> bool:
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            length = float(np.linalg.norm(end - start))
            sample_count = max(5, int(length / max(self.line_distance, self.geom_epsilon)) + 1)
            for i in range(sample_count + 1):
                t = i / sample_count
                point = start + t * (end - start)
                if not self._point_inside_or_on_outer(point):
                    return True
        return False

    def _outer_detour_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> List[np.ndarray]:
        """Длинные обходы через безопасные горизонтальные коридоры и стены bbox."""
        routes: List[np.ndarray] = []
        x_start = float(start[0])
        x_end = float(end[0])

        for route_y in self._enumerate_safe_route_ys(x_start, x_end):
            routes.append(
                np.array([
                    start,
                    [start[0], route_y],
                    [end[0], route_y],
                    end,
                ])
            )

        xmin, xmax, ymin, ymax = self.outer_bounds
        wall_x_values = [xmin, xmax]
        wall_y_values = [ymin, ymax]

        for wall_x in wall_x_values:
            routes.append(
                np.array([
                    start,
                    [wall_x, start[1]],
                    [wall_x, end[1]],
                    end,
                ])
            )
        for wall_y in wall_y_values:
            routes.append(
                np.array([
                    start,
                    [start[0], wall_y],
                    [end[0], wall_y],
                    end,
                ])
            )
        for wall_x in wall_x_values:
            for wall_y in wall_y_values:
                routes.append(
                    np.array([
                        start,
                        [wall_x, start[1]],
                        [wall_x, wall_y],
                        [end[0], wall_y],
                        end,
                    ])
                )
                routes.append(
                    np.array([
                        start,
                        [start[0], wall_y],
                        [wall_x, wall_y],
                        [wall_x, end[1]],
                        end,
                    ])
                )
        return routes

    def _polygon_vertices(self, polygon: np.ndarray) -> np.ndarray:
        if len(polygon) >= 2 and np.allclose(polygon[0], polygon[-1]):
            return polygon[:-1]
        return polygon

    def _project_to_polygon_boundary(
        self, point: np.ndarray, polygon: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Проекция точки на ближайшее ребро полигона."""
        best_dist = float("inf")
        best_point = point.copy()
        best_edge = 0
        vertices = self._polygon_vertices(polygon)
        vertex_count = len(vertices)
        for idx in range(vertex_count):
            start = vertices[idx]
            end = vertices[(idx + 1) % vertex_count]
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= self.geom_epsilon * self.geom_epsilon:
                projection = start.copy()
            else:
                t = float(np.dot(point - start, segment) / length_sq)
                t = max(0.0, min(1.0, t))
                projection = start + t * segment
            dist = float(np.linalg.norm(point - projection))
            if dist < best_dist:
                best_dist = dist
                best_point = projection
                best_edge = idx
        return best_point, best_edge

    def _approach_boundary(
        self, point: np.ndarray, boundary_point: np.ndarray
    ) -> List[np.ndarray]:
        """Безопасный L-образный подход к точке на стенке."""
        if float(np.linalg.norm(point - boundary_point)) <= self.geom_epsilon:
            return [point.copy()]

        via_y = np.array([point[0], boundary_point[1]], dtype=float)
        via_x = np.array([boundary_point[0], point[1]], dtype=float)
        options = [
            [point.copy(), via_y, boundary_point.copy()],
            [point.copy(), via_x, boundary_point.copy()],
        ]
        for route in options:
            if all(
                self._segment_in_travel_zone(route[idx], route[idx + 1])
                for idx in range(len(route) - 1)
            ):
                return route
        return [point.copy(), boundary_point.copy()]

    def _polygon_perimeter_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
        polygon: np.ndarray,
    ) -> List[np.ndarray]:
        """Маршруты вдоль рёбер полигона (в т.ч. наклонные стенки)."""
        vertices = self._polygon_vertices(polygon)
        vertex_count = len(vertices)
        if vertex_count < 3:
            return []

        start_proj, start_edge = self._project_to_polygon_boundary(start, polygon)
        end_proj, end_edge = self._project_to_polygon_boundary(end, polygon)
        routes: List[np.ndarray] = []

        for step in (1, -1):
            pts: List[np.ndarray] = []
            pts.extend(self._approach_boundary(start, start_proj))
            edge = start_edge
            for _ in range(vertex_count + 1):
                if edge == end_edge:
                    break
                if step > 0:
                    pts.append(vertices[(edge + 1) % vertex_count].copy())
                    edge = (edge + 1) % vertex_count
                else:
                    pts.append(vertices[edge].copy())
                    edge = (edge - 1) % vertex_count

            if float(np.linalg.norm(pts[-1] - end_proj)) > self.geom_epsilon:
                pts.append(end_proj.copy())
            pts.extend(self._approach_boundary(end_proj, end)[1:])
            deduped: List[np.ndarray] = [pts[0]]
            for point in pts[1:]:
                if float(np.linalg.norm(point - deduped[-1])) > self.geom_epsilon:
                    deduped.append(point)
            if len(deduped) >= 3:
                routes.append(np.array(deduped))

        return routes

    def _outer_contour_perimeter_routes(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> List[np.ndarray]:
        """Маршруты вдоль рёбер внешнего контура (в т.ч. наклонные стенки)."""
        return self._polygon_perimeter_routes(
            start, end, self.working_contour.points
        )

    def _polyline_is_safe_bridge(self, points: np.ndarray) -> bool:
        if self._polyline_crosses_hole_interior(points):
            return False
        if self._polyline_leaves_outer(points):
            return False
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            if not self._segment_in_travel_zone(start, end):
                return False
            dy = abs(float(end[1]) - float(start[1]))
            if dy <= self.geom_epsilon and not self._horizontal_segment_is_valid_bridge(
                float(start[0]), float(end[0]), float(start[1])
            ):
                return False
        return True

    def _score_route(
        self,
        route: np.ndarray,
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> float:
        length = self._polyline_length(route)
        detour = 0.0
        edge_penalty = 0.0
        offset = self._route_offset()
        if current_y is not None and next_y is not None:
            cur = float(current_y)
            nxt = float(next_y)
            y_lo = min(cur, nxt)
            y_hi = max(cur, nxt)
            for point in route[1:-1]:
                y = float(point[1])
                if y < y_lo - offset:
                    detour += (y_lo - offset - y) * 4.0
                elif y > y_hi + offset:
                    detour += (y - y_hi - offset) * 4.0
                if nxt < cur - self.geom_epsilon and y > nxt + offset:
                    detour += (y - nxt - offset) * 4.0
                elif nxt > cur + self.geom_epsilon and y < nxt - offset:
                    detour += (nxt - offset - y) * 4.0
                for hole in self.planning_holes:
                    xmin, xmax, ymin, ymax = self._hole_bounds(hole)
                    if (
                        abs(y - ymin) <= self.geom_epsilon
                        or abs(y - ymax) <= self.geom_epsilon
                    ) and xmin - self.geom_epsilon <= point[0] <= xmax + self.geom_epsilon:
                        edge_penalty += 5.0
        elif current_y is not None:
            for point in route[1:-1]:
                detour += abs(float(point[1]) - float(current_y)) * 2.0
                for hole in self.planning_holes:
                    xmin, xmax, ymin, ymax = self._hole_bounds(hole)
                    if (
                        abs(float(point[1]) - ymin) <= self.geom_epsilon
                        or abs(float(point[1]) - ymax) <= self.geom_epsilon
                    ) and xmin - self.geom_epsilon <= point[0] <= xmax + self.geom_epsilon:
                        edge_penalty += 5.0
        if next_y is not None:
            detour += abs(float(route[-1][1]) - float(next_y))
        return length + detour + edge_penalty

    def _build_bridge_candidates(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> List[np.ndarray]:
        """Собирает все валидные маршруты перехода."""
        candidates: List[np.ndarray] = []

        direct_route = np.array([start, end])
        if self._polyline_is_safe_bridge(direct_route):
            candidates.append(direct_route)
            return candidates

        crossed = self._crossing_holes(start, end)
        logger.debug(
            "bridge (%.2f, %.2f) -> (%.2f, %.2f): direct rejected, holes=%s",
            start[0],
            start[1],
            end[0],
            end[1],
            [i + 1 for i in crossed],
        )

        seen_routes: List[np.ndarray] = []
        holes_to_consider = (
            [self.planning_holes[i] for i in crossed]
            if crossed
            else self.planning_holes
        )

        for hole in holes_to_consider:
            for route_y in self._route_y_candidates(
                start, end, hole, current_y=current_y, next_y=next_y
            ):
                route = np.array([
                    start,
                    [start[0], route_y],
                    [end[0], route_y],
                    end,
                ])
                if self._polyline_is_safe_bridge(route):
                    seen_routes.append(route)

            for route in self._side_routes(start, end, hole):
                if self._polyline_is_safe_bridge(route):
                    seen_routes.append(route)

            for route in self._corner_routes(start, end, hole):
                if self._polyline_is_safe_bridge(route):
                    seen_routes.append(route)

        for route_y in self._combined_route_y_levels(
            holes_to_consider,
            current_y=current_y,
            next_y=next_y,
        ):
            route = np.array([
                start,
                [start[0], route_y],
                [end[0], route_y],
                end,
            ])
            if self._polyline_is_safe_bridge(route):
                seen_routes.append(route)

        for route_y in self._enumerate_safe_route_ys(
            float(start[0]),
            float(end[0]),
        ):
            route = np.array([
                start,
                [start[0], route_y],
                [end[0], route_y],
                end,
            ])
            if self._polyline_is_safe_bridge(route):
                seen_routes.append(route)

        for route in self._outer_contour_perimeter_routes(start, end):
            if self._polyline_is_safe_bridge(route):
                if not any(
                    route.shape == other.shape and np.allclose(route, other)
                    for other in seen_routes
                ):
                    seen_routes.append(route)

        for hole in holes_to_consider:
            for route in self._polygon_perimeter_routes(start, end, hole.points):
                if self._polyline_is_safe_bridge(route):
                    if not any(
                        route.shape == other.shape and np.allclose(route, other)
                        for other in seen_routes
                    ):
                        seen_routes.append(route)

        if not seen_routes:
            logger.debug(
                "bridge fallback: local route not found for "
                "(%.2f, %.2f) -> (%.2f, %.2f)",
                start[0],
                start[1],
                end[0],
                end[1],
            )
            fallback_route_y_values: List[float] = []
            for hole in holes_to_consider:
                fallback_route_y_values.extend(
                    self._route_y_candidates(
                        start,
                        end,
                        hole,
                        current_y=current_y,
                        next_y=next_y,
                    )
                )
            _, _, outer_ymin, outer_ymax = self.working_contour.bounds
            fallback_offset = self._route_offset()
            fallback_route_y_values.extend([
                outer_ymin - fallback_offset,
                outer_ymax + fallback_offset,
            ])

            for route_y in fallback_route_y_values:
                route = np.array([
                    start,
                    [start[0], route_y],
                    [end[0], route_y],
                    end,
                ])
                if self._polyline_is_safe_bridge(route):
                    seen_routes.append(route)

        for route in self._outer_detour_routes(start, end):
            if self._polyline_is_safe_bridge(route):
                if not any(
                    route.shape == other.shape and np.allclose(route, other)
                    for other in seen_routes
                ):
                    seen_routes.append(route)

        if not seen_routes:
            for route in self._wide_detour_routes(start, end):
                if self._polyline_is_safe_bridge(route):
                    if not any(
                        route.shape == other.shape and np.allclose(route, other)
                        for other in seen_routes
                    ):
                        seen_routes.append(route)

        for route in seen_routes:
            if not any(
                route.shape == other.shape and np.allclose(route, other)
                for other in candidates
            ):
                candidates.append(route)

        for route in candidates:
            score = self._score_route(route, current_y=current_y, next_y=next_y)
            route_y = float(route[1][1]) if len(route) > 2 else float(start[1])
            logger.debug(
                "bridge candidate route_y=%.2f cost=%.2f points=%d",
                route_y,
                score,
                len(route),
            )

        return candidates

    def _bridge_points_simple(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        current_y: Optional[float] = None,
        next_y: Optional[float] = None,
        log_error: bool = True,
    ) -> np.ndarray:
        """Прямой переход для основной змейки; обход только если прямая небезопасна."""
        start = self._snap_bridge_endpoint(start)
        end = self._snap_bridge_endpoint(end)
        if np.linalg.norm(end - start) <= self.geom_epsilon:
            return np.array([start])

        direct_route = np.array([start, end])
        if self._polyline_is_safe_bridge(direct_route):
            return direct_route

        return self._bridge_points(
            start,
            end,
            current_y=current_y,
            next_y=next_y,
            log_error=log_error,
        )

    def _bridge_cost_simple(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> float:
        try:
            route = self._bridge_points_simple(
                start,
                end,
                current_y=current_y,
                next_y=next_y,
                log_error=False,
            )
        except BridgePlanningError:
            return float("inf")
        return self._polyline_length(route)

    def _split_primary_deferred(
        self, lines: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Фаза 1: по одному сегменту на строку — основная змейка от стенки к стенке.
        Фаза 2: остальные сегменты строк, пропущенные из-за отверстий.
        """
        rows = self._group_lines_by_y(lines)
        primary: List[np.ndarray] = []
        deferred: List[np.ndarray] = []

        for row_idx, (y_level, row_lines) in enumerate(rows):
            preferred_ltr = row_idx % 2 == 0

            if len(row_lines) == 1:
                primary.append(row_lines[0])
                continue

            if len(row_lines) > 1:
                best_line = max(
                    row_lines,
                    key=lambda line: float(
                        np.linalg.norm(line[1] - line[0])
                    ),
                )
                primary.append(best_line)
                for line in row_lines:
                    if line is not best_line:
                        deferred.append(line)
                logger.debug(
                    "y=%.2f: основная (самый длинный), отложено %d",
                    y_level,
                    len(row_lines) - 1,
                )
                continue

        return primary, deferred

    def _connect_lines_primary(
        self, lines: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
        """Соединяет основную змейку простыми переходами между строками."""
        if not lines:
            return np.array([]), []

        rows = self._group_lines_by_y(lines)
        path_points: List[np.ndarray] = []
        segments: List[Tuple[int, int, str]] = []

        for row_idx, (y_level, row_lines) in enumerate(rows):
            preferred_ltr = row_idx % 2 == 0
            next_y = rows[row_idx + 1][0] if row_idx + 1 < len(rows) else None
            prev_point = path_points[-1] if path_points else None

            ordered_segments = self._order_row_segments_simple(
                row_lines,
                prev_point,
                preferred_ltr,
                y_level,
                next_y,
            )

            for seg_idx, (seg_start, seg_end) in enumerate(ordered_segments):
                if path_points:
                    try:
                        bridge = self._bridge_points_simple(
                            path_points[-1],
                            seg_start,
                            current_y=float(path_points[-1][1]),
                            next_y=float(seg_start[1]),
                            log_error=False,
                        )
                    except BridgePlanningError as exc:
                        raise BridgePlanningError(
                            f"Не удалось построить безопасный обход отверстия: "
                            f"({path_points[-1][0]:.2f}, {path_points[-1][1]:.2f}) -> "
                            f"({seg_start[0]:.2f}, {seg_start[1]:.2f})"
                        ) from exc
                    travel_start = len(path_points) - 1
                    self._append_polyline(path_points, bridge[1:])
                    self._record_path_segment(
                        segments,
                        travel_start,
                        len(path_points) - 1,
                        "travel",
                    )

                fill_start = self._ensure_point(path_points, seg_start)
                self._append_point(path_points, seg_end)
                self._record_path_segment(
                    segments,
                    fill_start,
                    len(path_points) - 1,
                    "fill",
                )
                logger.debug(
                    "Основная змейка y=%.2f сегмент %d: (%.2f, %.2f) -> (%.2f, %.2f)",
                    y_level,
                    seg_idx,
                    seg_start[0],
                    seg_start[1],
                    seg_end[0],
                    seg_end[1],
                )

        return np.array(path_points), segments

    def _connect_lines_primary_best_effort(
        self, lines: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, str]], int, List[np.ndarray]]:
        """
        Строит непрерывный безопасный фрагмент основной змейки без падения.

        Если следующий основной сегмент нельзя безопасно соединить с текущей
        траекторией, он пропускается и возвращается для дозаполнения.
        """
        if not lines:
            return np.array([]), [], 0, []

        rows = self._group_lines_by_y(lines)
        path_points: List[np.ndarray] = []
        segments: List[Tuple[int, int, str]] = []
        skipped = 0
        skipped_lines: List[np.ndarray] = []

        for row_idx, (y_level, row_lines) in enumerate(rows):
            preferred_ltr = row_idx % 2 == 0
            for line in row_lines:
                left, right = self._segment_endpoints(line)
                orientations = (
                    [(left, right), (right, left)]
                    if preferred_ltr
                    else [(right, left), (left, right)]
                )

                bridge: Optional[np.ndarray] = None
                seg_start, seg_end = orientations[0]
                if path_points:
                    for candidate_start, candidate_end in orientations:
                        try:
                            bridge = self._bridge_points_simple(
                                path_points[-1],
                                candidate_start,
                                current_y=float(path_points[-1][1]),
                                next_y=float(candidate_start[1]),
                                log_error=False,
                            )
                        except BridgePlanningError:
                            continue
                        seg_start, seg_end = candidate_start, candidate_end
                        break

                    if bridge is None:
                        skipped += 1
                        skipped_lines.append(line)
                        logger.debug(
                            "Пропуск основной змейки y=%.2f: "
                            "безопасный переход не найден, отложено дозаполнение",
                            y_level,
                        )
                        continue

                    travel_start = len(path_points) - 1
                    self._append_polyline(path_points, bridge[1:])
                    self._record_path_segment(
                        segments,
                        travel_start,
                        len(path_points) - 1,
                        "travel",
                    )

                fill_start = self._ensure_point(path_points, seg_start)
                self._append_point(path_points, seg_end)
                self._record_path_segment(
                    segments,
                    fill_start,
                    len(path_points) - 1,
                    "fill",
                )

        return np.array(path_points), segments, skipped, skipped_lines

    def _try_append_deferred_fill(
        self,
        path_points: List[np.ndarray],
        segments: List[Tuple[int, int, str]],
        line: np.ndarray,
        preferred_ltr: bool,
    ) -> bool:
        """Добавляет отложенный fill-сегмент через безопасный обход вдоль стенок."""
        preferred_start, preferred_end = self._orient_segment(line, preferred_ltr)
        orientations = [
            (preferred_start, preferred_end),
            (preferred_end, preferred_start),
        ]
        current = path_points[-1] if path_points else None
        best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None

        for orientation_idx, (seg_start, seg_end) in enumerate(orientations):
            if current is not None:
                try:
                    bridge = self._bridge_points(
                        current,
                        seg_start,
                        current_y=float(current[1]),
                        next_y=float(seg_start[1]),
                        log_error=False,
                    )
                except BridgePlanningError:
                    continue
            else:
                bridge = np.array([seg_start])

            score = self._score_route(
                bridge,
                current_y=float(current[1]) if current is not None else None,
                next_y=float(seg_start[1]),
            )
            if orientation_idx:
                score += self.line_distance * 0.1
            if best is None or score < best[0]:
                best = (score, bridge, seg_start, seg_end)

        if best is None:
            return False

        _, bridge, seg_start, seg_end = best
        if path_points and len(bridge) > 0:
            travel_start = len(path_points) - 1
            self._append_polyline(path_points, bridge[1:])
            if len(path_points) - 1 > travel_start:
                self._record_path_segment(
                    segments,
                    travel_start,
                    len(path_points) - 1,
                    "travel",
                )

        fill_start = self._ensure_point(path_points, seg_start)
        self._append_point(path_points, seg_end)
        self._record_path_segment(
            segments,
            fill_start,
            len(path_points) - 1,
            "fill",
        )
        return True

    def _order_row_segments_simple(
        self,
        row_lines: List[np.ndarray],
        prev_point: Optional[np.ndarray],
        preferred_ltr: bool,
        y_level: float,
        next_y: Optional[float],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Упорядочивает сегменты строки для основной змейки (обычно один сегмент)."""
        if not row_lines:
            return []

        if len(row_lines) == 1:
            line = row_lines[0]
            left, right = self._segment_endpoints(line)
            if preferred_ltr:
                return [(left, right)]
            return [(right, left)]

        return self._order_row_segments(
            row_lines,
            prev_point,
            preferred_ltr,
            y_level,
            next_y,
        )

    def _append_deferred_pass(
        self,
        path: np.ndarray,
        segments: List[Tuple[int, int, str]],
        deferred_lines: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
        """Фаза 2: дозаполняет пропуски безопасными обходами вдоль стенок."""
        path_points: List[np.ndarray] = list(path)
        pending: List[Tuple[float, np.ndarray, bool]] = []

        for row_pass_idx, (y_level, row_lines) in enumerate(
            self._group_lines_by_y(deferred_lines)
        ):
            preferred_ltr = row_pass_idx % 2 == 0
            sorted_lines = sorted(
                row_lines,
                key=lambda line: self._segment_endpoints(line)[0][0],
                reverse=not preferred_ltr,
            )
            pending.extend((y_level, line, preferred_ltr) for line in sorted_lines)

        while pending:
            if path_points:
                current = path_points[-1]

                def distance_to_current(
                    item: Tuple[float, np.ndarray, bool],
                ) -> float:
                    _, line, _ = item
                    left, right = self._segment_endpoints(line)
                    return min(
                        float(np.linalg.norm(current - left)),
                        float(np.linalg.norm(current - right)),
                    )

                pending.sort(key=distance_to_current)

            filled = False
            for idx, (y_level, line, preferred_ltr) in enumerate(pending[:3]):
                if self._try_append_deferred_fill(
                    path_points,
                    segments,
                    line,
                    preferred_ltr,
                ):
                    pending.pop(idx)
                    left, right = self._segment_endpoints(line)
                    logger.debug(
                        "Дозаполнение y=%.2f: (%.2f, %.2f) -> (%.2f, %.2f)",
                        y_level,
                        left[0],
                        left[1],
                        right[0],
                        right[1],
                    )
                    filled = True
                    break

            if filled:
                continue

            y_level, _, _ = pending.pop(0)
            self.skipped_deferred_segments += 1
            logger.warning(
                "Дозаполнение y=%.2f пропущено: безопасный переход не найден",
                y_level,
            )

        return np.array(path_points), segments

    def _deferred_bridge_option(
        self,
        path_points: List[np.ndarray],
        line: np.ndarray,
        preferred_ltr: bool,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """Лучший безопасный переход к одному отложенному сегменту."""
        preferred_start, preferred_end = self._orient_segment(line, preferred_ltr)
        orientations = [
            (preferred_start, preferred_end),
            (preferred_end, preferred_start),
        ]

        if not path_points:
            return 0.0, np.array([preferred_start]), preferred_start, preferred_end

        current = path_points[-1]
        best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
        for orientation_idx, (seg_start, seg_end) in enumerate(orientations):
            try:
                bridge = self._bridge_points(
                    current,
                    seg_start,
                    current_y=float(current[1]),
                    next_y=float(seg_start[1]),
                    log_error=False,
                )
            except BridgePlanningError:
                continue

            score = self._score_route(
                bridge,
                current_y=float(current[1]),
                next_y=float(seg_start[1]),
            )
            if orientation_idx:
                score += self.line_distance * 0.1
            if best is None or score < best[0]:
                best = (score, bridge, seg_start, seg_end)

        return best

    def _deferred_outer_bridge(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Быстрый дальний мост для дозаполнения без полного перебора кандидатов."""
        direct = np.array([start, end])
        if self._polyline_is_safe_bridge(direct):
            return direct

        best: Optional[np.ndarray] = None
        best_score = float("inf")
        xmin, xmax, _, _ = self.outer_bounds
        routes: List[np.ndarray] = []
        for wall_x in (xmin, xmax):
            routes.append(
                np.array([
                    start,
                    [wall_x, start[1]],
                    [wall_x, end[1]],
                    end,
                ])
            )

        for route in routes:
            if not self._polyline_is_safe_bridge(route):
                continue
            score = self._polyline_length(route)
            if score < best_score:
                best = route
                best_score = score
        return best

    def _bridge_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        current_y: Optional[float] = None,
        next_y: Optional[float] = None,
        log_error: bool = True,
    ) -> np.ndarray:
        """Строит переход между точками, обходя отверстия снаружи."""
        start = self._snap_bridge_endpoint(start)
        end = self._snap_bridge_endpoint(end)
        if np.linalg.norm(end - start) <= self.geom_epsilon:
            return np.array([start])

        cache_key = self._bridge_cache_key(start, end, current_y, next_y)
        cached = self._bridge_route_cache.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in self._bridge_failed_cache:
            msg = (
                f"Не удалось построить безопасный обход отверстия: "
                f"({start[0]:.2f}, {start[1]:.2f}) -> ({end[0]:.2f}, {end[1]:.2f})"
            )
            raise BridgePlanningError(msg)

        candidates = self._build_bridge_candidates(
            start, end, current_y=current_y, next_y=next_y
        )

        if not candidates:
            msg = (
                f"Не удалось построить безопасный обход отверстия: "
                f"({start[0]:.2f}, {start[1]:.2f}) -> ({end[0]:.2f}, {end[1]:.2f})"
            )
            if log_error:
                self._bridge_failed_cache.add(cache_key)
                logger.warning(msg)
            raise BridgePlanningError(msg)

        best = min(
            candidates,
            key=lambda route: self._score_route(
                route, current_y=current_y, next_y=next_y
            ),
        )
        score = self._score_route(best, current_y=current_y, next_y=next_y)
        route_y = float(best[1][1]) if len(best) > 2 else float(start[1])
        logger.debug(
            "bridge selected (%.2f, %.2f) -> (%.2f, %.2f), route_y=%.2f cost=%.2f",
            start[0],
            start[1],
            end[0],
            end[1],
            route_y,
            score,
        )
        self._bridge_route_cache[cache_key] = best
        return best

    def _bridge_cache_key(
        self,
        start: np.ndarray,
        end: np.ndarray,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> Tuple[float, float, float, float, Optional[float], Optional[float]]:
        precision = 6
        return (
            round(float(start[0]), precision),
            round(float(start[1]), precision),
            round(float(end[0]), precision),
            round(float(end[1]), precision),
            None if current_y is None else round(float(current_y), precision),
            None if next_y is None else round(float(next_y), precision),
        )

    def _bridge_cost(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        current_y: Optional[float],
        next_y: Optional[float],
    ) -> float:
        """Стоимость безопасного перехода; inf если маршрут невозможен."""
        try:
            route = self._bridge_points(
                start,
                end,
                current_y=current_y,
                next_y=next_y,
                log_error=False,
            )
        except BridgePlanningError:
            return float("inf")
        return self._score_route(route, current_y=current_y, next_y=next_y)

    def _order_row_segments(
        self,
        row_lines: List[np.ndarray],
        prev_point: Optional[np.ndarray],
        preferred_ltr: bool,
        y_level: float,
        next_y: Optional[float],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Greedy ordering с разворотом open segments по минимальной стоимости bridge.
        """
        if not row_lines:
            return []

        remaining = list(row_lines)
        ordered: List[Tuple[np.ndarray, np.ndarray]] = []
        current = prev_point

        while remaining:
            best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
            best_cost = float("inf")
            best_idx = -1

            for idx, line in enumerate(remaining):
                left, right = self._segment_endpoints(line)
                orientations = [(left, right), (right, left)]
                if len(remaining) == len(row_lines) and current is None:
                    if preferred_ltr:
                        orientations.sort(key=lambda pair: pair[0][0])
                    else:
                        orientations.sort(key=lambda pair: -pair[0][0])

                for seg_start, seg_end in orientations:
                    seg_y = float(seg_start[1])
                    if current is None:
                        cost = abs(float(seg_start[0]) - float(seg_end[0]) * 0.01)
                    else:
                        cost = self._bridge_cost(
                            current,
                            seg_start,
                            current_y=float(current[1]),
                            next_y=seg_y if not ordered else seg_y,
                        )
                    if best_pair is None or cost < best_cost:
                        best_cost = cost
                        best_pair = (seg_start, seg_end)
                        best_idx = idx

            if best_pair is None:
                raise BridgePlanningError(
                    f"Не удалось упорядочить сегменты на y={y_level:.2f}"
                )
            ordered.append(best_pair)
            current = best_pair[1]
            remaining.pop(best_idx)

        return ordered

    def _append_point(self, path: List[np.ndarray], point: np.ndarray) -> None:
        point = np.array(point, dtype=float)
        if path and np.linalg.norm(path[-1] - point) <= self.geom_epsilon:
            return
        path.append(point)

    def _ensure_point(self, path: List[np.ndarray], point: np.ndarray) -> int:
        """Добавляет точку при необходимости и возвращает её индекс в path."""
        before = len(path)
        self._append_point(path, point)
        if len(path) == before:
            return len(path) - 1
        return len(path) - 1

    def _append_polyline(self, path: List[np.ndarray], points: np.ndarray) -> None:
        for point in points:
            self._append_point(path, point)

    def _record_path_segment(
        self,
        segments: List[Tuple[int, int, str]],
        start_idx: int,
        end_idx: int,
        kind: str,
    ) -> None:
        if end_idx > start_idx:
            segments.append((start_idx, end_idx, kind))

    def _connect_lines(
        self, lines: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
        """
        Соединяет отрезки заливки в непрерывную змейку.

        Строки чередуют направление; внутри строки — greedy ordering по bridge-cost.
        """
        if not lines:
            return np.array([]), []

        rows = self._group_lines_by_y(lines)
        path_points: List[np.ndarray] = []
        segments: List[Tuple[int, int, str]] = []

        for row_idx, (y_level, row_lines) in enumerate(rows):
            preferred_ltr = row_idx % 2 == 0
            next_y = rows[row_idx + 1][0] if row_idx + 1 < len(rows) else None
            prev_point = path_points[-1] if path_points else None

            ordered_segments = self._order_row_segments(
                row_lines,
                prev_point,
                preferred_ltr,
                y_level,
                next_y,
            )

            for seg_idx, (seg_start, seg_end) in enumerate(ordered_segments):
                if path_points:
                    bridge = self._bridge_points(
                        path_points[-1],
                        seg_start,
                        current_y=float(path_points[-1][1]),
                        next_y=float(seg_start[1]),
                    )
                    travel_start = len(path_points) - 1
                    self._append_polyline(path_points, bridge[1:])
                    self._record_path_segment(
                        segments,
                        travel_start,
                        len(path_points) - 1,
                        "travel",
                    )

                fill_start = self._ensure_point(path_points, seg_start)
                self._append_point(path_points, seg_end)
                self._record_path_segment(
                    segments,
                    fill_start,
                    len(path_points) - 1,
                    "fill",
                )
                logger.debug(
                    "Змейка y=%.2f сегмент %d: (%.2f, %.2f) -> (%.2f, %.2f)",
                    y_level,
                    seg_idx,
                    seg_start[0],
                    seg_start[1],
                    seg_end[0],
                    seg_end[1],
                )

        return np.array(path_points), segments

    def _rotate_back(self, path: np.ndarray) -> np.ndarray:
        """Поворачивает траекторию обратно в исходную СК."""
        angle_rad = np.radians(-self.fill_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ])
        return path @ rotation_matrix.T

    def save_path(self, filepath: str, format: str = "txt") -> None:
        """Сохраняет траекторию в файл."""
        path = self.generate_path()
        if format == "txt":
            np.savetxt(filepath, path, fmt="%.6f", delimiter=" ")
            logger.info("Траектория сохранена в %s", filepath)
        elif format == "csv":
            np.savetxt(filepath, path, fmt="%.6f", delimiter=",")
            logger.info("Траектория сохранена в %s", filepath)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

    def get_statistics(self, path: Optional[np.ndarray] = None) -> dict:
        """Возвращает статистику по построенной траектории."""
        if path is None:
            path = self.generate_path()

        if len(path) == 0:
            return {
                "total_points": 0,
                "total_length": 0.0,
                "num_lines": 0,
                "filled_segments": 0,
                "skipped_primary_segments": self.skipped_primary_segments,
                "skipped_deferred_segments": self.skipped_deferred_segments,
                "disconnected_deferred_segments": (
                    self.disconnected_deferred_segments
                ),
                "y_range": (self.y_min, self.y_max),
            }
        total_length = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            total_length += np.linalg.norm(p2 - p1)
        return {
            "total_points": len(path),
            "total_length": total_length,
            "num_lines": len(self._generate_fill_lines()),
            "filled_segments": sum(
                1 for _, _, kind in self.path_segments if kind == "fill"
            ),
            "skipped_primary_segments": self.skipped_primary_segments,
            "skipped_deferred_segments": self.skipped_deferred_segments,
            "disconnected_deferred_segments": (
                self.disconnected_deferred_segments
            ),
            "y_range": (self.y_min, self.y_max),
            "line_distance": self.line_distance,
            "fill_angle": self.fill_angle,
            "hole_clearance": self.hole_clearance,
            "allow_clearance_contact": self.allow_clearance_contact,
        }

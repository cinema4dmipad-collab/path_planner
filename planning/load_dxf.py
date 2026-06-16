from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from exceptions import (
    DXFLoadError,
    FileNotFoundError as DXFNotFoundError,
    InsufficientPointsError,
    UnsupportedEntityError,
)
from logger import logger

try:
    import ezdxf

    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


Point2D = Tuple[float, float]
Segment = Tuple[np.ndarray, np.ndarray]


class ContourSelection(str, Enum):
    """Стратегия выбора контура из DXF."""

    LARGEST_CLOSED = "largest_closed"
    FIRST_CLOSED = "first_closed"
    FIRST = "first"


@dataclass(frozen=True)
class DXFContourInfo:
    """Описание одного контура, извлечённого из DXF."""

    points: np.ndarray
    is_closed: bool
    layer: str
    entity_types: Tuple[str, ...]
    area: float

    @property
    def num_points(self) -> int:
        return len(self.points)


class DXFLoader:
    """
    Загрузчик контуров из DXF.

    Извлекает отдельные кривые/цепочки, дискретизирует дуги и bulge-сегменты,
    затем выбирает подходящий контур для заливки.
    """

    SUPPORTED_ENTITIES = frozenset({
        "LINE",
        "LWPOLYLINE",
        "POLYLINE",
        "ARC",
        "CIRCLE",
        "SPLINE",
        "ELLIPSE",
        "INSERT",
    })

    def __init__(self, tolerance: float = 0.1, join_tolerance: float = 1e-3):
        if tolerance <= 0:
            raise ValueError("tolerance должен быть > 0")
        self.tolerance = tolerance
        self.join_tolerance = join_tolerance

    def load(
        self,
        filepath: Union[str, Path],
        *,
        layer: Optional[str] = None,
        selection: ContourSelection = ContourSelection.LARGEST_CLOSED,
        closed_only: bool = True,
    ) -> np.ndarray:
        """Загружает один контур из DXF (точки формы (n, 2))."""
        contour = self.load_selected_contour(
            filepath,
            layer=layer,
            selection=selection,
            closed_only=closed_only,
        )
        return contour.points

    def load_selected_contour(
        self,
        filepath: Union[str, Path],
        *,
        layer: Optional[str] = None,
        selection: ContourSelection = ContourSelection.LARGEST_CLOSED,
        closed_only: bool = True,
    ) -> DXFContourInfo:
        """Загружает и возвращает метаданные выбранного контура."""
        contours = self.load_contours(filepath, layer=layer)
        if not contours:
            raise DXFLoadError("В DXF-файле не найдено поддерживаемых примитивов")

        return self._select_contour(
            contours,
            selection=selection,
            closed_only=closed_only,
        )

    def load_contours(
        self,
        filepath: Union[str, Path],
        *,
        layer: Optional[str] = None,
    ) -> List[DXFContourInfo]:
        """Извлекает все найденные контуры из DXF."""
        if not EZDXF_AVAILABLE:
            raise ImportError(
                "Библиотека ezdxf не установлена. Установите: poetry add ezdxf"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise DXFNotFoundError(f"Файл не найден: {filepath}")

        logger.info("Загрузка DXF-файла: %s", filepath)

        try:
            doc = ezdxf.readfile(str(filepath))
        except Exception as exc:
            raise DXFLoadError(f"Не удалось прочитать DXF-файл: {exc}") from exc

        raw_contours: List[DXFContourInfo] = []
        line_segments: List[Segment] = []

        for entity in doc.modelspace():
            if layer is not None and entity.dxf.layer != layer:
                continue

            dxftype = entity.dxftype()
            if dxftype not in self.SUPPORTED_ENTITIES:
                logger.warning("Пропущен неподдерживаемый тип: %s", dxftype)
                continue

            try:
                if dxftype == "LINE":
                    line_segments.extend(self._process_line(entity))
                elif dxftype == "INSERT":
                    raw_contours.extend(self._process_insert(entity, layer))
                else:
                    raw_contours.extend(self._process_entity(entity))
            except Exception as exc:
                logger.error("Ошибка обработки %s: %s", dxftype, exc)
                continue

        if line_segments:
            raw_contours.extend(self._chain_line_segments(line_segments))

        contours = [self._finalize_contour(info) for info in raw_contours]
        contours = [info for info in contours if info.num_points >= 3]

        if not contours:
            raise DXFLoadError("Не удалось построить контур с минимум 3 точками")

        logger.info("Из DXF извлечено контуров: %d", len(contours))
        return contours

    def _process_insert(
        self,
        insert,
        layer_filter: Optional[str],
    ) -> List[DXFContourInfo]:
        contours: List[DXFContourInfo] = []
        line_segments: List[Segment] = []

        for entity in insert.virtual_entities():
            if layer_filter is not None and entity.dxf.layer != layer_filter:
                continue

            dxftype = entity.dxftype()
            if dxftype not in self.SUPPORTED_ENTITIES or dxftype == "INSERT":
                continue

            if dxftype == "LINE":
                line_segments.extend(self._process_line(entity))
            else:
                contours.extend(self._process_entity(entity))

        if line_segments:
            contours.extend(self._chain_line_segments(line_segments))

        return contours

    def _process_entity(self, entity) -> List[DXFContourInfo]:
        dxftype = entity.dxftype()
        layer = entity.dxf.layer

        if dxftype in ("LWPOLYLINE", "POLYLINE"):
            points = self._discretize_polyline(entity)
        elif dxftype == "ARC":
            points = self._discretize_arc(entity)
        elif dxftype == "CIRCLE":
            points = self._discretize_circle(entity)
        elif dxftype == "SPLINE":
            points = self._discretize_spline(entity)
        elif dxftype == "ELLIPSE":
            points = self._discretize_ellipse(entity)
        else:
            raise UnsupportedEntityError(f"Неподдерживаемый тип: {dxftype}")

        if len(points) < 2:
            return []

        is_closed = self._entity_is_closed(entity, points)
        area = self._polygon_area(points) if is_closed else 0.0
        return [
            DXFContourInfo(
                points=points,
                is_closed=is_closed,
                layer=layer,
                entity_types=(dxftype,),
                area=area,
            )
        ]

    def _process_line(self, line) -> List[Segment]:
        start = np.array([line.dxf.start.x, line.dxf.start.y], dtype=float)
        end = np.array([line.dxf.end.x, line.dxf.end.y], dtype=float)
        return [(start, end)]

    def _chain_line_segments(self, segments: Sequence[Segment]) -> List[DXFContourInfo]:
        if not segments:
            return []

        remaining = list(segments)
        chains: List[List[np.ndarray]] = []

        while remaining:
            current = remaining.pop(0)
            chain = [current[0].copy(), current[1].copy()]

            changed = True
            while changed:
                changed = False
                for idx in range(len(remaining) - 1, -1, -1):
                    seg = remaining[idx]
                    appended = self._try_append_segment(chain, seg)
                    if appended:
                        remaining.pop(idx)
                        changed = True

            layer = "0"
            points = np.array(chain, dtype=float)
            is_closed = self._points_are_closed(points)
            area = self._polygon_area(points) if is_closed else 0.0
            chains.append(
                DXFContourInfo(
                    points=points,
                    is_closed=is_closed,
                    layer=layer,
                    entity_types=("LINE",),
                    area=area,
                )
            )

        return chains

    def _try_append_segment(self, chain: List[np.ndarray], segment: Segment) -> bool:
        start, end = segment
        head = chain[0]
        tail = chain[-1]

        if self._points_close(tail, start):
            chain.append(end.copy())
            return True
        if self._points_close(tail, end):
            chain.append(start.copy())
            return True
        if self._points_close(head, end):
            chain.insert(0, start.copy())
            return True
        if self._points_close(head, start):
            chain.insert(0, end.copy())
            return True
        return False

    def _discretize_polyline(self, polyline) -> np.ndarray:
        if hasattr(polyline, "flattening"):
            try:
                vertices = list(polyline.flattening(self.tolerance))
                if vertices:
                    return np.array([[v.x, v.y] for v in vertices], dtype=float)
            except TypeError:
                pass

        points: List[Point2D] = []
        if polyline.dxftype() == "LWPOLYLINE":
            for x, y, *_ in polyline.get_points("xy"):
                points.append((float(x), float(y)))
        else:
            for vertex in polyline.vertices:
                loc = vertex.dxf.location
                points.append((float(loc.x), float(loc.y)))

        return np.array(points, dtype=float)

    def _curve_segment_count(
        self,
        radius: float,
        total_angle: float = 2 * math.pi,
        *,
        min_segments: int = 8,
    ) -> int:
        """Число сегментов по хордовой ошибке (как flattening в ezdxf)."""
        if radius <= 0 or total_angle <= 0:
            return min_segments
        max_error = self.tolerance
        if max_error >= radius:
            return min_segments
        cos_val = max(-1.0, min(1.0, 1.0 - max_error / radius))
        segment_angle = 2.0 * math.acos(cos_val)
        if segment_angle <= 1e-12:
            return min_segments
        return max(min_segments, int(math.ceil(total_angle / segment_angle)))

    def _discretize_arc(self, arc) -> np.ndarray:
        center = arc.dxf.center
        radius = float(arc.dxf.radius)
        start_rad = np.radians(float(arc.dxf.start_angle))
        end_rad = np.radians(float(arc.dxf.end_angle))

        if end_rad < start_rad:
            end_rad += 2 * np.pi

        arc_angle = abs(end_rad - start_rad)
        num_segments = self._curve_segment_count(
            radius, arc_angle, min_segments=3
        )

        points: List[Point2D] = []
        for i in range(num_segments + 1):
            t = i / num_segments
            angle = start_rad + t * (end_rad - start_rad)
            points.append(
                (
                    center.x + radius * np.cos(angle),
                    center.y + radius * np.sin(angle),
                )
            )
        return np.array(points, dtype=float)

    def _discretize_circle(self, circle) -> np.ndarray:
        center = circle.dxf.center
        radius = float(circle.dxf.radius)
        num_segments = self._curve_segment_count(radius)

        points: List[Point2D] = []
        for i in range(num_segments + 1):
            angle = 2 * np.pi * i / num_segments
            points.append(
                (
                    center.x + radius * np.cos(angle),
                    center.y + radius * np.sin(angle),
                )
            )
        return np.array(points, dtype=float)

    def _discretize_spline(self, spline) -> np.ndarray:
        vertices = list(spline.flattening(self.tolerance))
        return np.array([[v.x, v.y] for v in vertices], dtype=float)

    def _discretize_ellipse(self, ellipse) -> np.ndarray:
        center = ellipse.dxf.center
        major_axis = ellipse.dxf.major_axis
        ratio = float(ellipse.dxf.ratio)

        a = float(np.hypot(major_axis.x, major_axis.y))
        b = a * ratio
        angle = float(np.arctan2(major_axis.y, major_axis.x))

        num_segments = self._curve_segment_count(max(a, b))

        points: List[Point2D] = []
        for i in range(num_segments + 1):
            t = 2 * np.pi * i / num_segments
            x_local = a * np.cos(t)
            y_local = b * np.sin(t)
            x_rot = x_local * np.cos(angle) - y_local * np.sin(angle)
            y_rot = x_local * np.sin(angle) + y_local * np.cos(angle)
            points.append((center.x + x_rot, center.y + y_rot))

        return np.array(points, dtype=float)

    def _entity_is_closed(self, entity, points: np.ndarray) -> bool:
        if hasattr(entity, "closed"):
            if bool(entity.closed):
                return True
        if hasattr(entity, "is_closed"):
            if bool(entity.is_closed):
                return True
        if entity.dxftype() == "CIRCLE":
            return True
        return self._points_are_closed(points)

    def _points_are_closed(self, points: np.ndarray) -> bool:
        if len(points) < 3:
            return False
        return bool(np.linalg.norm(points[0] - points[-1]) <= self.join_tolerance)

    def _points_close(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.linalg.norm(a - b) <= self.join_tolerance)

    def _polygon_area(self, points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        pts = points
        if not self._points_are_closed(pts):
            pts = np.vstack([pts, pts[0]])
        x = pts[:, 0]
        y = pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _finalize_contour(self, info: DXFContourInfo) -> DXFContourInfo:
        points = self._dedupe_consecutive(info.points)
        is_closed = info.is_closed or self._points_are_closed(points)
        area = self._polygon_area(points) if is_closed else 0.0
        return DXFContourInfo(
            points=points,
            is_closed=is_closed,
            layer=info.layer,
            entity_types=info.entity_types,
            area=area,
        )

    def _dedupe_consecutive(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        deduped = [points[0]]
        for point in points[1:]:
            if not self._points_close(deduped[-1], point):
                deduped.append(point)
        return np.array(deduped, dtype=float)

    def _select_contour(
        self,
        contours: Sequence[DXFContourInfo],
        *,
        selection: ContourSelection,
        closed_only: bool,
    ) -> DXFContourInfo:
        if closed_only:
            closed = [contour for contour in contours if contour.is_closed]
            if not closed:
                open_count = len(contours)
                raise DXFLoadError(
                    "В DXF не найдено замкнутых контуров "
                    f"(найдено открытых: {open_count})"
                )
            pool = closed
        else:
            pool = list(contours)

        if selection == ContourSelection.FIRST:
            chosen = pool[0]
        elif selection == ContourSelection.FIRST_CLOSED:
            chosen = pool[0]
        else:
            chosen = max(pool, key=lambda contour: contour.area)

        if chosen.num_points < 3:
            raise InsufficientPointsError(
                f"Выбранный контур содержит только {chosen.num_points} точек"
            )

        logger.info(
            "Выбран контур: layer=%s, types=%s, closed=%s, points=%d, area=%.2f",
            chosen.layer,
            ",".join(chosen.entity_types),
            chosen.is_closed,
            chosen.num_points,
            chosen.area,
        )
        return chosen


def load_dxf(
    filepath: Union[str, Path],
    tolerance: float = 0.1,
    *,
    layer: Optional[str] = None,
    selection: ContourSelection | str = ContourSelection.LARGEST_CLOSED,
    closed_only: bool = True,
) -> np.ndarray:
    """
    Загружает контур из DXF-файла.

    По умолчанию выбирается самый большой замкнутый контур.
    """
    if isinstance(selection, str):
        selection = ContourSelection(selection)

    loader = DXFLoader(tolerance=tolerance)
    return loader.load(
        filepath,
        layer=layer,
        selection=selection,
        closed_only=closed_only,
    )


def load_dxf_contours(
    filepath: Union[str, Path],
    tolerance: float = 0.1,
    *,
    layer: Optional[str] = None,
) -> List[DXFContourInfo]:
    """Возвращает все контуры, найденные в DXF."""
    loader = DXFLoader(tolerance=tolerance)
    return loader.load_contours(filepath, layer=layer)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from contour_nesting import analyze_contour_nesting, get_direct_holes
from geometry import Contour
from load_dxf import load_dxf_contours
from load_txt import load_txt
from logger import logger
from planner import PathPlanner


@dataclass
class ContourPlanResult:
    """Результат планирования для одного контура."""

    index: int
    layer: str
    entity_types: Tuple[str, ...]
    is_closed: bool
    area: float
    original_points: np.ndarray
    contour: Optional[Contour]
    path: np.ndarray
    stats: dict
    skipped: bool = False
    skip_reason: str = ""
    nesting_depth: int = 0
    hole_count: int = 0
    hole_points: Tuple[np.ndarray, ...] = ()
    path_segments: Tuple[Tuple[int, int, str], ...] = ()

    @property
    def label(self) -> str:
        types = ", ".join(self.entity_types) if self.entity_types else "TXT"
        return f"#{self.index + 1} [{self.layer}] {types}"


def plan_contour_points(
    points: np.ndarray,
    *,
    tolerance: float,
    line_distance: float,
    fill_angle: float,
    mode: str,
    index: int = 0,
    layer: str = "0",
    entity_types: Tuple[str, ...] = ("TXT",),
    is_closed: bool = True,
    area: float = 0.0,
    holes: Optional[List[Contour]] = None,
    nesting_depth: int = 0,
    precomputed_contour: Optional[Contour] = None,
    hole_clearance: float = 0.0,
    allow_clearance_contact: bool = True,
) -> ContourPlanResult:
    """Строит траекторию заливки для одного набора точек."""
    contour = precomputed_contour
    if contour is None:
        contour = Contour(points, approximation_tolerance=tolerance)
        contour = contour.discretize(mode=mode)

    hole_contours = holes or []
    planner = PathPlanner(
        contour=contour,
        line_distance=line_distance,
        fill_angle=fill_angle,
        tolerance=tolerance,
        holes=hole_contours,
        hole_clearance=hole_clearance,
        allow_clearance_contact=allow_clearance_contact,
    )
    path = planner.generate_path()
    stats = planner.get_statistics(path)
    stats["hole_count"] = len(hole_contours)
    stats["nesting_depth"] = nesting_depth
    stats["hole_clearance"] = hole_clearance
    stats["allow_clearance_contact"] = allow_clearance_contact
    path_segments = planner.get_path_segments()

    hole_points = tuple(hole.points.copy() for hole in hole_contours)

    return ContourPlanResult(
        index=index,
        layer=layer,
        entity_types=entity_types,
        is_closed=is_closed,
        area=area,
        original_points=points.copy(),
        contour=contour,
        path=path,
        stats=stats,
        nesting_depth=nesting_depth,
        hole_count=len(hole_contours),
        hole_points=hole_points,
        path_segments=path_segments,
    )


def _skipped_result(
    *,
    index: int,
    info,
    skip_reason: str,
    nesting_depth: int = 0,
) -> ContourPlanResult:
    return ContourPlanResult(
        index=index,
        layer=info.layer,
        entity_types=info.entity_types,
        is_closed=info.is_closed,
        area=info.area,
        original_points=info.points.copy(),
        contour=None,
        path=np.array([]),
        stats={
            "total_points": 0,
            "total_length": 0.0,
            "num_lines": 0,
            "hole_count": 0,
            "nesting_depth": nesting_depth,
        },
        skipped=True,
        skip_reason=skip_reason,
        nesting_depth=nesting_depth,
    )


def plan_dxf_contours(
    filepath: Union[str, Path],
    *,
    tolerance: float = 0.1,
    line_distance: float = 2.0,
    fill_angle: float = 0.0,
    mode: str = "as_is",
    closed_only: bool = True,
    layer: Optional[str] = None,
    hole_clearance: float = 0.0,
    allow_clearance_contact: bool = True,
) -> List[ContourPlanResult]:
    """
    Загружает и обрабатывает контуры из DXF с учётом вложенности.

    Внутренние контуры (отверстия) не заливаются отдельно — они вырезаются
    из заливки внешнего контура.
    """
    dxf_contours = load_dxf_contours(filepath, tolerance=tolerance, layer=layer)
    results: List[ContourPlanResult] = []

    closed_items: List[Tuple[int, object, Contour]] = []
    for index, info in enumerate(dxf_contours):
        if closed_only and not info.is_closed:
            logger.warning(
                "Пропущен открытый контур #%d (layer=%s)", index + 1, info.layer
            )
            results.append(
                _skipped_result(index=index, info=info, skip_reason="открытый контур")
            )
            continue

        contour = Contour(info.points, approximation_tolerance=tolerance)
        contour = contour.discretize(mode=mode)
        closed_items.append((index, info, contour))

    if not closed_items and results:
        return results

    if closed_items:
        nesting = analyze_contour_nesting([item[2] for item in closed_items])
    else:
        nesting = []

    processed_indices = set()

    for local_idx, (index, info, contour) in enumerate(closed_items):
        nest = nesting[local_idx]

        if nest.is_hole:
            logger.info(
                "Контур #%d пропущен: отверстие (depth=%d)",
                index + 1,
                nest.depth,
            )
            results.append(
                _skipped_result(
                    index=index,
                    info=info,
                    skip_reason="отверстие внутри другого контура",
                    nesting_depth=nest.depth,
                )
            )
            continue

        hole_indices = get_direct_holes(local_idx, nesting)
        hole_contours = [closed_items[hi][2] for hi in hole_indices]

        try:
            result = plan_contour_points(
                info.points,
                tolerance=tolerance,
                line_distance=line_distance,
                fill_angle=fill_angle,
                mode=mode,
                index=index,
                layer=info.layer,
                entity_types=info.entity_types,
                is_closed=info.is_closed,
                area=info.area,
                holes=hole_contours,
                nesting_depth=nest.depth,
                precomputed_contour=contour,
                hole_clearance=hole_clearance,
                allow_clearance_contact=allow_clearance_contact,
            )
            results.append(result)
            processed_indices.add(index)
            logger.info(
                "Контур #%d: path=%d точек, длина=%.2f мм, отверстий=%d",
                index + 1,
                result.stats["total_points"],
                result.stats["total_length"],
                result.hole_count,
            )
        except Exception as exc:
            logger.error("Ошибка контура #%d: %s", index + 1, exc)
            results.append(
                _skipped_result(
                    index=index,
                    info=info,
                    skip_reason=str(exc),
                    nesting_depth=nest.depth,
                )
            )

    results.sort(key=lambda item: item.index)
    return results


def plan_file_contours(
    filepath: Union[str, Path],
    *,
    tolerance: float = 0.1,
    line_distance: float = 2.0,
    fill_angle: float = 0.0,
    mode: str = "as_is",
    closed_only: bool = True,
    layer: Optional[str] = None,
    hole_clearance: float = 0.0,
    allow_clearance_contact: bool = True,
) -> List[ContourPlanResult]:
    """Обрабатывает TXT (один контур) или DXF (все контуры)."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".txt":
        points = load_txt(filepath)
        return [
            plan_contour_points(
                points,
                tolerance=tolerance,
                line_distance=line_distance,
                fill_angle=fill_angle,
                mode=mode,
                hole_clearance=hole_clearance,
                allow_clearance_contact=allow_clearance_contact,
            )
        ]

    if suffix == ".dxf":
        return plan_dxf_contours(
            filepath,
            tolerance=tolerance,
            line_distance=line_distance,
            fill_angle=fill_angle,
            mode=mode,
            closed_only=closed_only,
            layer=layer,
            hole_clearance=hole_clearance,
            allow_clearance_contact=allow_clearance_contact,
        )

    raise ValueError(f"Неподдерживаемый формат: {suffix}")


def summarize_results(results: Sequence[ContourPlanResult]) -> dict:
    """Сводная статистика по пакетной обработке."""
    processed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    return {
        "total_contours": len(results),
        "processed": len(processed),
        "skipped": len(skipped),
        "total_path_points": sum(r.stats["total_points"] for r in processed),
        "total_path_length": sum(r.stats["total_length"] for r in processed),
        "total_fill_lines": sum(r.stats["num_lines"] for r in processed),
        "total_holes": sum(r.hole_count for r in processed),
    }

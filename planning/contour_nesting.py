from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry import Contour


@dataclass(frozen=True)
class ContourNestingInfo:
    """Иерархия вложенности контура."""

    index: int
    depth: int
    parent: Optional[int]
    children: Tuple[int, ...]
    is_fill_boundary: bool
    is_hole: bool


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Проверяет, лежит ли точка внутри замкнутого полигона (ray casting)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    pts = polygon
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]

        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < x_intersect:
                inside = not inside

    return inside


def contour_contains(outer: Contour, inner: Contour) -> bool:
    """True, если inner целиком внутри outer."""
    inner_pts = inner.points
    outer_pts = outer.points

    outer_area = abs(_polygon_area(outer_pts))
    inner_area = abs(_polygon_area(inner_pts))
    if inner_area >= outer_area:
        return False

    if len(inner_pts) > 1 and np.linalg.norm(inner_pts[0] - inner_pts[-1]) < 1e-6:
        inner_unique = inner_pts[:-1]
    else:
        inner_unique = inner_pts

    if len(inner_unique) == 0:
        return point_in_polygon(inner_pts[0], outer_pts)

    margin = max(float(outer.approximation_tolerance), 1e-6) * 0.01
    for pt in inner_unique:
        if not point_in_polygon(pt, outer_pts):
            x, y = float(pt[0]), float(pt[1])
            ox_min, ox_max = outer_pts[:, 0].min(), outer_pts[:, 0].max()
            oy_min, oy_max = outer_pts[:, 1].min(), outer_pts[:, 1].max()
            on_outer_edge = (
                abs(x - ox_min) <= margin
                or abs(x - ox_max) <= margin
                or abs(y - oy_min) <= margin
                or abs(y - oy_max) <= margin
            )
            if not on_outer_edge:
                return False

    return True


def _representative_points(points: np.ndarray) -> List[np.ndarray]:
    unique = points
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-6:
        unique = points[:-1]
    if len(unique) == 0:
        return [points[0]]

    centroid = np.mean(unique, axis=0)
    samples = [centroid, unique[0], unique[len(unique) // 2]]
    return samples


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = points
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        pts = points[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def find_immediate_parent(
    index: int,
    contours: Sequence[Contour],
) -> Optional[int]:
    """Индекс ближайшего родителя — минимального контура, содержащего данный."""
    inner = contours[index]
    inner_area = abs(_polygon_area(inner.points))
    best_parent: Optional[int] = None
    best_area = float("inf")

    for candidate_idx, candidate in enumerate(contours):
        if candidate_idx == index:
            continue
        candidate_area = abs(_polygon_area(candidate.points))
        if candidate_area <= inner_area:
            continue
        if not contour_contains(candidate, inner):
            continue
        if candidate_area < best_area:
            best_area = candidate_area
            best_parent = candidate_idx

    return best_parent


def analyze_contour_nesting(contours: Sequence[Contour]) -> List[ContourNestingInfo]:
    """Строит дерево вложенности для списка контуров."""
    n = len(contours)
    parents: List[Optional[int]] = [
        find_immediate_parent(i, contours) for i in range(n)
    ]

    depths = [_compute_depth(i, parents) for i in range(n)]

    children_map: Dict[int, List[int]] = defaultdict(list)
    for idx, parent in enumerate(parents):
        if parent is not None:
            children_map[parent].append(idx)

    return [
        ContourNestingInfo(
            index=i,
            depth=depths[i],
            parent=parents[i],
            children=tuple(children_map[i]),
            is_fill_boundary=depths[i] % 2 == 0,
            is_hole=depths[i] % 2 == 1,
        )
        for i in range(n)
    ]


def _compute_depth(index: int, parents: Sequence[Optional[int]]) -> int:
    depth = 0
    current: Optional[int] = parents[index]
    visited = set()

    while current is not None:
        if current in visited:
            break
        visited.add(current)
        depth += 1
        current = parents[current]

    return depth


def get_direct_holes(
    fill_index: int,
    nesting: Sequence[ContourNestingInfo],
) -> Tuple[int, ...]:
    """Возвращает индексы непосредственных отверстий для контура заливки."""
    info = nesting[fill_index]
    return tuple(
        child_idx
        for child_idx in info.children
        if nesting[child_idx].is_hole
    )

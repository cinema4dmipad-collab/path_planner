import tempfile
import unittest
from pathlib import Path

import ezdxf
import numpy as np

from batch_planner import plan_dxf_contours, plan_file_contours, summarize_results
from contour_nesting import analyze_contour_nesting, contour_contains
from geometry import Contour
from planner import GEOM_EPSILON, PathPlanner, fill_segments_at_y


def assert_path_avoids_rect_holes(
    test_case: unittest.TestCase,
    path: np.ndarray,
    holes: list[tuple[float, float, float, float]],
    margin: float = 0.01,
) -> None:
    """Проверяет, что точки и сегменты не попадают внутрь прямоугольных отверстий."""
    for xmin, ymin, xmax, ymax in holes:
        inside = (
            (path[:, 0] > xmin + margin)
            & (path[:, 0] < xmax - margin)
            & (path[:, 1] > ymin + margin)
            & (path[:, 1] < ymax - margin)
        )
        test_case.assertFalse(
            np.any(inside),
            f"Точка траектории внутри отверстия ({xmin},{ymin})-({xmax},{ymax})",
        )

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue
        for t in (0.25, 0.5, 0.75):
            mid = p1 + t * (p2 - p1)
            for xmin, ymin, xmax, ymax in holes:
                if (
                    xmin + margin < mid[0] < xmax - margin
                    and ymin + margin < mid[1] < ymax - margin
                ):
                    test_case.fail(
                        f"Сегмент пересекает отверстие ({xmin},{ymin})-({xmax},{ymax}): "
                        f"{p1} -> {p2}, mid={mid}"
                    )


def write_dxf(draw) -> Path:
    doc = ezdxf.new()
    draw(doc.modelspace())
    path = Path(tempfile.mktemp(suffix=".dxf"))
    doc.saveas(path)
    return path


def regular_hexagon(center: tuple[float, float], radius: float) -> np.ndarray:
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    points = np.array([
        [cx + np.cos(angle) * radius, cy + np.sin(angle) * radius]
        for angle in angles
    ])
    return np.vstack([points, points[0]])


class TestContourNesting(unittest.TestCase):
    def test_detects_inner_contour(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        inner = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        self.assertTrue(contour_contains(outer, inner))
        nesting = analyze_contour_nesting([outer, inner])
        self.assertTrue(nesting[0].is_fill_boundary)
        self.assertTrue(nesting[1].is_hole)
        self.assertEqual(nesting[1].parent, 0)

    def test_separate_contours_not_nested(self):
        a = Contour(np.array([[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]), 0.1)
        b = Contour(np.array([[10, 0], [15, 0], [15, 5], [10, 5], [10, 0]]), 0.1)
        nesting = analyze_contour_nesting([a, b])
        self.assertEqual(nesting[0].depth, 0)
        self.assertEqual(nesting[1].depth, 0)
        self.assertFalse(nesting[1].is_hole)


class TestHoleFillExclusion(unittest.TestCase):
    def test_fill_segments_exclude_hole(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        segments = fill_segments_at_y(outer, 5.0, [hole])
        self.assertEqual(segments, [(0.0, 3.0), (7.0, 10.0)])

    def test_fill_segments_two_holes_same_scanline(self):
        outer = Contour(
            np.array([[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]]),
            0.1,
        )
        hole_a = Contour(np.array([[3, 3], [5, 3], [5, 7], [3, 7], [3, 3]]), 0.1)
        hole_b = Contour(np.array([[8, 3], [10, 3], [10, 7], [8, 7], [8, 3]]), 0.1)
        segments = fill_segments_at_y(outer, 5.0, [hole_a, hole_b])
        self.assertEqual(segments, [(0.0, 3.0), (5.0, 8.0), (10.0, 20.0)])

    def test_fill_segments_no_hole_intersection_on_scanline(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        segments = fill_segments_at_y(outer, 2.0, [hole])
        self.assertEqual(segments, [(0.0, 10.0)])

    def test_inner_contour_skipped_and_path_avoids_hole(self):
        path = write_dxf(
            lambda msp: (
                msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True),
                msp.add_lwpolyline([(3, 3), (7, 3), (7, 7), (3, 7)], close=True),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        results = plan_dxf_contours(path, tolerance=0.1, line_distance=2.0)
        self.assertEqual(len(results), 2)

        outer = next(r for r in results if r.area > 50)
        inner = next(r for r in results if r.area < 50)

        self.assertFalse(outer.skipped)
        self.assertEqual(outer.hole_count, 1)
        self.assertTrue(inner.skipped)
        self.assertIn("отверстие", inner.skip_reason)

        inside_hole = (
            (outer.path[:, 0] > 3.01)
            & (outer.path[:, 0] < 6.99)
            & (outer.path[:, 1] > 3.01)
            & (outer.path[:, 1] < 6.99)
        )
        self.assertFalse(
            np.any(inside_hole),
            "Траектория проходит через внутреннее отверстие",
        )

        for i in range(len(outer.path) - 1):
            p1 = outer.path[i]
            p2 = outer.path[i + 1]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            for t in (0.25, 0.5, 0.75):
                mid = p1 + t * (p2 - p1)
                if (
                    3.01 < mid[0] < 6.99
                    and 3.01 < mid[1] < 6.99
                ):
                    self.fail(
                        f"Сегмент траектории пересекает отверстие: "
                        f"{p1} -> {p2}, mid={mid}"
                    )

        # мост между сегментами на y=5 над отверстием не должен уходить внутрь по Y
        for i in range(len(outer.path) - 1):
            p1 = outer.path[i]
            p2 = outer.path[i + 1]
            if abs(p1[0] - p2[0]) >= 0.01:
                continue
            x = float(p1[0])
            if not (2.99 < x < 7.01):
                continue
            for y in (float(p1[1]), float(p2[1])):
                if abs(y - 5.0) >= 0.01:
                    continue
                other_y = float(p2[1] if abs(p1[1] - 5.0) < 0.01 else p1[1])
                if 3.01 < other_y < 6.99:
                    self.fail(
                        f"Обход между сегментами y=5 не должен идти через отверстие: "
                        f"{p1} -> {p2}"
                    )

    def test_path_avoids_two_holes(self):
        outer = Contour(
            np.array([[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]]),
            0.1,
        )
        hole_a = Contour(np.array([[3, 3], [5, 3], [5, 7], [3, 7], [3, 3]]), 0.1)
        hole_b = Contour(np.array([[8, 3], [10, 3], [10, 7], [8, 7], [8, 3]]), 0.1)
        planner = PathPlanner(outer, line_distance=2.0, holes=[hole_a, hole_b])
        path = planner.generate_path()
        self.assertGreater(len(path), 0)
        assert_path_avoids_rect_holes(
            self,
            path,
            [(3, 3, 5, 7), (8, 3, 10, 7)],
        )

    def test_path_with_rotated_fill_angle_avoids_hole(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        planner = PathPlanner(
            outer,
            line_distance=2.0,
            fill_angle=45.0,
            holes=[hole],
        )
        path = planner.generate_path()
        self.assertGreater(len(path), 0)
        assert_path_avoids_rect_holes(self, path, [(3, 3, 7, 7)])

    def test_no_horizontal_bridge_through_hole(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        planner = PathPlanner(outer, line_distance=2.0, holes=[hole])
        path = planner.generate_path()

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            if abs(p1[1] - p2[1]) > GEOM_EPSILON:
                continue
            y = float(p1[1])
            x_lo = min(float(p1[0]), float(p2[0]))
            x_hi = max(float(p1[0]), float(p2[0]))
            if x_lo < 6.99 and x_hi > 3.01 and 3.01 < y < 6.99:
                self.fail(
                    f"Горизонтальный переход через отверстие на y={y:.2f}: {p1} -> {p2}"
                )

    def test_hexagon_hole_travel_stays_local_and_inside_outer(self):
        outer = Contour(
            np.array([[0, 0], [20, 0], [20, 16], [0, 16], [0, 0]]),
            0.1,
        )
        hex_hole = Contour(regular_hexagon((10, 8), 3), 0.1)
        planner = PathPlanner(outer, line_distance=2.0, holes=[hex_hole])
        path = planner.generate_path()

        self.assertGreater(len(path), 0)
        self.assertTrue(np.all(path[:, 0] >= -GEOM_EPSILON))
        self.assertTrue(np.all(path[:, 0] <= 20 + GEOM_EPSILON))
        self.assertTrue(np.all(path[:, 1] >= -GEOM_EPSILON))
        self.assertTrue(np.all(path[:, 1] <= 16 + GEOM_EPSILON))

        for start_idx, end_idx, kind in planner.get_path_segments():
            if kind != "travel":
                continue
            travel = path[start_idx : end_idx + 1]
            self.assertFalse(
                planner._polyline_crosses_hole_interior(travel),
                f"Переход пересекает внутренний шестиугольник: {travel}",
            )
            self.assertFalse(
                planner._polyline_leaves_outer(travel),
                f"Переход выходит за внешний контур: {travel}",
            )
            if len(travel) >= 3:
                route_y = float(travel[1][1])
                self.assertLessEqual(
                    route_y,
                    11.05,
                    f"Обход шестиугольника слишком высоко: {travel}",
                )
                self.assertGreaterEqual(
                    route_y,
                    4.45,
                    f"Обход шестиугольника слишком низко: {travel}",
                )

    def test_hole_clearance_widens_fill_gap(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        without = fill_segments_at_y(outer, 5.0, [hole])
        with_clearance = fill_segments_at_y(outer, 5.0, [hole.offset_outward(1.0)])
        self.assertEqual(without, [(0.0, 3.0), (7.0, 10.0)])
        self.assertEqual(with_clearance, [(0.0, 2.0), (8.0, 10.0)])

    def test_path_respects_hole_clearance(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 7]]), 0.1)
        planner = PathPlanner(
            outer,
            line_distance=2.0,
            holes=[hole],
            hole_clearance=1.0,
        )
        path = planner.generate_path()
        self.assertGreater(len(path), 0)
        assert_path_avoids_rect_holes(self, path, [(2, 2, 8, 8)], margin=0.05)

    def test_strict_clearance_forbids_boundary_contact(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        expanded = hole.offset_outward(1.0)
        boundary_point = np.array([2.0, 5.0])

        allow = PathPlanner(
            outer,
            line_distance=2.0,
            holes=[hole],
            hole_clearance=1.0,
            allow_clearance_contact=True,
        )
        strict = PathPlanner(
            outer,
            line_distance=2.0,
            holes=[hole],
            hole_clearance=1.0,
            allow_clearance_contact=False,
        )

        self.assertFalse(
            allow._point_violates_clearance(boundary_point, allow.planning_holes[0])
        )
        self.assertTrue(
            strict._point_violates_clearance(boundary_point, strict.planning_holes[0])
        )

        allow_segments = fill_segments_at_y(outer, 5.0, [expanded])
        strict_segments = fill_segments_at_y(
            outer,
            5.0,
            [expanded],
            endpoint_inset=strict._fill_endpoint_inset(),
        )
        self.assertEqual(allow_segments[0][1], 2.0)
        self.assertLess(strict_segments[0][1], 2.0)

    def test_two_phase_fill_returns_to_hole_gaps(self):
        outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
        hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
        planner = PathPlanner(outer, line_distance=2.0, holes=[hole])
        path = planner.generate_path()
        segments = planner.get_path_segments()

        self.assertGreater(len(path), 0)
        fill_count = sum(1 for _, _, kind in segments if kind == "fill")
        self.assertGreater(fill_count, 1, "Должны быть и основная змейка, и дозаполнение")

        fill_segments = [
            path[start:end + 1]
            for start, end, kind in segments
            if kind == "fill"
        ]
        self.assertGreater(
            len(fill_segments),
            1,
            "Пропуски у отверстия должны заполняться отдельными проходами",
        )
        assert_path_avoids_rect_holes(self, path, [(3, 3, 7, 7)])

    def test_snake_alternates_direction_without_holes(self):
        outer = Contour(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]),
            0.1,
        )
        planner = PathPlanner(outer, line_distance=2.0)
        rows = planner._group_lines_by_y(planner._generate_fill_lines())
        self.assertGreaterEqual(len(rows), 2)

        for row_idx, (_, row_lines) in enumerate(rows[:4]):
            left_to_right = row_idx % 2 == 0
            for line in row_lines:
                start, end = planner._orient_segment(line, left_to_right)
                if left_to_right:
                    self.assertLessEqual(start[0], end[0] + 1e-6)
                else:
                    self.assertGreaterEqual(start[0], end[0] - 1e-6)


class TestBatchPlanner(unittest.TestCase):
    def test_plan_all_closed_contours_in_dxf(self):
        path = write_dxf(
            lambda msp: (
                msp.add_lwpolyline([(0, 0), (5, 0), (5, 5), (0, 5)], close=True),
                msp.add_lwpolyline([(10, 0), (20, 0), (20, 10), (10, 10)], close=True),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        results = plan_dxf_contours(path, tolerance=0.1, line_distance=2.0)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not r.skipped for r in results))

        summary = summarize_results(results)
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["skipped"], 0)

    def test_open_contour_skipped_in_batch(self):
        path = write_dxf(
            lambda msp: (
                msp.add_lwpolyline([(0, 0), (5, 0), (5, 5), (0, 5)], close=True),
                msp.add_lwpolyline(
                    [(-5, 10), (-5, 0), (0, -4), (5, 0), (5, 10)],
                    close=False,
                ),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        results = plan_dxf_contours(path, tolerance=0.1, line_distance=2.0)
        summary = summarize_results(results)
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped"], 1)

    def test_plan_txt_single_contour(self):
        txt_path = Path(tempfile.mktemp(suffix=".txt"))
        txt_path.write_text("0 0\n10 0\n10 10\n0 10\n0 0\n", encoding="utf-8")
        self.addCleanup(lambda: txt_path.unlink(missing_ok=True))

        results = plan_file_contours(txt_path, tolerance=0.1, line_distance=2.0)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0].path), 0)


if __name__ == "__main__":
    unittest.main()

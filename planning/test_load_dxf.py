import tempfile
import unittest
from pathlib import Path

import ezdxf

from load_dxf import ContourSelection, DXFLoader, load_dxf, load_dxf_contours


def write_dxf(draw) -> Path:
    doc = ezdxf.new()
    draw(doc.modelspace())
    path = Path(tempfile.mktemp(suffix=".dxf"))
    doc.saveas(path)
    return path


class TestDXFLoader(unittest.TestCase):
    def test_load_closed_rectangle(self):
        path = write_dxf(
            lambda msp: msp.add_lwpolyline(
                [(0, 0), (10, 0), (10, 8), (0, 8)],
                close=True,
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        points = load_dxf(path, tolerance=0.1)
        self.assertEqual(points.shape[1], 2)
        self.assertGreaterEqual(len(points), 4)

    def test_select_largest_closed_contour(self):
        path = write_dxf(
            lambda msp: (
                msp.add_lwpolyline([(0, 0), (5, 0), (5, 5), (0, 5)], close=True),
                msp.add_lwpolyline([(0, 0), (20, 0), (20, 10), (0, 10)], close=True),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        contour = DXFLoader(tolerance=0.1).load_selected_contour(path)
        self.assertAlmostEqual(contour.area, 200.0, delta=10.0)
        self.assertTrue(contour.is_closed)

    def test_chain_line_segments_into_closed_contour(self):
        path = write_dxf(
            lambda msp: (
                msp.add_line((0, 0), (10, 0)),
                msp.add_line((10, 0), (10, 6)),
                msp.add_line((10, 6), (0, 6)),
                msp.add_line((0, 6), (0, 0)),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        contours = load_dxf_contours(path, tolerance=0.1)
        self.assertEqual(len(contours), 1)
        self.assertTrue(contours[0].is_closed)
        self.assertAlmostEqual(contours[0].area, 60.0, delta=5.0)

    def test_open_contour_rejected_when_closed_only(self):
        path = write_dxf(
            lambda msp: msp.add_lwpolyline(
                [(-5, 10), (-5, 0), (0, -4), (5, 0), (5, 10)],
                close=False,
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        with self.assertRaises(Exception) as ctx:
            load_dxf(path, tolerance=0.1, closed_only=True)
        self.assertIn("замкнут", str(ctx.exception).lower())

    def test_open_contour_can_be_loaded_when_allowed(self):
        path = write_dxf(
            lambda msp: msp.add_lwpolyline(
                [(-5, 10), (-5, 0), (0, -4), (5, 0), (5, 10)],
                close=False,
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        points = load_dxf(
            path,
            tolerance=0.1,
            closed_only=False,
            selection=ContourSelection.FIRST,
        )
        self.assertGreaterEqual(len(points), 3)

    def test_layer_filter(self):
        path = write_dxf(
            lambda msp: (
                msp.add_lwpolyline(
                    [(0, 0), (4, 0), (4, 4), (0, 4)],
                    close=True,
                    dxfattribs={"layer": "outline"},
                ),
                msp.add_lwpolyline(
                    [(0, 0), (20, 0), (20, 20), (0, 20)],
                    close=True,
                    dxfattribs={"layer": "ignore"},
                ),
            )
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        contour = DXFLoader(tolerance=0.1).load_selected_contour(path, layer="outline")
        self.assertEqual(contour.layer, "outline")
        self.assertAlmostEqual(contour.area, 16.0, delta=2.0)

    def test_circle_discretization_uses_chord_error(self):
        path = write_dxf(lambda msp: msp.add_circle((0, 0), radius=10))
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        for tolerance, max_points in ((0.1, 30), (0.5, 15), (2.0, 12)):
            with self.subTest(tolerance=tolerance):
                points = load_dxf(path, tolerance=tolerance)
                self.assertLessEqual(
                    len(points),
                    max_points,
                    f"Слишком много точек окружности при tolerance={tolerance}",
                )


if __name__ == "__main__":
    unittest.main()

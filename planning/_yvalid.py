import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[-12, -45], [32, -45], [32, 55], [-12, 55], [-12, -45]]),
    0.1,
)
holes = [Contour(np.array([[0, -30], [15, -30], [15, -10], [0, -10], [0, -30]]), 0.1)]
planner = PathPlanner(
    outer,
    line_distance=2.0,
    fill_angle=45.0,
    holes=holes,
    hole_clearance=1.0,
)

lines = []
for y in [planner.y_max, planner.y_min, 8.6, -21.71, 30, 50]:
    valid = planner._horizontal_segment_is_valid_bridge(4.73, 22.14, y)
    inside = planner._horizontal_segment_inside_outer(4.73, 22.14, y)
    holes = planner._horizontal_segment_crosses_holes(4.73, 22.14, y)
    lines.append(f"y={y:.2f} valid={valid} inside={inside} holes={holes}")

open("_yvalid.txt", "w").write("\n".join(lines))

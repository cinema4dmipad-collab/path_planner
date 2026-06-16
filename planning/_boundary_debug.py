import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[-12, -45], [32, -45], [32, 55], [-12, 55], [-12, -45]]),
    0.1,
)
holes = [
    Contour(np.array([[0, -30], [15, -30], [15, -10], [0, -10], [0, -30]]), 0.1),
    Contour(np.array([[-8, -5], [8, -5], [8, 15], [-8, 15], [-8, -5]]), 0.1),
]
planner = PathPlanner(
    outer,
    line_distance=2.0,
    fill_angle=45.0,
    holes=holes,
    hole_clearance=1.0,
)

ys = planner._boundary_clearance_route_ys(4.73, 22.14, -21.71, -21.71)
lines = [f"boundary ys {ys}"]
for direction in ("above", "below"):
    y = planner._find_boundary_safe_route_y(
        4.73, 22.14, direction=direction, reference_y=-21.71
    )
    lines.append(f"{direction} {y}")

open("_boundary.txt", "w", encoding="utf-8").write("\n".join(lines))

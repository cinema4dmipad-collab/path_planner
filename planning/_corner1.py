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
route = np.array([
    [22.14, -21.71],
    [22.14, -22.87741699796952],
    [5.40685424949238, -22.87741699796952],
    [4.73, -22.87741699796952],
    [4.73, -21.71],
])

lines = []
lines.append(f"safe {planner._polyline_is_safe_bridge(route)}")
for idx in range(len(route) - 1):
    seg = np.array([route[idx], route[idx + 1]])
    lines.append(
        f"seg {idx}: hole={planner._polyline_crosses_hole_interior(seg)} "
        f"outer={planner._polyline_leaves_outer(seg)} {seg.tolist()}"
    )

open("_corner1.txt", "w").write("\n".join(lines))

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
start = np.array([22.14, -21.71])
end = np.array([4.73, -21.71])
route_y = 8.63884447758931
route = np.array([
    start,
    [start[0], route_y],
    [end[0], route_y],
    end,
])

lines = []
lines.append(f"route {route.tolist()}")
lines.append(f"hole {planner._polyline_crosses_hole_interior(route)}")
lines.append(f"outer {planner._polyline_leaves_outer(route)}")
for idx in range(len(route) - 1):
    seg = np.array([route[idx], route[idx + 1]])
    lines.append(
        f"seg {idx} hole={planner._polyline_crosses_hole_interior(seg)} "
        f"outer={planner._polyline_leaves_outer(seg)} {seg.tolist()}"
    )

open("_seg.txt", "w", encoding="utf-8").write("\n".join(lines))

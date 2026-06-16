import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[-12, -45], [32, -45], [32, 55], [-12, 55], [-12, -55]]),
    0.1,
)
# fix typo
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
hole = planner.planning_holes[0]

lines = []
for name, routes in [
    ("side", planner._side_routes(start, end, hole)),
    ("corner", planner._corner_routes(start, end, hole)),
]:
    lines.append(f"{name} routes {len(routes)}")
    for route in routes:
        safe = planner._polyline_is_safe_bridge(route)
        lines.append(
            f"  safe={safe} hole={planner._polyline_crosses_hole_interior(route)} "
            f"outer={planner._polyline_leaves_outer(route)} {route.tolist()}"
        )

open("_side_corner.txt", "w", encoding="utf-8").write("\n".join(lines))

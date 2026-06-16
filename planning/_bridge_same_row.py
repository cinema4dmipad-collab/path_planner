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

with open("_bridge_same_row.txt", "w", encoding="utf-8") as out:
    out.write(f"direct safe {planner._polyline_is_safe_bridge(np.array([start, end]))}\n")
    out.write(f"crossing holes {planner._crossing_holes(start, end)}\n")
    safe_ys = planner._enumerate_safe_route_ys(float(start[0]), float(end[0]))
    out.write(f"safe ys {len(safe_ys)} {safe_ys}\n")
    cands = planner._build_bridge_candidates(
        start, end, current_y=-21.71, next_y=-21.71
    )
    out.write(f"candidates {len(cands)}\n")
    for route in cands[:8]:
        out.write(f"  {route.tolist()}\n")

print("done")

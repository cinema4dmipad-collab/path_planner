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
start = np.array([22.135533905932732, -21.705086527633213])
end = np.array([4.734523779156071, -21.705086527633213])
route_y = 8.63884447758931
bx = 41.934523779156066

route = np.array([
    start,
    [bx, start[1]],
    [bx, route_y],
    [end[0], route_y],
    end,
])
print("safe", planner._polyline_is_safe_bridge(route))
for idx in range(len(route) - 1):
    seg = np.array([route[idx], route[idx + 1]])
    print(idx, planner._polyline_crosses_hole_interior(seg), planner._polyline_leaves_outer(seg))

cands = planner._build_bridge_candidates(start, end, current_y=start[1], next_y=end[1])
print("candidates", len(cands))

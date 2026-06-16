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
bypass_x = 33.48401871576773
route_y = 8.63884447758931

for bypass_x in [33.48, 5.41, -2.0, 25.0]:
    route = np.array([
        start,
        [bypass_x, start[1]],
        [bypass_x, route_y],
        [end[0], route_y],
        end,
    ])
    print(
        f"bypass_x={bypass_x:.2f} safe={planner._polyline_is_safe_bridge(route)} "
        f"hole={planner._polyline_crosses_hole_interior(route)} "
        f"outer={planner._polyline_leaves_outer(route)}"
    )

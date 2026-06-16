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

for x in [41.934, 36.6, 33.5, 22.135]:
    seg = np.array([[x, -21.705], [x, 8.64]])
    print(
        f"x={x:.2f} hole={planner._polyline_crosses_hole_interior(seg)} "
        f"outer={planner._polyline_leaves_outer(seg)}"
    )

start = np.array([22.135533905932732, -21.705086527633213])
end = np.array([4.734523779156071, -21.705086527633213])
route_y = 8.63884447758931

for bx in [36.614833995939044, 33.5, 41.934]:
    route = np.array([
        start,
        [bx, start[1]],
        [bx, route_y],
        [end[0], route_y],
        end,
    ])
    print("bx", bx, "safe", planner._polyline_is_safe_bridge(route))

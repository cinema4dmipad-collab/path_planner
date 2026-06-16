import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(np.array([[0, 0], [30, 0], [30, 50], [0, 50], [0, 0]]), 0.1)
hole = Contour(np.array([[10, 15], [20, 15], [20, 35], [10, 35], [10, 15]]), 0.1)
planner = PathPlanner(
    outer, line_distance=2.0, fill_angle=45.0, holes=[hole], hole_clearance=1.0
)
start = np.array([16.60, 16.60])
end = np.array([-18.60, 18.60])
print("start segs", planner._fill_x_extents_at_y(float(start[1])))
print("end segs", planner._fill_x_extents_at_y(float(end[1])))

for bx in [-16.6, -3.87, -3.2, 16.6, -18.6, -5.87, -1.2, 18.6, -30, 30]:
    route = np.array([start, [bx, start[1]], [bx, end[1]], end])
    if planner._polyline_is_safe_bridge(route):
        print("OK bx", bx, route.tolist())

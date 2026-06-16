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
print("safe ys", planner._enumerate_safe_route_ys(float(start[0]), float(end[0])))
print("wide routes", len(planner._wide_detour_routes(start, end)))
for route in planner._wide_detour_routes(start, end)[:20]:
    if planner._polyline_is_safe_bridge(route):
        print("FOUND", route.tolist())
        break
else:
    print("none safe in first 20")

# try 3-step: up at start.x, across at end.y, to end
route = np.array([start, [start[0], end[1]], end])
print("simple L", planner._polyline_is_safe_bridge(route))
route2 = np.array([start, [end[0], start[1]], end])
print("simple L2", planner._polyline_is_safe_bridge(route2))

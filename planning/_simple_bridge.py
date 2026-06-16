import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[0, 0], [30, 0], [30, 50], [0, 50], [0, 0]]),
    0.1,
)
hole = Contour(
    np.array([[10, 15], [20, 15], [20, 35], [10, 35], [10, 15]]),
    0.1,
)
planner = PathPlanner(
    outer,
    line_distance=2.0,
    fill_angle=45.0,
    holes=[hole],
    hole_clearance=1.0,
)
start = np.array([16.60, 16.60])
end = np.array([-18.60, 18.60])
try:
    route = planner._bridge_points(start, end, current_y=16.60, next_y=18.60)
    print("OK", len(route), route.tolist())
except Exception as exc:
    print("FAIL", exc)
    cands = planner._build_bridge_candidates(
        start, end, current_y=16.60, next_y=18.60
    )
    print("candidates", len(cands))

path = planner.generate_path()
print("path len", len(path))

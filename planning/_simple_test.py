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
planner = PathPlanner(outer, line_distance=2.0, holes=[hole], hole_clearance=0.0)
start = np.array([25.0, 25.0])
end = np.array([5.0, 25.0])
cands = planner._build_bridge_candidates(start, end, current_y=25.0, next_y=25.0)
print("no angle clearance0", len(cands))
path = planner.generate_path()
print("path", len(path))

planner2 = PathPlanner(
    outer, line_distance=2.0, fill_angle=45.0, holes=[hole], hole_clearance=1.0
)
path2 = planner2.generate_path()
print("angle45 clearance1", len(path2))

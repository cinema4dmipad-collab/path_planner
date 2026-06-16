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

for y in [-21.71, -15, -10, 0, 5, 8.64, 15]:
    pt = np.array([4.73, y])
    print(y, planner._point_inside_or_on_outer(pt))

for y in [-21.71, -15, -10, 0, 5, 8.64]:
    pt = np.array([34.0, y])
    print("x34", y, planner._point_inside_or_on_outer(pt))

import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[-12, -45], [32, -45], [32, 55], [-12, 55], [-12, -45]]),
    0.1,
)
holes = [
    Contour(np.array([[0, -30], [15, -30], [15, -10], [0, -10], [0, -30]]), 0.1),
]
planner = PathPlanner(
    outer,
    line_distance=2.0,
    fill_angle=45.0,
    holes=holes,
    hole_clearance=1.0,
)
hole = planner.planning_holes[0]
print("planning hole bounds", planner._hole_bounds(hole))
print("points sample", hole.points[:5])

seg = np.array([[22.14, -21.71], [22.14, 8.64]])
for i in range(13):
    t = i / 12.0
    pt = seg[0] + t * (seg[1] - seg[0])
    bad = any(
        planner._point_violates_clearance(pt, h)
        for h in planner.planning_holes
    )
    if bad:
        print(f"t={t:.2f} pt={pt} BAD")

print("crosses", planner._polyline_crosses_hole_interior(seg))

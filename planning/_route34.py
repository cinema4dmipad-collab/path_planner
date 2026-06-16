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

seg = np.array([[4.73, 8.64], [4.73, -21.71]])
for i in range(13):
    t = i / 12.0
    pt = seg[0] + t * (seg[1] - seg[0])
    bad = planner._polyline_crosses_hole_interior(np.array([pt, pt]))
    viol = any(planner._point_violates_clearance(pt, h) for h in planner.planning_holes)
    inside = any(planner._point_in_hole_interior(pt, h) for h in planner.planning_holes)
    if viol or inside:
        print(f"t={t:.2f} y={pt[1]:.2f} viol={viol} inside={inside}")

route = np.array([
    [22.14, -21.71],
    [34.0, -21.71],
    [34.0, 8.64],
    [4.73, 8.64],
    [4.73, -21.71],
])
print("route safe", planner._polyline_is_safe_bridge(route))
for idx in range(len(route) - 1):
    seg = np.array([route[idx], route[idx + 1]])
    print(
        idx,
        planner._polyline_crosses_hole_interior(seg),
        planner._polyline_leaves_outer(seg),
        seg.tolist(),
    )

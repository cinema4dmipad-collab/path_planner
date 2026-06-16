import numpy as np
from geometry import Contour
from planner import PathPlanner, fill_segments_at_y

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

y = -21.705086527633213
segs = fill_segments_at_y(
    planner.working_contour,
    y,
    planner.planning_holes,
    tolerance=planner.geom_epsilon,
    endpoint_inset=planner._fill_endpoint_inset(),
)
print("segments at y", y, segs)
for x in (4.73, 4.734523779156071):
    pt = np.array([x, y])
    print(x, planner._point_inside_or_on_outer(pt))

lines = [l for l in planner._generate_fill_lines() if abs(l[0, 1] - y) < 0.01]
print("lines", len(lines))
for line in lines[:5]:
    print(line.tolist(), planner._point_inside_or_on_outer(line[0]), planner._point_inside_or_on_outer(line[1]))

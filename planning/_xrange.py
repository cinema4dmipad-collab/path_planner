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

for y in [8.64, -21.705, 20.0]:
    segs = fill_segments_at_y(
        planner.working_contour, y, planner.planning_holes,
        endpoint_inset=planner._fill_endpoint_inset(),
    )
    print("y", y, "segs", segs)

# find rightmost x inside at y=8.64 on same row as fill
y = 8.63884447758931
for x in np.linspace(4, 45, 20):
    print(x, planner._point_inside_or_on_outer(np.array([x, y])))

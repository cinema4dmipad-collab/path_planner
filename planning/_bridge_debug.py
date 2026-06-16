import numpy as np
from geometry import Contour
from planner import PathPlanner

outer = Contour(
    np.array([[-12, -45], [32, -45], [32, 55], [-12, 55], [-12, -45]]),
    0.1,
)
holes = [
    Contour(np.array([[0, -30], [15, -30], [15, -10], [0, -10], [0, -30]]), 0.1),
    Contour(np.array([[-8, -5], [8, -5], [8, 15], [-8, 15], [-8, -5]]), 0.1),
]
planner = PathPlanner(
    outer,
    line_distance=2.0,
    fill_angle=45.0,
    holes=holes,
    hole_clearance=1.0,
)

lines = planner._generate_fill_lines()
print("fill lines", len(lines))
if lines:
    print("first line", lines[0].tolist())
    print("last line", lines[-1].tolist())

primary, deferred = planner._split_primary_deferred(lines)
print("primary", len(primary), "deferred", len(deferred))

start = np.array([28.35, 50.85])
end = np.array([-4.18, -41.15])

# use actual endpoints from fill if our test coords are wrong
if deferred:
    d0 = deferred[0]
    start = primary[-1][1] if primary else d0[0]
    end = d0[0]
    print("using actual start", start.tolist(), "end", end.tolist())

print("bounds", planner.outer_bounds)
print("y_min/max", planner.y_min, planner.y_max)

safe_ys = planner._enumerate_safe_route_ys(float(start[0]), float(end[0]))
print("safe route ys", len(safe_ys), safe_ys[:10] if safe_ys else [])

for route_y in safe_ys[:5]:
    route = np.array([
        start,
        [start[0], route_y],
        [end[0], route_y],
        end,
    ])
    hole = planner._polyline_crosses_hole_interior(route)
    outer = planner._polyline_leaves_outer(route)
    safe = planner._polyline_is_safe_bridge(route)
    print(f"  y={route_y:.2f}: safe={safe} hole={hole} outer={outer}")

candidates = planner._build_bridge_candidates(
    start, end, current_y=50.85, next_y=-41.15
)
print("candidates", len(candidates))

for i, route in enumerate(planner._outer_detour_routes(start, end)):
    hole = planner._polyline_crosses_hole_interior(route)
    outer = planner._polyline_leaves_outer(route)
    safe = planner._polyline_is_safe_bridge(route)
    print(
        f"detour {i}: safe={safe} hole={hole} outer={outer} "
        f"len={planner._polyline_length(route):.1f}"
    )

# try horizontal at outer walls
xmin, xmax, ymin, ymax = planner.outer_bounds
for route_y in [ymin, ymax, ymin - 2, ymax + 2, -46, 58]:
    route = np.array([
        start,
        [start[0], route_y],
        [end[0], route_y],
        end,
    ])
    safe = planner._polyline_is_safe_bridge(route)
    hole = planner._polyline_crosses_hole_interior(route)
    outer = planner._polyline_leaves_outer(route)
    print(f"y={route_y}: safe={safe} hole={hole} outer={outer}")

try:
    path = planner.generate_path()
    print("path len", len(path))
except Exception as exc:
    print("path FAIL", exc)

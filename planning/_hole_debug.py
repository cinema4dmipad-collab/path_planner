import tempfile
from pathlib import Path

import ezdxf

from batch_planner import plan_dxf_contours
from planner import PathPlanner
from geometry import Contour
import numpy as np

out = []

doc = ezdxf.new()
msp = doc.modelspace()
msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True)
msp.add_lwpolyline([(3, 3), (7, 3), (7, 7), (3, 7)], close=True)
dxf_path = Path(tempfile.mktemp(suffix=".dxf"))
doc.saveas(dxf_path)

results = plan_dxf_contours(dxf_path, tolerance=0.1, line_distance=2.0)
for r in results:
    out.append(
        f"{r.label} skipped={r.skipped} reason={r.skip_reason!r} "
        f"path={len(r.path)} holes={r.hole_count}"
    )

outer = Contour(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]), 0.1)
hole = Contour(np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]), 0.1)
try:
    p = PathPlanner(outer, line_distance=2.0, holes=[hole], tolerance=0.1)
    path = p.generate_path()
    out.append(f"direct planner path len={len(path)}")
except Exception as exc:
    out.append(f"direct planner FAILED: {type(exc).__name__}: {exc}")

dxf_path.unlink()
Path("_hole_debug.txt").write_text("\n".join(out), encoding="utf-8")

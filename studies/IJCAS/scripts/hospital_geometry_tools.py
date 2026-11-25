"""Conservative grid projection of exported hospital collision meshes."""

import math

import numpy as np


def clip_height(vertices, lower, upper):
    """Clip a convex face to a vertical slab, retaining edges and points."""
    polygon = [np.asarray(v, dtype=float) for v in vertices]
    for height, sign in ((lower, 1), (upper, -1)):
        result = []
        if not polygon:
            break
        previous = polygon[-1]
        previous_inside = sign * (previous[2] - height) >= -1e-12
        for current in polygon:
            current_inside = sign * (current[2] - height) >= -1e-12
            if current_inside != previous_inside:
                fraction = (height - previous[2]) / (current[2] - previous[2])
                result.append(previous + fraction * (current - previous))
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        polygon = result
    return polygon


def polygon_cells(polygon, origin, resolution, shape):
    """SAT polygon vs cell squares; include all touched cells, including edges."""
    points = np.asarray(polygon, dtype=float)[:, :2]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Polygon must contain finite points")
    origin = np.asarray(origin)
    lo = np.floor((points.min(axis=0) - origin) / resolution - 1e-10).astype(int)
    hi = np.floor((points.max(axis=0) - origin) / resolution + 1e-10).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape[::-1]) - 1)
    if np.any(hi < lo):
        return np.array([], dtype=int), np.array([], dtype=int)
    columns, rows = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    centers_x = origin[0] + (columns + .5) * resolution
    centers_y = origin[1] + (rows + .5) * resolution
    selected = np.ones(columns.shape, dtype=bool)
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        edge = second - first
        if np.linalg.norm(edge) > 1e-12:
            axes.append(np.array([-edge[1], edge[0]]))
    for axis in axes:
        projected = points @ axis
        centers = centers_x * axis[0] + centers_y * axis[1]
        radius = resolution * .5 * np.abs(axis).sum()
        selected &= (centers + radius >= projected.min() - 1e-12)
        selected &= (centers - radius <= projected.max() + 1e-12)
    return rows[selected], columns[selected]


def footprint_polygon(pose, length=1.630, width=.965, clearance=.010):
    x, y, yaw = pose
    a, b = length / 2 + clearance, width / 2 + clearance
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(x + c * u - s * v, y + s * u + c * v)
                     for u, v in ((a, b), (a, -b), (-a, -b), (-a, b))])


def pose_collision(occupancy, pose, origin, resolution, clearance=.010):
    polygon = footprint_polygon(pose, clearance=clearance)
    maximum = np.asarray(origin) + np.asarray(occupancy.shape[::-1]) * resolution
    if np.any(polygon < origin) or np.any(polygon > maximum):
        return True
    rows, columns = polygon_cells(polygon, origin, resolution, occupancy.shape)
    return bool(np.any(occupancy[rows, columns]))


def mesh_mask(mesh, origin, resolution, shape, lower, upper):
    points = np.asarray(mesh["world_points_m"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid exported mesh vertices")
    counts = mesh["face_vertex_counts"]
    indices = np.asarray(mesh["face_vertex_indices"], dtype=int)
    if any(count != 3 for count in counts):
        raise ValueError("Expected triangular hospital collision meshes")
    if len(indices) != 3 * len(counts) or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("Invalid mesh topology")
    triangles = points[indices.reshape(-1, 3)]
    selected = (triangles[:, :, 2].max(axis=1) >= lower) & (triangles[:, :, 2].min(axis=1) <= upper)
    mask = np.zeros(shape, dtype=bool)
    for triangle in triangles[selected]:
        polygon = clip_height(triangle, lower, upper)
        if polygon:
            rows, columns = polygon_cells(polygon, origin, resolution, shape)
            mask[rows, columns] = True
    return mask


def line_free_intervals(meshes, x_m, y_min, y_max, height_m):
    """Intersect authored triangular solids with a line; return clear intervals.

    Intersections with each closed mesh use parity; intervals from different
    meshes are unioned. This is an authored-geometry diagnostic, not PhysX truth.
    """
    occupied = []
    for mesh in meshes:
        points = np.asarray(mesh["world_points_m"], dtype=float)
        indices = np.asarray(mesh["face_vertex_indices"], dtype=int)
        if any(n != 3 for n in mesh["face_vertex_counts"]):
            raise ValueError("Line test requires triangular mesh")
        hits = []
        for triangle in points[indices.reshape(-1, 3)]:
            a, b, c = triangle
            matrix = np.array([[b[0]-a[0], c[0]-a[0]], [b[2]-a[2], c[2]-a[2]]])
            determinant = np.linalg.det(matrix)
            if abs(determinant) < 1e-12:
                continue
            uv = np.linalg.solve(matrix, [x_m-a[0], height_m-a[2]])
            if uv[0] >= -1e-9 and uv[1] >= -1e-9 and uv.sum() <= 1 + 1e-9:
                hits.append(float(a[1] + uv[0]*(b[1]-a[1]) + uv[1]*(c[1]-a[1])))
        hits.sort()
        unique = []
        for hit in hits:
            if not unique or abs(hit - unique[-1]) > 1e-7:
                unique.append(hit)
        if len(unique) % 2:
            raise ValueError("Odd line intersections; mesh closure needs inspection")
        occupied.extend(zip(unique[::2], unique[1::2]))
    intervals = []
    cursor = y_min
    for lo, hi in sorted(occupied):
        if hi <= y_min or lo >= y_max:
            continue
        lo, hi = max(lo, y_min), min(hi, y_max)
        if lo > cursor + 1e-9:
            intervals.append([cursor, lo])
        cursor = max(cursor, hi)
    if cursor < y_max:
        intervals.append([cursor, y_max])
    return intervals


def combine_material_meshes(meshes):
    """Restore one solid before parity tests on material-split mesh sections."""
    combined = {"world_points_m": [], "face_vertex_counts": [], "face_vertex_indices": []}
    for mesh in meshes:
        offset = len(combined["world_points_m"])
        combined["world_points_m"].extend(mesh["world_points_m"])
        combined["face_vertex_counts"].extend(mesh["face_vertex_counts"])
        combined["face_vertex_indices"].extend(i + offset for i in mesh["face_vertex_indices"])
    return combined

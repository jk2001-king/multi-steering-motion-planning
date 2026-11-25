"""Sampled 2D footprint-to-authored-AABB distances, never cooked clearance.

Exact rectangle distance at each pose, conditional on the exported AABBs.
Spatial interpolation is sampled; no continuous trajectory or 3D certificate.
"""

import math

import numpy as np


def densify_poses(poses, translation_m=.005, heading_deg=.25):
    values = np.asarray(poses, float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.isfinite(values).all()
            or not math.isfinite(translation_m) or translation_m <= 0.
            or not math.isfinite(heading_deg) or heading_deg <= 0.):
        raise ValueError('Finite XY/yaw poses and positive finite sample spacing required')
    out = [values[0].tolist()]
    for first, last in zip(values, values[1:]):
        turn = math.atan2(math.sin(last[2]-first[2]), math.cos(last[2]-first[2]))
        n = max(1, math.ceil(math.hypot(*(last[:2]-first[:2]))/translation_m),
            math.ceil(abs(turn)/math.radians(heading_deg)))
        out.extend([[first[0]+i/n*(last[0]-first[0]), first[1]+i/n*(last[1]-first[1]), first[2]+i/n*turn]
            for i in range(1, n+1)])
    return np.asarray(out)


def rectangle_aabb_distances(poses, bound, length=1.630, width=.965, clearance=0.):
    """Euclidean distance via SAT plus both polygons' vertex-to-rectangle distances."""
    p = np.asarray(poses, float)
    lo, hi = np.asarray(bound['minimum_m'][:2], float), np.asarray(bound['maximum_m'][:2], float)
    if (p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all()
            or not np.isfinite([*lo, *hi, length, width, clearance]).all()
            or np.any(lo > hi) or length <= 0. or width <= 0. or clearance < 0.):
        raise ValueError('Finite positive full footprint and ordered AABB required')
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    a, b = length/2+clearance, width/2+clearance
    center, half = (lo+hi)/2, (hi-lo)/2
    delta = center-p[:, :2]
    gaps = np.column_stack((np.abs(delta[:, 0])-(a*np.abs(c)+b*np.abs(s)+half[0]),
        np.abs(delta[:, 1])-(a*np.abs(s)+b*np.abs(c)+half[1]),
        np.abs(delta[:, 0]*c+delta[:, 1]*s)-(a+half[0]*np.abs(c)+half[1]*np.abs(s)),
        np.abs(-delta[:, 0]*s+delta[:, 1]*c)-(b+half[0]*np.abs(s)+half[1]*np.abs(c))))
    intersect = np.max(gaps, axis=1) <= 0.
    corners = np.array(((a, b), (a, -b), (-a, -b), (-a, b)))
    world = np.empty((len(p), 4, 2))
    world[:, :, 0] = p[:, 0, None]+c[:, None]*corners[:, 0]-s[:, None]*corners[:, 1]
    world[:, :, 1] = p[:, 1, None]+s[:, None]*corners[:, 0]+c[:, None]*corners[:, 1]
    body_to_box = np.linalg.norm(world-np.clip(world, lo, hi), axis=2).min(axis=1)
    box_corners = np.array(((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])))
    offset = box_corners[None, :, :]-p[:, None, :2]
    local_x = offset[:, :, 0]*c[:, None]+offset[:, :, 1]*s[:, None]
    local_y = -offset[:, :, 0]*s[:, None]+offset[:, :, 1]*c[:, None]
    box_to_body = np.hypot(np.maximum(np.abs(local_x)-a, 0.), np.maximum(np.abs(local_y)-b, 0.)).min(axis=1)
    return np.where(intersect, 0., np.minimum(body_to_box, box_to_body))


def sampled_separation(poses, colliders, clearance=0.):
    values = densify_poses(poses)
    if not colliders or len({row['path'] for row in colliders}) != len(colliders):
        raise ValueError('Nonempty unique original collider bounds required')
    minimum, nearest, pose_index = math.inf, None, None
    collisions = 0
    for row in colliders:
        distances = rectangle_aabb_distances(values, row['world_aligned_bound'], clearance=clearance)
        index = int(np.argmin(distances))
        collisions += int(np.count_nonzero(distances == 0.))
        if distances[index] < minimum:
            minimum, nearest, pose_index = float(distances[index]), row['path'], index
    return dict(minimum_sampled_rectangle_to_authored_AABB_distance_m=minimum,
        nearest_original_collider=nearest, minimum_at_sampled_world_xyyaw=values[pose_index].tolist(),
        sampled_pose_count=len(values), pose_collider_intersections=collisions,
        footprint=dict(length_m=1.630, width_m=.965, expansion_each_side_m=clearance),
        spatial_sample_spacing_m_deg=[.005, .25],
        authored_AABBs_are_conservative_at_each_sample_NOT_continuous_or_cooked_clearance=True)

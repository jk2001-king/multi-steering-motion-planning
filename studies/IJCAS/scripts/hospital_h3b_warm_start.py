"""Seeded continuous-curvature H3-B geometry optimization, NOT PhysX proof.

Uses the original legal counterphase seed and same 4 cusps, hospital map,
footprint, endpoints, curvature bound and reverse-resource limits. Curvature is
piecewise linear in travelled distance, with one shared value at every cusp.
This is a NEW search representation, not the old discrete-grid comparison.
Distance-field samples guide optimization only; final approval uses original
finite-cell polygon SAT at dense independently integrated poses.
"""

import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares

from hospital_geometry_tools import pose_collision
from hospital_h3_forward_connectivity import in_bounds
from hospital_h3b_policy import POLICY
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_planning_tools import local_to_world


CONTRACT = {"version": "H3B_seeded_continuous_curvature_v1", "mode": "coordinated_counterphase",
    "curvature_bound_1_m": .7, "initial_and_final_nominal_curvature": 0.,
    "nominal_curvature_shared_at_cusps": True, "optimization_substeps_per_control": 2,
    "audit_maximum_step_m": .0025, "field_is_guidance_only_not_collision_certificate": True,
    "same_search_revision_required_for_methods_in_any_future_comparison": True,
    "identified_rigid_body_dynamics_propagated": False, "actual_steering_or_PhysX_verified": False}


class SeedModel:
    def __init__(self, actions, straight_cusps=False):
        if not actions or any(a["mode"] != CONTRACT["mode"] or a["direction"] not in (-1, 1)
                or not math.isfinite(a["curvature_1_m"]) or abs(a["curvature_1_m"]) > .7 for a in actions):
            raise ValueError("Original bounded counterphase seed required")
        self.actions = actions
        self.n = len(actions)
        self.direction = np.asarray([a["direction"] for a in actions])
        self.leg = np.cumsum(np.r_[0, self.direction[1:] != self.direction[:-1]])
        self.leg_counts = np.bincount(self.leg)
        self.leg_directions = self.direction[np.r_[0, np.flatnonzero(np.diff(self.leg)) + 1]]
        self.cusps = np.flatnonzero(self.direction[1:] != self.direction[:-1]) + 1
        if len(self.cusps) > POLICY["maximum_direction_changes"]:
            raise ValueError("Original cusp resource limit exceeded")
        self.reference_lengths = self.leg_counts * .05
        self.fixed = {0, self.n} | (set(map(int, self.cusps)) if straight_cusps else set())
        self.free = np.asarray([i for i in range(self.n + 1) if i not in self.fixed], dtype=int)
        self.p = len(self.free) + len(self.leg_counts)
        self.k_jac = np.zeros((self.n + 1, self.p))
        self.k_jac[self.free, np.arange(len(self.free))] = 1.
        self.straight_cusps = bool(straight_cusps)

    def initial(self, extended_reverse=False):
        old = np.asarray([a["curvature_1_m"] for a in self.actions])
        knots = np.r_[0., (old[:-1] + old[1:]) / 2., 0.]
        knots[list(self.fixed)] = 0.
        lengths = self.reference_lengths.copy()
        if extended_reverse:
            lengths[self.leg_directions < 0] = np.linspace(.30, .35, np.sum(self.leg_directions < 0))
        return np.r_[knots[self.free], lengths]

    def bounds(self):
        low_length = np.maximum(.05, self.reference_lengths - .50)
        high_length = self.reference_lengths + .60
        low_length[self.leg_directions < 0] = .05
        high_length[self.leg_directions < 0] = POLICY["maximum_reverse_episode_m"]
        return np.r_[np.full(len(self.free), -.7), low_length], np.r_[np.full(len(self.free), .7), high_length]

    def unpack(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.p,) or not np.isfinite(z).all():
            raise ValueError("Finite parameter vector required")
        knots = np.zeros(self.n + 1)
        knots[self.free] = z[:len(self.free)]
        lengths = z[len(self.free):]
        if np.any(lengths <= 0.):
            raise ValueError("Positive geometric leg lengths required")
        return knots, lengths

    def integrate(self, z, substeps=2, jacobian=False, independent=False):
        knots, lengths = self.unpack(z)
        pose = np.zeros(3)
        derivative = np.zeros((3, self.p))
        poses = [pose.copy()]
        derivatives = [derivative.copy()]
        arc = [0.]
        curvatures = [knots[0]]
        controls = [-1]
        for i in range(self.n):
            count = max(substeps, math.ceil(lengths[self.leg[i]] / self.leg_counts[self.leg[i]] / .0025)) if independent else substeps
            ds = lengths[self.leg[i]] / self.leg_counts[self.leg[i]] / count * self.direction[i]
            dds = np.zeros(self.p)
            dds[len(self.free) + self.leg[i]] = self.direction[i] / self.leg_counts[self.leg[i]] / count
            for j in range(count):
                a, b = j / count, (j + 1) / count
                left = (1 - a) * knots[i] + a * knots[i + 1]
                right = (1 - b) * knots[i] + b * knots[i + 1]
                q = .375 * left + .125 * right
                middle = pose[2] + q * ds
                end_yaw = pose[2] + .5 * (left + right) * ds
                if independent:
                    # Simpson integration of the exact quadratic heading along
                    # this linear-curvature subinterval; separate from optimizer.
                    pose[:2] += ds / 6. * (np.asarray([math.cos(pose[2]), math.sin(pose[2])])
                        + 4. * np.asarray([math.cos(middle), math.sin(middle)])
                        + np.asarray([math.cos(end_yaw), math.sin(end_yaw)]))
                else:
                    pose[:2] += ds * np.asarray([math.cos(middle), math.sin(middle)])
                if jacobian:
                    dl = (1 - a) * self.k_jac[i] + a * self.k_jac[i + 1]
                    dr = (1 - b) * self.k_jac[i] + b * self.k_jac[i + 1]
                    dm = derivative[2] + ds * (.375 * dl + .125 * dr) + q * dds
                    derivative[0] += dds * math.cos(middle) - ds * math.sin(middle) * dm
                    derivative[1] += dds * math.sin(middle) + ds * math.cos(middle) * dm
                    derivative[2] += ds * .5 * (dl + dr) + .5 * (left + right) * dds
                pose[2] = end_yaw
                poses.append(pose.copy())
                if jacobian:
                    derivatives.append(derivative.copy())
                arc.append(arc[-1] + abs(ds))
                curvatures.append(right)
                controls.append(i)
        return np.asarray(poses), np.asarray(derivatives) if jacobian else None, np.asarray(arc), np.asarray(curvatures), np.asarray(controls)


class Objective:
    def __init__(self, model, manifest, occupied, origin, resolution, deadline=None, heartbeat=None):
        self.model, self.manifest = model, manifest
        self.origin, self.resolution = np.asarray(origin), resolution
        self.field = (distance_transform_edt(~occupied) - distance_transform_edt(occupied)) * resolution
        self.goal = np.asarray(manifest["local_goal_xyyaw"])
        self.start = manifest["start_xyyaw"]
        self.rotation = np.asarray([[math.cos(self.start[2]), -math.sin(self.start[2])], [math.sin(self.start[2]), math.cos(self.start[2])]])
        self.points = np.asarray([(x, y) for x in np.linspace(-.825, .825, 7) for y in np.linspace(-.4925, .4925, 5)])
        self.margin = resolution * math.sqrt(2.) / 2. + .003
        self.deadline, self.heartbeat = deadline, heartbeat
        self.cached = None
        self.best = None
        self.calls = 0
        self.started = self.last_watch = time.perf_counter()

    def field_query(self, points):
        uv = (points - self.origin) / self.resolution - .5
        base = np.floor(uv).astype(int)
        valid = (base[:, 0] >= 0) & (base[:, 1] >= 0) & (base[:, 0] < self.field.shape[1] - 1) & (base[:, 1] < self.field.shape[0] - 1)
        ix = np.clip(base[:, 0], 0, self.field.shape[1] - 2)
        iy = np.clip(base[:, 1], 0, self.field.shape[0] - 2)
        tx, ty = (uv - base).T
        a, b, c, d = self.field[iy, ix], self.field[iy, ix + 1], self.field[iy + 1, ix], self.field[iy + 1, ix + 1]
        value = (1 - tx) * (1 - ty) * a + tx * (1 - ty) * b + (1 - tx) * ty * c + tx * ty * d
        gradient = np.column_stack(((1 - ty) * (b - a) + ty * (d - c), (1 - tx) * (c - a) + tx * (d - b))) / self.resolution
        value[~valid] = -1.
        gradient[~valid] = 0.
        return value, gradient

    def evaluate(self, z):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError("Warm-start optimization wall-time budget")
        if self.cached is not None and np.array_equal(z, self.cached[0]):
            return self.cached[1:]
        poses, jac, _, _, _ = self.model.integrate(z, jacobian=True)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        px, py = self.points.T
        x = poses[:, 0, None] + c[:, None] * px - s[:, None] * py
        y = poses[:, 1, None] + s[:, None] * px + c[:, None] * py
        local = np.stack((x, y), axis=-1).reshape(-1, 2)
        world = local @ self.rotation.T + self.start[:2]
        values, world_gradient = self.field_query(world)
        gradient = world_gradient @ self.rotation
        deficit = np.maximum(0., self.margin - values)
        dx = jac[:, 0, None, :] + (-s[:, None] * px - c[:, None] * py)[:, :, None] * jac[:, 2, None, :]
        dy = jac[:, 1, None, :] + (c[:, None] * px - s[:, None] * py)[:, :, None] * jac[:, 2, None, :]
        obstacle_jac = -300. * (gradient[:, 0, None] * dx.reshape(-1, self.model.p) + gradient[:, 1, None] * dy.reshape(-1, self.model.p))
        obstacle_jac[deficit <= 0.] = 0.
        scale = np.asarray([100., 100., 50.])
        end_res = (poses[-1] - self.goal) * scale
        end_jac = jac[-1] * scale[:, None]
        knots, lengths = self.model.unpack(z)
        smooth = .03 * np.diff(knots)
        smooth_jac = .03 * np.diff(self.model.k_jac, axis=0)
        half_y = .815 * abs(s) + .4825 * abs(c)
        dh = .815 * np.sign(s) * c - .4825 * np.sign(c) * s
        limits = (poses[:, 0] - (self.goal[0] + .4), -.3 - poses[:, 0], poses[:, 1] + half_y - 1.99, -1.99 - poses[:, 1] + half_y)
        limit_jac = (jac[:, 0], -jac[:, 0], jac[:, 1] + dh[:, None] * jac[:, 2], -jac[:, 1] + dh[:, None] * jac[:, 2])
        residuals = [end_res, 300. * deficit, smooth]
        derivatives = [end_jac, obstacle_jac, smooth_jac]
        for value, derivative in zip(limits, limit_jac):
            residuals.append(100. * np.maximum(value, 0.))
            derivatives.append(100. * derivative * (value > 0.)[:, None])
        residual, derivative = np.concatenate(residuals), np.concatenate(derivatives)
        cost = float(residual @ residual)
        self.calls += 1
        if self.best is None or cost < self.best[0]:
            self.best = (cost, np.asarray(z).copy())
        self.cached = (np.asarray(z).copy(), residual, derivative)
        if self.heartbeat is not None and time.perf_counter() - self.last_watch >= 10.:
            self.heartbeat({"elapsed_s": time.perf_counter() - self.started, "calls": self.calls, "cost": cost,
                "position_m": float(np.linalg.norm(poses[-1, :2] - self.goal[:2])), "heading_deg": math.degrees(poses[-1, 2] - self.goal[2])})
            self.last_watch = time.perf_counter()
        return residual, derivative


def audit(model, z, manifest, occupied, origin, resolution):
    poses, _, arc, curvature, controls = model.integrate(z, independent=True)
    knots, lengths = model.unpack(z)
    native = NativeHospitalEnvelopeChecker(occupied, origin, resolution, manifest["start_xyyaw"])
    collided = native(poses)
    independent_collisions = 0
    if not collided:
        independent_collisions = sum(bool(pose_collision(occupied, local_to_world(pose, manifest["start_xyyaw"]), origin, resolution, .01)) for pose in poses)
    position = float(np.linalg.norm(poses[-1, :2] - manifest["local_goal_xyyaw"][:2]))
    heading = abs(float(poses[-1, 2] - manifest["local_goal_xyyaw"][2]))
    reverse = float(sum(lengths[model.leg_directions < 0]))
    checks = {"unchanged_full_envelope_and_clearance_clear": not collided and independent_collisions == 0,
        "original_position_under_25mm": position <= .025, "original_heading_under_2deg": heading <= math.radians(2.),
        "unchanged_search_bounds": in_bounds(poses, manifest["local_goal_xyyaw"]),
        "original_curvature_bound": max(abs(knots)) <= .7 + 1e-10,
        "original_reverse_episode_bound": all(length <= .5 + 1e-10 for length in lengths[model.leg_directions < 0]),
        "original_total_reverse_bound": reverse <= 1. + 1e-10,
        "original_direction_change_bound": len(model.cusps) <= 4,
        "initial_final_nominal_steering_straight": abs(knots[0]) < 1e-12 and abs(knots[-1]) < 1e-12}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": {key: bool(value) for key, value in checks.items()},
        "terminal_position_error_m": position, "terminal_heading_error_deg": math.degrees(heading),
        "native_collision_detected": bool(collided), "independent_original_polygon_SAT_run": not collided,
        "independent_colliding_samples": independent_collisions if not collided else None,
        "sample_count": len(poses), "total_path_length_m": float(arc[-1]), "total_reverse_m": reverse,
        "leg_lengths_m": lengths.tolist(), "cusp_curvatures_1_m": knots[model.cusps].tolist(),
        "cusp_nominal_front_angles_deg": np.degrees(np.arctan(.645 * knots[model.cusps])).tolist(),
        "curvature_knots_1_m": knots.tolist(), "leg_control_counts": model.leg_counts.tolist(),
        "leg_directions": model.leg_directions.tolist(), "poses_local_xyyaw": poses.tolist(),
        "travelled_arc_m": arc.tolist(), "sample_curvature_1_m": curvature.tolist(), "sample_control_indices": controls.tolist(),
        "nominal_curvature_continuous_at_cusps_by_shared_parameter": True,
        "dynamic_feasibility_certified": False, "actual_nonzero_steering_reversal_verified": False, "PhysX_executed": False}


def optimize(model, manifest, occupied, origin, resolution, budget_s=90., extended_reverse=False, heartbeat=None):
    if not math.isfinite(budget_s) or not 0 < budget_s <= 120.:
        raise ValueError("Bounded development optimization <=120 s required")
    started = time.perf_counter()
    objective = Objective(model, manifest, occupied, origin, resolution, started + budget_s, heartbeat)
    initial = model.initial(extended_reverse)
    status, optimizer = "OPTIMIZATION_COMPLETED_NOT_A_FEASIBILITY_CERTIFICATE", None
    try:
        optimizer = least_squares(lambda z: objective.evaluate(z)[0], initial, jac=lambda z: objective.evaluate(z)[1],
            bounds=model.bounds(), method="trf", tr_solver="lsmr", x_scale="jac", max_nfev=160,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
    except TimeoutError:
        status = "OPTIMIZATION_TIME_BUDGET_EXHAUSTED_NOT_GLOBAL_INFEASIBILITY"
    chosen = objective.best[1] if objective.best is not None else initial
    result = audit(model, chosen, manifest, occupied, origin, resolution)
    return {"status": "CONTINUOUS_CURVATURE_SPATIAL_CANDIDATE_AUDITED_STOP_TIME_AND_DYNAMICS_PENDING" if result["verdict"] == "PASS" else status,
        "geometry_audit": result, "parameters": chosen.tolist(), "straight_cusps_required": model.straight_cusps,
        "extended_reverse_initialization": extended_reverse, "objective_evaluations": objective.calls,
        "optimizer_message": str(optimizer.message) if optimizer is not None else None,
        "optimizer_cost_is_NOT_collision_or_feasibility_certificate": True,
        "best_guidance_squared_residual": objective.best[0] if objective.best is not None else None,
        "elapsed_s": time.perf_counter() - started, "budget_s": budget_s,
        "global_infeasibility_proven": False, "PhysX_executed": False}

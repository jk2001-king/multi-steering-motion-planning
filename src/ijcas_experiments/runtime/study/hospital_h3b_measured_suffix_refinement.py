"""Offline suffix search revision: finer controls and denser obstacle guidance.

No robot, model, goal, map, acceptance or physical limit is changed. The old
failed search is kept. Bounds now conservatively enforce the EXISTING actual
center-arc reverse episode limit, rather than only longitudinal parameter.
Guidance is not a certificate: independent dense polygon SAT remains required.
"""

import math

import numpy as np

from hospital_h3b_measured_suffix import MeasuredSuffixModel, from_saved_candidate
from hospital_h3b_warm_start import Objective
from plan_hospital_h3b_clearance_reserve_seed import NOMINAL_CLEARANCE, NOMINAL_CURVATURE_CAP, reserve_bounds


VERSION = 'H3B_measured_suffix_dense_guidance_v2'


def refined_model(candidate, measured_start, subdivisions=2):
    """Subdivide the remaining spatial controls, never move the physical start."""
    if subdivisions not in (2, 3):
        raise ValueError('Predeclared two- or three-way spatial subdivision required')
    old = from_saved_candidate(candidate, measured_start)
    knots, lengths = old.unpack(old.initial())
    new_knots = np.r_[np.concatenate([
        np.linspace(a, b, subdivisions, endpoint=False) for a, b in zip(knots[:-1], knots[1:])]), knots[-1]]
    actions = []
    for i, direction in enumerate(old.direction):
        for j in range(subdivisions):
            k = i * subdivisions + j
            actions.append(dict(mode='coordinated_counterphase', direction=int(direction),
                curvature_1_m=float((new_knots[k] + new_knots[k + 1]) / 2)))
    model = MeasuredSuffixModel(actions, old.family, measured_start)
    # Parameterization resolution must not silently enlarge the old length box.
    model.reference_lengths = old.reference_lengths.copy()
    model.initial_parameters = np.r_[new_knots[model.free], lengths]
    return model


def center_arc_safe_bounds(model):
    lower, upper = reserve_bounds(model)
    # sqrt(1+(arm*k)^2) bounds center distance / longitudinal distance pointwise.
    # This is conservative and strengthens, never relaxes, the original 0.5 m.
    for i, (direction, mode) in enumerate(zip(model.leg_directions, model.leg_modes)):
        if direction < 0:
            arm = .645 if mode == 'front_only' else 0.
            upper[len(model.free) + i] = .5 / math.sqrt(1 + (arm * NOMINAL_CURVATURE_CAP) ** 2)
    return lower, upper


class DenseSuffixObjective(Objective):
    """Analytic guidance with dense body points and stronger original bounds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a, b = .815 + NOMINAL_CLEARANCE, .4825 + NOMINAL_CLEARANCE
        self.points = np.array([(x, y) for x in np.linspace(-a, a, 29) for y in np.linspace(-b, b, 19)])

    def evaluate(self, z):
        # Parent caches and scores its own coarse guidance. Use it for field,
        # geometry and analytic Jacobians, but retain our independently scored
        # best result after adding exact original search-bound residuals.
        parent_best = self.best
        residual, derivative = super().evaluate(z)
        self.best = parent_best
        poses, jac, *_ = self.model.integrate(z, jacobian=True)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        half_y = .815 * abs(s) + .4825 * abs(c)
        dh = .815 * np.sign(s) * c - .4825 * np.sign(c) * s
        limits = (poses[:, 0] - (self.goal[0] + .4), -.3 - poses[:, 0],
            poses[:, 1] + half_y - 1.99, -1.99 - poses[:, 1] + half_y)
        limit_jac = (jac[:, 0], -jac[:, 0], jac[:, 1] + dh[:, None] * jac[:, 2],
            -jac[:, 1] + dh[:, None] * jac[:, 2])
        # Existing 100-weight bounds remain; add stronger guidance, not a
        # different legal region. Endpoint weight restores balance after denser
        # footprint samples; final original endpoint audit is unchanged.
        extra = [(poses[-1] - self.goal) * np.array([900., 900., 450.])]
        extra_jac = [jac[-1] * np.array([900., 900., 450.])[:, None]]
        for value, dj in zip(limits, limit_jac):
            extra.append(1500. * np.maximum(value, 0.))
            extra_jac.append(1500. * dj * (value > 0.)[:, None])
        residual = np.concatenate([residual, *extra])
        derivative = np.concatenate([derivative, *extra_jac])
        score = float(residual @ residual)
        if self.best is None or score < self.best[0]:
            self.best = (score, np.asarray(z).copy())
        return residual, derivative

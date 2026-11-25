"""Mass-normalized planar rigid-body feasibility for lattice edges.

The model deliberately avoids claiming an accurately measured total mass.  It
uses Newton--Euler equations divided by mass and checks a conservative
aggregate wrench envelope over bounded yaw radius, CoM height, and friction.
It is an optional planning constraint; the legacy planner is unchanged when no
``MassNormalizedRigidBodyEdgeModel`` is supplied.
"""

import math
from dataclasses import dataclass


GRAVITY_M_S2 = 9.80665


def _toward(current, target, maximum_change):
    return current + max(-maximum_change, min(maximum_change, target - current))


def _clamp(value, maximum_absolute):
    return max(-maximum_absolute, min(maximum_absolute, value))


@dataclass(frozen=True)
class RigidBodyParameterBounds:
    """Explicit, auditable uncertainty and feasibility limits.

    Values are intentionally mandatory: physical-platform limits must not be
    silently invented by the planner.
    """

    yaw_radius_min_m: float
    yaw_radius_max_m: float
    com_height_min_m: float
    com_height_max_m: float
    com_longitudinal_min_m: float
    com_longitudinal_max_m: float
    friction_min: float
    track_width_m: float
    support_radius_m: float
    front_support_x_m: float
    rear_support_x_m: float
    longitudinal_response_time_s: float
    lateral_response_time_s: float
    yaw_response_time_s: float
    max_longitudinal_accel_m_s2: float
    max_lateral_accel_m_s2: float
    max_yaw_rate_rad_s: float
    max_yaw_accel_rad_s2: float
    max_friction_utilization: float
    max_abs_ltr: float
    com_lateral_min_m: float = 0.0
    com_lateral_max_m: float = 0.0
    caster_specific_force_x_abs_max_m_s2: float = 0.0
    caster_specific_force_y_abs_max_m_s2: float = 0.0
    caster_specific_yaw_moment_abs_max_m2_s2: float = 0.0
    friction_cost_weight_s: float = 0.0
    ltr_cost_weight_s: float = 0.0
    actuator_cost_weight_s: float = 0.0

    def __post_init__(self):
        positive = {
            "yaw_radius_min_m": self.yaw_radius_min_m,
            "yaw_radius_max_m": self.yaw_radius_max_m,
            "com_height_min_m": self.com_height_min_m,
            "com_height_max_m": self.com_height_max_m,
            "friction_min": self.friction_min,
            "track_width_m": self.track_width_m,
            "support_radius_m": self.support_radius_m,
            "front_support_x_m": self.front_support_x_m,
            "longitudinal_response_time_s": self.longitudinal_response_time_s,
            "lateral_response_time_s": self.lateral_response_time_s,
            "yaw_response_time_s": self.yaw_response_time_s,
            "max_longitudinal_accel_m_s2": self.max_longitudinal_accel_m_s2,
            "max_lateral_accel_m_s2": self.max_lateral_accel_m_s2,
            "max_yaw_rate_rad_s": self.max_yaw_rate_rad_s,
            "max_yaw_accel_rad_s2": self.max_yaw_accel_rad_s2,
            "max_friction_utilization": self.max_friction_utilization,
            "max_abs_ltr": self.max_abs_ltr,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("rigid-body parameters must be positive: " + ", ".join(invalid))
        if self.yaw_radius_min_m > self.yaw_radius_max_m:
            raise ValueError("yaw_radius_min_m must not exceed yaw_radius_max_m")
        if self.com_height_min_m > self.com_height_max_m:
            raise ValueError("com_height_min_m must not exceed com_height_max_m")
        if self.com_longitudinal_min_m > self.com_longitudinal_max_m:
            raise ValueError(
                "com_longitudinal_min_m must not exceed com_longitudinal_max_m"
            )
        if self.com_lateral_min_m > self.com_lateral_max_m:
            raise ValueError(
                "com_lateral_min_m must not exceed com_lateral_max_m"
            )
        if not (
            -self.track_width_m / 2.0 < self.com_lateral_min_m
            and self.com_lateral_max_m < self.track_width_m / 2.0
        ):
            raise ValueError("lateral CoM bounds must lie inside the support width")
        disturbances = {
            "caster_specific_force_x_abs_max_m_s2": (
                self.caster_specific_force_x_abs_max_m_s2
            ),
            "caster_specific_force_y_abs_max_m_s2": (
                self.caster_specific_force_y_abs_max_m_s2
            ),
            "caster_specific_yaw_moment_abs_max_m2_s2": (
                self.caster_specific_yaw_moment_abs_max_m2_s2
            ),
        }
        invalid_disturbances = [
            name for name, value in disturbances.items() if value < 0.0
        ]
        if invalid_disturbances:
            raise ValueError(
                "caster disturbance bounds must be nonnegative: "
                + ", ".join(invalid_disturbances)
            )
        if self.rear_support_x_m >= self.front_support_x_m:
            raise ValueError("rear_support_x_m must be behind front_support_x_m")
        if not (
            self.rear_support_x_m < self.com_longitudinal_min_m
            and self.com_longitudinal_max_m < self.front_support_x_m
        ):
            raise ValueError("longitudinal CoM bounds must lie inside the support base")
        if any(
            value < 0.0
            for value in (
                self.friction_cost_weight_s,
                self.ltr_cost_weight_s,
                self.actuator_cost_weight_s,
            )
        ):
            raise ValueError("rigid-body cost weights must be nonnegative")


@dataclass(frozen=True)
class RigidBodyStep:
    longitudinal_velocity_m_s: float
    lateral_velocity_m_s: float
    yaw_rate_rad_s: float
    longitudinal_accel_m_s2: float
    lateral_accel_m_s2: float
    yaw_accel_rad_s2: float
    friction_utilization: float
    front_friction_utilization: float
    rear_friction_utilization: float
    minimum_normal_load_fraction: float
    abs_ltr: float
    actuator_utilization: float
    feasible: bool
    penalty_s: float


class MassNormalizedRigidBodyEdgeModel:
    """Propagate one time step and evaluate worst-case normalized feasibility."""

    def __init__(self, bounds):
        if not isinstance(bounds, RigidBodyParameterBounds):
            raise TypeError("bounds must be RigidBodyParameterBounds")
        self.bounds = bounds
        self._support_length = bounds.front_support_x_m - bounds.rear_support_x_m
        self._support_arms = tuple(
            (
                bounds.front_support_x_m - com_x,
                com_x - bounds.rear_support_x_m,
            )
            for com_x in (
                bounds.com_longitudinal_min_m,
                bounds.com_longitudinal_max_m,
            )
        )
        self._com_heights = (
            bounds.com_height_min_m,
            bounds.com_height_max_m,
        )
        self._yaw_radius_squared = (
            bounds.yaw_radius_min_m**2,
            bounds.yaw_radius_max_m**2,
        )

    def step(
        self,
        longitudinal_velocity_m_s,
        lateral_velocity_m_s,
        yaw_rate_rad_s,
        target_longitudinal_velocity_m_s,
        target_lateral_velocity_m_s,
        target_yaw_rate_rad_s,
        dt_s,
    ):
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        b = self.bounds
        # The controller saturates acceleration instead of rejecting the
        # command.  Propagate that saturated response so the lattice endpoint
        # reflects the velocity actually reachable within one primitive.
        longitudinal_accel = _clamp(
            (target_longitudinal_velocity_m_s - longitudinal_velocity_m_s)
            / b.longitudinal_response_time_s,
            b.max_longitudinal_accel_m_s2,
        )
        lateral_velocity_rate = _clamp(
            (target_lateral_velocity_m_s - lateral_velocity_m_s)
            / b.lateral_response_time_s,
            b.max_lateral_accel_m_s2,
        )
        yaw_accel = _clamp(
            (target_yaw_rate_rad_s - yaw_rate_rad_s) / b.yaw_response_time_s,
            b.max_yaw_accel_rad_s2,
        )

        # Newton--Euler specific wrench (force and moment divided by mass).
        specific_force_x = longitudinal_accel - yaw_rate_rad_s * lateral_velocity_m_s
        specific_force_y = lateral_velocity_rate + yaw_rate_rad_s * longitudinal_velocity_m_s
        # Conservative two-support-line Newton--Euler allocation.  All terms
        # are divided by total mass, so the model remains useful when the
        # physical total mass is unavailable.  The worst corner of the bounded
        # yaw radius, longitudinal CoM and height is used for each edge.
        support_length = self._support_length
        maximum_front_utilization = 0.0
        maximum_rear_utilization = 0.0
        minimum_normal_load_fraction = math.inf
        maximum_abs_ltr = 0.0
        support_feasible = True
        # The caster disturbance set is an axis-aligned, sign-symmetric box.
        # The maximum norm of each axle wrench is therefore attained at a box
        # corner and can be evaluated in closed form.  This is exactly
        # equivalent to enumerating all 2^3 disturbance signs, while avoiding
        # that enumeration at every integration sub-step of every lattice
        # edge.
        caster_force_x_bound = b.caster_specific_force_x_abs_max_m_s2
        caster_force_y_bound = b.caster_specific_force_y_abs_max_m_s2
        caster_yaw_bound = b.caster_specific_yaw_moment_abs_max_m2_s2
        yaw_moment_candidates = (
            self._yaw_radius_squared[0] * yaw_accel,
            self._yaw_radius_squared[1] * yaw_accel,
        )
        for front_arm, rear_arm in self._support_arms:
            for com_height in self._com_heights:
                front_normal = (
                    GRAVITY_M_S2 * rear_arm
                    - com_height * specific_force_x
                ) / support_length
                rear_normal = GRAVITY_M_S2 - front_normal
                if front_normal <= 0.0 or rear_normal <= 0.0:
                    support_feasible = False
                    continue
                lateral_load_shift = (
                    com_height * specific_force_y / GRAVITY_M_S2
                )
                worst_abs_ltr = 2.0 * max(
                    abs(b.com_lateral_min_m + lateral_load_shift),
                    abs(b.com_lateral_max_m + lateral_load_shift),
                ) / b.track_width_m
                maximum_abs_ltr = max(maximum_abs_ltr, worst_abs_ltr)
                minimum_side_fraction = 0.5 * (1.0 - worst_abs_ltr)
                minimum_normal_load_fraction = min(
                    minimum_normal_load_fraction,
                    front_normal / GRAVITY_M_S2 * minimum_side_fraction,
                    rear_normal / GRAVITY_M_S2 * minimum_side_fraction,
                )
                if minimum_side_fraction <= 0.0:
                    support_feasible = False
                    continue
                maximum_required_force_x = (
                    abs(specific_force_x) + caster_force_x_bound
                )
                front_force_x_abs = (
                    maximum_required_force_x
                    * front_normal / GRAVITY_M_S2
                )
                rear_force_x_abs = (
                    maximum_required_force_x
                    * rear_normal / GRAVITY_M_S2
                )
                front_force_y_abs = (
                    max(
                        abs(yaw_moment_candidates[0] + rear_arm * specific_force_y),
                        abs(yaw_moment_candidates[1] + rear_arm * specific_force_y),
                    )
                    + caster_yaw_bound
                    + rear_arm * caster_force_y_bound
                ) / support_length
                rear_force_y_abs = (
                    max(
                        abs(front_arm * specific_force_y - yaw_moment_candidates[0]),
                        abs(front_arm * specific_force_y - yaw_moment_candidates[1]),
                    )
                    + caster_yaw_bound
                    + front_arm * caster_force_y_bound
                ) / support_length
                maximum_front_utilization = max(
                    maximum_front_utilization,
                    math.hypot(front_force_x_abs, front_force_y_abs)
                    / (b.friction_min * front_normal),
                )
                maximum_rear_utilization = max(
                    maximum_rear_utilization,
                    math.hypot(rear_force_x_abs, rear_force_y_abs)
                    / (b.friction_min * rear_normal),
                )
        friction_utilization = max(
            maximum_front_utilization,
            maximum_rear_utilization,
        )
        abs_ltr = maximum_abs_ltr
        actuator_utilization = max(
            abs(longitudinal_accel) / b.max_longitudinal_accel_m_s2,
            abs(lateral_velocity_rate) / b.max_lateral_accel_m_s2,
            abs(yaw_accel) / b.max_yaw_accel_rad_s2,
        )

        feasible = (
            abs(target_yaw_rate_rad_s) <= b.max_yaw_rate_rad_s
            and support_feasible
            and friction_utilization <= b.max_friction_utilization
            and abs_ltr <= b.max_abs_ltr
        )
        penalty = (
            b.friction_cost_weight_s * friction_utilization**2
            + b.ltr_cost_weight_s * abs_ltr**2
            + b.actuator_cost_weight_s * actuator_utilization**2
        ) * dt_s

        next_u = _toward(
            longitudinal_velocity_m_s,
            target_longitudinal_velocity_m_s,
            abs(longitudinal_accel) * dt_s,
        )
        next_v = _toward(
            lateral_velocity_m_s,
            target_lateral_velocity_m_s,
            abs(lateral_velocity_rate) * dt_s,
        )
        next_r = _toward(
            yaw_rate_rad_s,
            target_yaw_rate_rad_s,
            abs(yaw_accel) * dt_s,
        )
        return RigidBodyStep(
            next_u,
            next_v,
            next_r,
            specific_force_x,
            specific_force_y,
            yaw_accel,
            friction_utilization,
            maximum_front_utilization,
            maximum_rear_utilization,
            minimum_normal_load_fraction,
            abs_ltr,
            actuator_utilization,
            feasible,
            penalty,
        )

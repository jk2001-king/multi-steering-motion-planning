"""Development-only standstill motor gains; never lock passive caster joints."""

import math

import numpy as np


def drive_hold_gain_arrays(original_kp, original_kd, drive_indices, hold_kp=10000.):
    """Use runtime SI/radian gains, NOT USD angular per-degree values.

    Targets must be captured from measured wheel positions before applying
    these gains. Existing damping and torque limits are not increased.
    The gain is a provisional simulation test, not a hardware identification.
    """
    kp, kd = np.asarray(original_kp, dtype=float), np.asarray(original_kd, dtype=float)
    indices = tuple(drive_indices)
    if kp.ndim != 1 or kd.shape != kp.shape or not len(kp):
        raise ValueError("Expected equally sized gain vectors")
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("Gains must be finite and nonnegative")
    if (len(indices) != 2 or len(set(indices)) != 2 or any(
            not isinstance(index, (int, np.integer)) or index < 0 or index >= len(kp) for index in indices)):
        raise ValueError("Exactly two distinct drive joint indices are required")
    if not math.isfinite(hold_kp) or not 0 < hold_kp <= 10000.:
        raise ValueError("Development position gain must be in (0, 10000] Nm/rad")
    held_kp, held_kd = kp.copy(), kd.copy()
    held_kp[list(indices)] = hold_kp
    return held_kp, held_kd


def zero_command_phases(control):
    if control == "zero_velocity":
        return ({"name": "zero_velocity", "settle_s": .5, "sample_s": 3., "position_hold": False},)
    if control == "drive_position_hold_aba":
        return (
            {"name": "baseline_before", "settle_s": .5, "sample_s": 2., "position_hold": False},
            {"name": "drive_position_hold", "settle_s": 1., "sample_s": 3., "position_hold": True},
            {"name": "baseline_after", "settle_s": .5, "sample_s": 2., "position_hold": False},
        )
    raise ValueError("Unsupported zero-command control mode")


class StraightStopProfile:
    """Ordinary forward acceleration/braking; no yaw input or CICS policy.

    Motor position hold may be captured only when the speed command reaches
    zero and measured body speed is <=10 mm/s. This is a low-speed brake
    engagement threshold, not the final 5 mm/s acceptance criterion.
    """

    def __init__(self, distance_m=.30, speed_m_s=.05, acceleration_m_s2=.02):
        values = (distance_m, speed_m_s, acceleration_m_s2)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Straight-stop parameters must be finite and positive")
        if not .10 <= distance_m <= .30 or speed_m_s > .05 or acceleration_m_s2 > .02:
            raise ValueError("Development stop test is bounded to 0.30 m, 0.05 m/s, 0.02 m/s^2")
        self.distance_m, self.speed_m_s, self.acceleration_m_s2 = values
        self.previous_time_s = 0.
        self.command_speed_m_s = 0.
        self.braking = False

    def step(self, elapsed_s, progress_m, measured_speed_m_s):
        if not all(math.isfinite(value) for value in (elapsed_s, progress_m, measured_speed_m_s)):
            raise ValueError("Non-finite straight-stop state")
        if elapsed_s < self.previous_time_s:
            raise ValueError("Straight-stop time must not go backwards")
        dt = elapsed_s-self.previous_time_s
        self.previous_time_s = elapsed_s
        remaining = self.distance_m-progress_m
        if remaining <= .015:
            self.braking = True
        target = 0. if self.braking else min(self.speed_m_s,
            math.sqrt(2*self.acceleration_m_s2*max(0., remaining-.010)))
        maximum_change = self.acceleration_m_s2*dt
        self.command_speed_m_s += max(-maximum_change, min(maximum_change, target-self.command_speed_m_s))
        hold_ready = self.braking and self.command_speed_m_s <= 1e-7 and abs(measured_speed_m_s) <= .010
        return self.command_speed_m_s, hold_ready

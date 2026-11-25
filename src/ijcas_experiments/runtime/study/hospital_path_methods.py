"""Explicit H1 plan selection and execution naming; no controller tuning."""

METHODS = {
    "B0_fixed_counterphase_ramp": ("hospital_h1_b0_fixed_path", "kinematic"),
    "B1_multimode_kinematic_ramp": ("hospital_h1_b1_kinematic_path", "kinematic"),
    "B2_multimode_transition_cost_ramp": ("hospital_h1_b2_transition_path", "transition_cost"),
    "P_dynamics_full_envelope": ("hospital_h1_proposed_path", "full"),
}

COMMON_HELPER_FILES = (
    "hospital_path_execution.py", "hospital_drive_hold.py",
    "hospital_contact_rules.py", "hospital_geometry_tools.py",
)

PHYSICS_COMPARISON_KEYS = (
    "geometry_manifest_sha256", "condition", "simulation_mass_kg", "physics_hz", "gravity_m_s2",
    "static_friction", "dynamic_friction", "contact_stiffness_n_m", "contact_damping_n_s_m",
    "support_floor", "support_floor_top_z_m", "bound_wheel_bodies", "rigid_bodies",
    "disabled_legacy_floor_colliders",
)


def method_settings(method):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("Select an exact H1 method: "+", ".join(METHODS))
    prefix, dynamics = METHODS[method]
    return {"method": method, "run_prefix": prefix, "planner_dynamics": dynamics}


def comparison_mismatches(reference, current):
    """Name mismatches, never silently pool non-matched development trials.

    The adapter source changes to add method selection. Hash the unchanged
    common controller/contact helpers, and compare limits/physics separately.
    No physical seed pairing or equal runtime caster state is implied.
    """
    differences = []
    old_version = reference.get("execution_version", "legacy_v1")
    new_version = current.get("execution_version", "legacy_v1")
    if old_version != new_version:
        differences.append("execution_version")
    for key in ("geometry_manifest_sha256", "start_world_xyyaw", "goal_world_xyyaw", "command_limits",
            "acceptance", "terminal_capture_inner_band", "common_controller", "feedback_source"):
        if key not in reference or key not in current or reference[key] != current[key]:
            differences.append(key)
    for name in COMMON_HELPER_FILES:
        old = reference.get("source_sha256", {}).get(name)
        new = current.get("source_sha256", {}).get(name)
        if not old or old != new:
            differences.append("source_sha256."+name)
    staged_versions = {"stop_align_v2", "stop_align_v3"}
    if old_version in staged_versions or new_version in staged_versions:
        for key in ("execution_cohort", "execution_role", "terminal_controller_config"):
            if key not in reference or key not in current or reference[key] != current[key]:
                differences.append(key)
        name = "hospital_path_execution_v2.py"
        old = reference.get("source_sha256", {}).get(name)
        if not old or old != current.get("source_sha256", {}).get(name):
            differences.append("source_sha256."+name)
    if old_version == "stop_align_v3" or new_version == "stop_align_v3":
        for name in ("hospital_path_execution_v3.py", "isaac_run_ijcas_hospital_h1_path_gate.py"):
            old = reference.get("source_sha256", {}).get(name)
            new = current.get("source_sha256", {}).get(name)
            if not old or old != new:
                differences.append("source_sha256."+name)
    old_physics, new_physics = reference.get("physics_preparation_snapshot", {}), current.get("physics_preparation_snapshot", {})
    for key in PHYSICS_COMPARISON_KEYS:
        if key not in old_physics or key not in new_physics or old_physics[key] != new_physics[key]:
            differences.append("physics."+key)
    return differences

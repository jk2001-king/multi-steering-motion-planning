"""H4 schema/frame binding to the SAME H1/H2 v3 control implementations.

Only plan validation and construction differ. No new gain, course feedback,
terminal timing law, body model, caster state or acceptance band is invented.
This is an offline execution binding, not a live H4 path admission.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import math
from pathlib import Path

import hospital_h2_path_execution as common


ROOT = _ijcas_workspace()
SCENARIO = 'H4_ward_corridor_turn'
EXECUTION_VERSION = 'stop_align_v3_H4_schema_binding_v1'
CONTROLLER_CONFIG = dict(common.CONTROLLER_CONFIG)
GUARDS, ACCEPTANCE, COMMAND_LIMITS = dict(common.GUARDS), dict(common.ACCEPTANCE), dict(common.COMMAND_LIMITS)
direct_joint_targets, relative_pose, wrap = common.direct_joint_targets, common.relative_pose, common.wrap
motor_settings_record = common.motor_settings_record
METHODS = tuple(common._methods['METHODS'])
MODES = ('coordinated_counterphase', 'front_only', 'rear_only')
FOOTPRINT = dict(length_m=1.630, width_m=.965, clearance_m=.010)
COMMON_SOURCE_FILES = ('hospital_h4_path_execution.py', 'hospital_h2_path_execution.py',
    'hospital_path_execution.py', 'hospital_path_execution_v2.py', 'hospital_path_execution_v3.py',
    'hospital_drive_hold.py', 'hospital_stop_analysis.py', 'hospital_geometry_tools.py', 'hospital_path_methods.py')


def method_settings(method):
    result = common._methods['method_settings'](method)
    return {**result, 'run_prefix': 'hospital_h4_shared_v3_'+method}


def validate_plan(plan):
    actions, states = plan.get('selected_actions', []), plan.get('selected_states', [])
    if (plan.get('scenario') != SCENARIO or plan.get('plan_success') is not True
            or not actions or len(actions) != len(states)):
        raise ValueError('Need a complete successful H4 native MODEL plan, not a relabelled H1/H2/H3 result')
    settings = method_settings(plan['method'])
    allowed = ['coordinated_counterphase'] if plan['method'] == METHODS[0] else list(MODES)
    if plan['planner_dynamics'] != settings['planner_dynamics'] or plan['allowed_modes'] != allowed:
        raise ValueError('H4 method/dynamics/planned-mode constraint mismatch')
    for key in ('start_world_xyyaw', 'goal_world_xyyaw', 'goal_local_xyyaw'):
        if len(plan[key]) != 3 or not all(math.isfinite(v) for v in plan[key]):
            raise ValueError('Finite H4 world/local task poses required')
    goal = relative_pose(plan['goal_world_xyyaw'], plan['start_world_xyyaw'])
    if any(abs(a-b) > 1e-10 for a, b in zip(goal, plan['goal_local_xyyaw'])):
        raise ValueError('H4 local/world task frame mismatch')
    if (plan['common_max_command_speed_m_s'] != .05 or plan['common_max_acceleration_m_s2'] != .02
            or plan['full_envelope_clearance_m'] != .010 or plan['footprint'] != FOOTPRINT):
        raise ValueError('H4 common control/footprint/clearance limits must remain equal to H1/H2')
    for action, state in zip(actions, states):
        if (action['mode'] not in allowed or isinstance(action['direction'], bool) or action['direction'] != 1
                or not math.isfinite(action['curvature_1_m']) or abs(action['curvature_1_m']) > .70+1e-9
                or not math.isfinite(action['target_speed_m_s']) or not 0 < action['target_speed_m_s'] <= .05):
            raise ValueError('Only bounded forward H4 planned actions are supported')
        pose = state['world_xyyaw']
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError('Finite saved H4 world pose required')
        local = relative_pose(pose, plan['start_world_xyyaw'])
        expected = (state['x_m'], state['y_m'], math.radians(state['yaw_deg']))
        if any(abs(a-b) > 1e-9 for a, b in zip(local, expected)):
            raise ValueError('Saved native H4 state local/world coordinates disagree')
    if (plan.get('actual_motion_authorized') is not False or plan.get('use_for_final_statistics') is not False
            or plan.get('native_MODEL_success_is_NOT_actual_support_or_drive_admission') is not True):
        raise ValueError('Do not promote a native model plan into actual motion or paper evidence')


class PathReference(common.PathReference):
    """Original projection methods; only task-specific construction differs."""

    def __init__(self, plan):
        validate_plan(plan)
        self.actions = plan['selected_actions']
        self.poses = [tuple(plan['start_world_xyyaw'])]+[tuple(s['world_xyyaw']) for s in plan['selected_states']]
        self.arc = [0.]
        for first, last in zip(self.poses, self.poses[1:]):
            distance = math.hypot(last[0]-first[0], last[1]-first[1])
            if not .001 <= distance <= .10:
                raise ValueError('Unexpected zero/oversized H4 reference edge')
            self.arc.append(self.arc[-1]+distance)
        self.progress = 0.


class H4TrackingController(common.H2TrackingController):
    """Task construction ONLY; every tracking/terminal method stays frozen v3."""

    def __init__(self, plan):
        self.reference = PathReference(plan)
        self.goal = tuple(plan['goal_world_xyyaw'])
        self.phase, self.speed, self.previous_time = 'path_tracking', 0., None
        self.mode = plan['selected_actions'][0]['mode']
        self.curvature = plan['selected_actions'][0]['curvature_1_m']
        self.direction, self.terminal_mode, self.terminal_switches = 0, '', 0
        self.handoff = None
        self.config = dict(CONTROLLER_CONFIG)
        self.terminal_stage = 'tracking'
        self.stage_started = self.terminal_started = self.aligned_since = None
        self.pulse_count, self.pulse_origin, self.pulse_distance = 0, None, 0.
        self.drive_hold_requested, self.preferred_direction = False, 1
        self.hold_entry_candidate = None


def comparison_mismatches(reference, current):
    """Fail closed for nonmatched H4 trials; never merge development versions."""
    fields = ('scenario', 'execution_version', 'execution_role', 'execution_cohort', 'geometry_manifest_sha256',
        'start_world_xyyaw', 'goal_world_xyyaw', 'planning_occupancy_digest', 'command_limits', 'guards',
        'acceptance', 'terminal_controller_config', 'feedback_source', 'shared_candidate_pool_sha256',
        'generator_selector_version', 'source_authored_caster_seed_policy', 'physical_condition',
        'physics_baseline_sha256')
    differences = [key for key in fields if key not in reference or key not in current or reference[key] != current[key]]
    for name in COMMON_SOURCE_FILES:
        a, b = reference.get('source_sha256', {}).get(name), current.get('source_sha256', {}).get(name)
        if not a or a != b:
            differences.append('source_sha256.'+name)
    return differences

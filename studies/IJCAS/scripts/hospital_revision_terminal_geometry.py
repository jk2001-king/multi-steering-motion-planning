"""Tolerance-aware straight terminal pulses; never an arrival criterion.

The original chooser commands full-curvature F/R steering for lateral error
over2.5mm, even when a straight ray intersects its unchanged15mm capture disk.
Use a zero-curvature counterphase pulse only inside HALF that capture radius
and the original moving-capture heading tolerance. A retained, bounded
physical probe found almost no movement at10mm/s but movement at20mm/s.
Use20mm/s for terminal pulses of EVERY method, retaining the original ramp,
measured braking, ACKs, pulse resources, time and actual success guards.
Neither the probe nor the nominal model establishes physical convergence.
"""
import copy
import csv
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import hospital_revision_endpoint_handoff as parent

runtime = parent.runtime
VERSION = 'COMMON1330_ENDPOINT_v5_STRAIGHT_TERMINAL20_COMMON_v7'
OUTPUT = parent.OUTPUT.parent/'tolerance_aware_terminal_paired_v7'
PROTOCOL = OUTPUT/'protocol.json'
hold_implementation = parent.hold_implementation


def straight_capture_ray_allowed(controller, pose):
    # Body-frame longitudinal/lateral/heading errors, same existing helper.
    error = parent.parent.base.goal.parent.sequence.lifecycle.relative_pose(controller.goal, pose)
    radius = float(controller.config['capture_position_m'])
    heading = math.radians(float(controller.config['capture_heading_deg']))
    return (abs(error[1]) <= .5*radius and abs(error[2]) <= heading)


def bind_terminal(controller):
    if getattr(controller, 'tolerance_aware_straight_version', None):
        raise ValueError('Terminal chooser must be bound exactly once')
    if controller.config['terminal_speed_limit_m_s'] != .010:
        raise ValueError('Original10mm/s terminal policy required before single binding')
    controller.config['terminal_speed_limit_m_s'] = .020
    old = controller._choose_pulse
    def choose(self, pose):
        before_mode, before_switches = self.terminal_mode, self.terminal_switches
        allowed = straight_capture_ray_allowed(self, pose)
        old(pose)
        if allowed:
            self.terminal_mode = self.mode = 'coordinated_counterphase'
            self.curvature = 0.
            self.terminal_switches = before_switches + bool(before_mode and before_mode != self.mode)
    controller._choose_pulse = MethodType(choose, controller)
    controller.tolerance_aware_straight_version = VERSION
    return controller


def bind_motion(motion):
    if getattr(motion, 'tolerance_aware_straight_version', None):
        raise ValueError('One terminal factory binding required')
    old = motion.raw
    motion.raw = lambda *a, **k: bind_terminal(old(*a, **k))
    bind_terminal(motion.tracker)
    motion.tolerance_aware_straight_version = VERSION
    return motion


def contract(profile, *, simulation_budget_s, wall_budget_s=900.):
    value = copy.deepcopy(parent.contract(profile, simulation_budget_s=simulation_budget_s,
        wall_budget_s=wall_budget_s))
    value.update(version=VERSION, parent_controller_version=parent.VERSION,
        changed_terminal_geometry_only=dict(straight_lateral_bound='half original15mm moving-capture radius',
            straight_heading_bound='original1.5deg moving-capture tolerance',
            old_nonstraight_chooser_retained=True, same_direction_and_pulse_distance=True),
        terminal_speed_m_s=.020, original_terminal_speed_m_s=.010,
        original20mm_s2_ramp_and_speed_aware_braking_unchanged=True,
        original15mm_moving_capture_and30mm_fixed_hold_NOT_relaxed=True,
        original_gains_torque_limits50mm_s_cap_counts_and_time_caps_unchanged=True,
        common_all_methods_all_tasks=True)
    return value


def controller(plan, geometry, *, profile, simulation_budget_s, start_time_s=0., wall_budget_s=900.):
    value = parent.controller(plan, geometry, profile=profile, simulation_budget_s=simulation_budget_s,
        start_time_s=start_time_s, wall_budget_s=wall_budget_s)
    bind_motion(value.motion)
    value.contract = contract(profile, simulation_budget_s=simulation_budget_s, wall_budget_s=wall_budget_s)
    return value


def private_sequence():
    value = parent.private_sequence()
    return SimpleNamespace(**dict(vars(value), unified_geometry_lifecycle_controller=controller,
        unified_geometry_lifecycle_contract=contract))


def model_binding(*args, **kwargs):
    return runtime.bind(parent.parent.base.goal.model_binding,
        private_sequence=private_sequence, VERSION=VERSION)(*args, **kwargs)


base = SimpleNamespace(**dict(vars(parent.base), model_binding=model_binding))
extra_source_paths = (*parent.extra_source_paths, Path(parent.__file__))


def candidate():
    return SimpleNamespace(**dict(vars(parent.candidate()), VERSION=VERSION,
        private_sequence=private_sequence, contract=contract, model_binding=model_binding))


def bound_cell():
    call = runtime.bind(parent.bound_cell, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, candidate=candidate)()
    prior = call.__globals__['runtime']; context = prior.context_for
    private = SimpleNamespace(**dict(vars(prior), context_for=lambda folder, actor, geometry:
        context(folder, actor+'_straight_terminal20_v7', geometry)))
    return runtime.bind(call, runtime=private)


# Reuse one common preparation/static/route validator, not another SDK loop.
def validation_binding():
    import hospital_revision_accepted_reference_validation as shared
    import hospital_revision_endpoint_validation as prior
    control = SimpleNamespace(**dict(globals(), extra_source_paths=(
        *extra_source_paths, Path(shared.__file__), Path(prior.__file__))))
    return shared, dict(control=control, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
        PLAN_FOLDERS=prior.PLAN_FOLDERS,
        PREDECESSOR_POLICY='exact bounded neutral-forward diagnostic PASS; actual Pause and original restoration',
        __file__=__file__)


def export_protocol():
    shared, scope = validation_binding()
    return runtime.bind(shared.export_protocol, **scope)()


def verify_sources():
    shared, scope = validation_binding()
    return runtime.bind(shared.verify_sources, **scope)()


def planning_namespace():
    shared, scope = validation_binding()
    return runtime.bind(shared.planning_namespace, **scope)()


def validation_model_binding(*args, **kwargs):
    shared, scope = validation_binding()
    return runtime.bind(shared.model_binding, **scope)(*args, **kwargs)


def model_screen(name):
    shared, scope = validation_binding()
    return runtime.bind(shared.model_screen, **dict(scope,
        planning_namespace=planning_namespace, export_protocol=export_protocol,
        verify_sources=verify_sources, model_binding=validation_model_binding))(name)


def validation_cell():
    shared, scope = validation_binding()
    return runtime.bind(shared.bound_cell, **dict(scope, planning_namespace=planning_namespace))()


def policy_file():
    shared, scope = validation_binding()
    return runtime.bind(shared.policy_file, **scope)()


def probe_review():
    import hospital_revision_forward_response as probe
    path = probe.OUTPUT/'probe_readback.json'
    raw = runtime.read(path)
    if (raw['verdict'] != 'PASS' or not parent.parent.base.shared.execution.restored_boundary(raw)
            or not all(raw['checks'].values())):
        raise ValueError('Completed physically guarded diagnostic and actual restoration required')
    trace = Path(raw['trace'])
    if runtime.sha(trace) != raw['trace_sha256']:
        raise ValueError('Original trace fingerprint required')
    with trace.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = []
    for level in (.010, .020):
        samples = [r for r in rows if r['phase'] == 'tracking' and float(r['requested_level_m_s']) == level]
        if not samples:
            raise ValueError('Both prescribed levels must have actual observations')
        first, last = samples[0], samples[-1]
        dx = float(last['x_m'])-float(first['x_m']); dy = float(last['y_m'])-float(first['y_m'])
        yaw = float(first['yaw_rad'])
        speeds = [math.hypot(float(r['vx_physx_m_s']), float(r['vy_physx_m_s'])) for r in samples]
        groups.append(dict(requested_level_m_s=level, samples=len(samples),
            duration_s=float(last['time_s'])-float(first['time_s']),
            net_body_forward_m=dx*math.cos(yaw)+dy*math.sin(yaw),
            net_body_lateral_m=-dx*math.sin(yaw)+dy*math.cos(yaw),
            mean_raw_CoM_speed_m_s=sum(speeds)/len(speeds), maximum_raw_CoM_speed_m_s=max(speeds)))
    value = dict(probe_report=str(path), probe_report_sha256=runtime.sha(path),
        trace=str(trace), trace_sha256=runtime.sha(trace), levels=groups,
        observed_neutral10mm_s_movement_under1mm=abs(groups[0]['net_body_forward_m']) < .001,
        observed_neutral20mm_s_forward_movement_over1mm=groups[1]['net_body_forward_m'] > .001,
        original_15s30mm_caps_and_all_environment_contact_cleanup_checks_pass=True,
        finite_local_diagnostic_NOT_global_speed_identification_or_H4_mission_success=True,
        terminal_candidate20mm_s_is_common_all_methods=True, original_failed_trials_preserved=True)
    target = OUTPUT/'forward_response_review.json'
    if target.exists():
        if runtime.read(target) != value:
            raise ValueError('Diagnostic review frozen')
    else:
        runtime.evidence.write_exclusive(target, value)
    return value


def predecessor_admitted(previous):
    reviewed = probe_review()
    return (Path(previous['last_actual_report']) == Path(reviewed['probe_report'])
        and previous['report_verdict'] == 'PASS')


async def run(previous, *, authorized=False):
    shared, scope = validation_binding()
    return await runtime.bind(shared.after_current, **dict(scope,
        predecessor_admitted=predecessor_admitted, verify_sources=verify_sources,
        policy_file=policy_file, bound_cell=validation_cell))(previous, authorized=authorized)


def dispatch():
    shared, scope = validation_binding()
    export_protocol(); verify_sources(); reviewed = probe_review()
    if not (reviewed['observed_neutral10mm_s_movement_under1mm']
            and reviewed['observed_neutral20mm_s_forward_movement_over1mm']):
        raise ValueError('Actual bounded response basis required, not guessed speed')
    for name in scope['PLAN_FOLDERS']:
        result = runtime.read(OUTPUT/'model'/name/'model_screen.json')
        if '_P_' in name and not result['both_nominal_resolutions_pass']:
            raise ValueError('Own two-resolution P model checks required')
    modules = [shared.__name__, 'hospital_revision_endpoint_validation', __name__]
    source = ("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_neutral_forward_probe_r04',None)\n"
        "assert parent is not None and parent.done() and not parent.exception(),'Exact completed diagnostic required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is parent,'Intervening motion'\n"
        "assert not hasattr(builtins,'ijcas_revision_terminal_geometry_v7'),'Already reserved'\n"
        f"folder=Path({str(Path(__file__).resolve().parent)!r})\n"
        f"for name in {modules!r}:\n"
        " p=folder/(name+'.py');m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m\n"
        " exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run(parent,authorized=True))\n"
        "builtins.ijcas_revision_terminal_geometry_v7=job\n"
        "print({'common_terminal20_H4_pair_then_H3_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'reservation.json', dict(source=source,
        maximum_routes=len(scope['PLAN_FOLDERS']), automatic_retry=False,
        one_static_per_fresh_source=True, same_candidate_all_methods=True))
    return parent.parent.base.shared.runner.remote(source)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review', 'models', 'dispatch'))
    args = parser.parse_args()
    if args.action == 'review':
        print(probe_review())
    elif args.action == 'models':
        for name in validation_binding()[1]['PLAN_FOLDERS']:
            model_screen(name)
    else:
        print(dispatch())

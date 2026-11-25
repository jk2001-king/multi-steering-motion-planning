"""Common hold-damping development candidate; source plant stays unchanged.

Keep the original10000Nm/rad hold stiffness. Only while position holding,
use a documented overdamped two-drive inertia surrogate to choose damping.
The original velocity-drive damping is restored on release and cleanup.
This is a control hypothesis, not an identified physical or safety model.
"""
import ast
import copy
import inspect
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import hospital_revision_remaining_validation as base

runtime = base.runtime
VERSION = 'COMMON1330_ACTUAL_GOAL_v2_HOLD10000_DAMPING_SURROGATE_v4'
OUTPUT = base.OUTPUT.parent / 'remaining_stop_damping_paired_v4'
PROTOCOL = OUTPUT / 'protocol.json'
EFFECTIVE_INERTIA = 265. * .074**2 / 2.
HOLD_DAMPING = 4. * math.sqrt(10000. * EFFECTIVE_INERTIA)


def hold_implementation():
    original = base.goal.parent.sequence.lifecycle.motion.RetainedTensorDriveAdapter.request_hold
    source = textwrap.dedent(inspect.getsource(original)); tree = ast.parse(source)
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == 'enabled')
    index = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'drive_hold_gain_arrays') + 1
    enabled = ast.parse("self.expected['kd']=self.original['kd'].copy()\n"
        "self.expected['kd'][self.drive]=_HOLD_DAMPING\n"
        "self._set('set_dof_dampings',self.expected['kd'])").body
    released = ast.parse("self.expected['kd']=self.original['kd'].copy()\n"
        "self._set('set_dof_dampings',self.expected['kd'])").body
    block.body[index:index] = enabled; block.orelse[0:0] = released
    namespace = dict(original.__globals__, _HOLD_DAMPING=HOLD_DAMPING)
    exec(compile(ast.fix_missing_locations(tree), '<common_hold_damping_only>', 'exec'), namespace)
    value = namespace[original.__name__]
    del block.body[index:index+len(enabled)]; del block.orelse[:len(released)]
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
        raise ValueError('Undo must recover every original hold/ACK/target/restore statement')
    value.damping_binding = dict(hold_stiffness_Nm_rad=10000., hold_damping_Nm_s_rad=HOLD_DAMPING,
        mass_surrogate_kg=265., drive_radius_m=.074, assumed_equal_two_drive_mass_share=True,
        surrogate_effective_inertia_kg_m2=EFFECTIVE_INERTIA, surrogate_damping_ratio=2.,
        NOT_identified_contact_or_articulated_inertia_or_stability_guarantee=True,
        original_source_damping_restored_on_release_and_cleanup=True,
        original_torque_limits_passive_entries_capture_and_5mm_s_guard_unchanged=True,
        exact_original_transaction_AST_recovered_on_undo=True)
    return value


class DampedHoldAdapter(base.goal.parent.sequence.lifecycle.motion.RetainedTensorDriveAdapter):
    request_hold = hold_implementation()


def bind_adapter(port):
    old = port.adapter
    if old.dirty or old.capture_count or old.pending_stamp is not None:
        raise ValueError('Untouched original baseline before the single route required')
    port.adapter = DampedHoldAdapter(old.view, port.raw['runtime_motor_baseline'], encode=old.encode,
        check_context=old.check_context, paused_check=old.paused_check,
        physics_stamp=old.last_stamp, authorized=True)


def contract(profile, *, simulation_budget_s, wall_budget_s=900.):
    value = copy.deepcopy(base.goal.contract(profile, simulation_budget_s=simulation_budget_s,
        wall_budget_s=wall_budget_s))
    value.update(version=VERSION, parent_controller_version=base.goal.VERSION,
        common_hold_control_candidate=hold_implementation().damping_binding,
        all_methods_share_same_candidate_and_original_speed_and_acceptance=True,
        actual_gain_candidate_requires_physical_static_prehold_and_full_route_validation=True)
    return value


def controller(plan, geometry, *, profile, simulation_budget_s, start_time_s=0., wall_budget_s=900.):
    value = base.goal.controller(plan, geometry, profile=profile, simulation_budget_s=simulation_budget_s,
        start_time_s=start_time_s, wall_budget_s=wall_budget_s)
    value.contract = contract(profile, simulation_budget_s=simulation_budget_s, wall_budget_s=wall_budget_s)
    return value


def private_sequence():
    original = base.goal.private_sequence()
    return SimpleNamespace(**dict(vars(original), unified_geometry_lifecycle_controller=controller,
        unified_geometry_lifecycle_contract=contract))


def candidate():
    return SimpleNamespace(**dict(vars(base.shared.candidate), VERSION=VERSION,
        private_sequence=private_sequence, contract=contract))


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT / 'sources.json'))


def export_protocol():
    old = runtime.read(base.PROTOCOL)
    value = dict(old, version=VERSION,
        sole_additional_actual_change='original10000 hold stiffness; temporary hold-only damping from mass/radius surrogate',
        common_hold_control_candidate=hold_implementation().damping_binding,
        previous1000_stiffness_candidate_REJECTED_as_not_sufficient=True,
        previous1000_actual_HOLD_preserved=str(base.OUTPUT / 'stopped_on_P.json'),
        source_static_policy='original support static unchanged; new hold damping screened in original own2s route prehold',
        no_claim_that_planner_model_contains_captured_wheel_contact_PD_dynamics=True)
    # Remove the rejected v3 policy rather than ambiguously inherit its gains.
    value.pop('own_route_prehold_gain_policy')
    value['own_route_prehold_gain_policy'] = 'original10000 stiffness, declared temporary damping; same measured2s gate'
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Prospective damping definition frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(base.__file__), Path(base.goal.__file__),
            Path(base.shared.__file__), Path(base.goal.parent.sequence.__file__))
        pins = {str(p.resolve()): runtime.sha(p) for p in files}
        runtime.evidence.write_exclusive(OUTPUT / 'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip', pins)
    for name in base.CELL_IDS:
        source = base.OUTPUT / 'model' / name / 'model_screen.json'
        target = OUTPUT / 'model' / name / 'model_screen.json'
        if target.exists():
            continue
        measured = runtime.read(source)
        for row in measured['cases']:
            if runtime.sha(row['trace']) != row['trace_sha256']:
                raise ValueError('Reused nominal model output changed')
        runtime.evidence.write_exclusive(target, dict(measured, controller_version=VERSION,
            reused_same_pose_response_model_without_hold_PD=True,
            reused_source=str(source), reused_source_sha256=runtime.sha(source),
            new_model_runs=0, no_actual_stability_claim_from_reuse=True))
    return PROTOCOL


def bound_cell():
    call = runtime.bind(base.shared.bound_cell, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, candidate=candidate())()
    original = call.__globals__['runtime']; context_for = original.context_for
    private = SimpleNamespace(**dict(vars(original), context_for=lambda folder, actor, geometry:
        context_for(folder, actor + '_hold_damping_v4', geometry)))
    parent = call.__globals__['PrimaryRouteHost']
    class DampedHost(parent):
        async def preflight(self):
            value = await super().preflight(); bind_adapter(self)
            return dict(value, explicit_hold_damping_control_candidate=hold_implementation().damping_binding)

        def command_readback(self):
            value = super().command_readback()
            actual = self.adapter._motor()
            value.update({f'drive_{axis}_actual_kd': float(actual['kd'][index])
                for axis, index in zip(('front', 'rear'), self.adapter.drive)})
            return value
    return runtime.bind(call, runtime=private, PrimaryRouteHost=DampedHost)


def policy_file():
    return runtime.bind(base.policy_file, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, candidate=candidate())()


async def run(parent, *, authorized=False):
    return await runtime.bind(base.run, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, VERSION=VERSION,
        verify_sources=verify_sources, bound_cell=bound_cell, policy_file=policy_file)(parent, authorized=authorized)


def dispatch():
    export_protocol(); verify_sources()
    if (OUTPUT / 'live').exists():
        raise ValueError('Consumed physical candidate; no redispatch')
    source = ("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_remaining_stop_v3',None)\n"
        "assert parent is not None and parent.done() and not parent.exception(),'Completed exact parent required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is parent,'No intervening trial'\n"
        "assert not hasattr(builtins,'ijcas_revision_hold_damping_v4'),'Already reserved'\n"
        f"p=Path({str(Path(__file__).resolve())!r})\n"
        "m=types.ModuleType('hospital_revision_hold_damping');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run(parent,authorized=True))\n"
        "builtins.ijcas_revision_hold_damping_v4=job\n"
        "print({'common_hold_damping_pairs_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT / 'reservation.json', dict(source=source,
        previous_candidate_HOLD_not_retried_or_overwritten=True, automatic_physical_retry=False))
    return base.shared.runner.remote(source)


if __name__ == '__main__':
    print(dispatch())

"""Task-only bindings for the frozen common hospital v3 executor.

The H4 implementation is immutable evidence. Reuse its exact validation,
reference projection and controller method code instead of editing it or
copying a controller per task. Only H4/H5 forward references are currently
supported; reverse/integrated missions require their own declared workflow.
This module has no live scene/timeline/physics access or motion authority.
"""

from types import FunctionType, SimpleNamespace

import hospital_h4_path_execution as shared


SCENARIOS = ('H4_ward_corridor_turn', 'H5_obstacle_bypass_return')


def _bind(function, **values):
    result=FunctionType(function.__code__,dict(function.__globals__,**values),
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    return result


def task_execution_binding(scenario, *, simulation_budget_s=180.):
    """Construct isolated identities/resource settings, not a new controller.

    240/300s are prospective offline resource bindings, never a physical
    authorization or a post-hoc extension of any recorded trial. All methods
    in an execution bundle must use the same declared duration.
    """
    if scenario not in SCENARIOS:
        raise ValueError('Only explicit H4/H5 bounded forward task bindings are implemented')
    if type(simulation_budget_s) not in (int,float) or simulation_budget_s not in (180.,240.,300.):
        raise ValueError('Explicit common 180/240/300s prospective resource binding required')
    config=dict(shared.CONTROLLER_CONFIG)
    validate=_bind(shared.validate_plan,SCENARIO=scenario)
    Reference=type('HospitalTaskPathReference',(shared.PathReference,),{
        '__init__':_bind(shared.PathReference.__init__,validate_plan=validate)})
    Controller=type('HospitalTaskTrackingController',(shared.H4TrackingController,),{
        '__init__':_bind(shared.H4TrackingController.__init__,PathReference=Reference,CONTROLLER_CONFIG=config)})
    fields={key:value for key,value in vars(shared).items() if not key.startswith('__')}
    fields.update(SCENARIO=scenario,EXECUTION_VERSION='stop_align_v3_explicit_forward_task_binding_v1',
        validate_plan=validate,PathReference=Reference,H4TrackingController=Controller,
        TrackingController=Controller,CONTROLLER_CONFIG=config,
        GUARDS=dict(shared.GUARDS,maximum_simulation_duration_s=float(simulation_budget_s)),
        ACCEPTANCE=dict(shared.ACCEPTANCE),COMMAND_LIMITS=dict(shared.COMMAND_LIMITS),
        original_reference_projection_and_control_method_code_unchanged=True,
        current_binding_is_NOT_a_motion_authorization=True)
    return SimpleNamespace(**fields)

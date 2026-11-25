"""Preserve zero-argument-super closure in a private authority-only binding.

The completed static's source file is immutable. This separate connector
continues only a route that never reserved or requested Play; no static retry,
Stop, source construction, target/gain assignment, or safety change.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np
import hospital_controlled_runtime as runtime


def route_initializer():
    original = runtime.lc.RetainedMissionSDKHost.__init__
    source = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree = copy.deepcopy(source)
    body = tree.body[0].body
    found = [i for i,n in enumerate(body) if isinstance(n,ast.ImportFrom)
        and n.module == 'hospital_mission_sequence'
        and 'checked_route_lifecycle_contract' in [a.name for a in n.names]]
    if len(found) != 1:
        raise ValueError('One original authority import required')
    index = found[0]+1
    body[index:index] = ast.parse('checked_route_lifecycle_contract = _controlled_contract\n'
                                 'REMAINING_NO_NAV2_APPROVAL = _controlled_approval').body
    undo = copy.deepcopy(tree)
    del undo.body[0].body[index:index+2]
    if ast.dump(undo,include_attributes=False) != ast.dump(source,include_attributes=False):
        raise ValueError('Complete original physical initializer must be recoverable')
    # Compile in class scope so Python emits the original __class__ freevar,
    # then bind its ORIGINAL closure, not a different super() starting point.
    wrapper = ast.ClassDef(name='_PrivateRouteInitializer',bases=[],keywords=[],
                           body=tree.body,decorator_list=[])
    namespace = dict(original.__globals__, _controlled_contract=runtime.controlled_contract,
                     _controlled_approval=copy.deepcopy(runtime.APPROVAL))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
                 '<controlled_authority_preserved_class_cell>','exec'),namespace)
    compiled = namespace['_PrivateRouteInitializer'].__dict__['__init__']
    if compiled.__code__.co_freevars != original.__code__.co_freevars:
        raise ValueError('Original super/class closure layout required')
    result = FunctionType(compiled.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


class ControlledRouteHost(runtime.lc.RetainedMissionSDKHost):
    __init__ = route_initializer()


async def continue_unstarted_route(cell_folder, *, authorized=False):
    if authorized is not True:
        raise ValueError('Explicit current additional-experiment authority required')
    folder = Path(cell_folder).resolve()
    if folder.parent != runtime.BASE:
        raise ValueError('Only the current own controlled engineering cell is supported')
    output = folder/'route'
    if any((output/n).exists() for n in ('single_route_reserved.json','route_trace.csv','route_readback.json')):
        raise FileExistsError('Route already consumed; no retry')
    error = runtime.read(folder/'workflow_error.json')
    if error['error'] != 'RuntimeError: super(): __class__ cell not found':
        raise ValueError('Only this documented pre-Play constructor failure is in scope')
    packet = runtime.read(output/'sdk_binding.json')
    runtime.checked_approval(packet['explicit_route_approval'])
    runtime.retained.runtime.task.verify_pins(packet['input_sha256'])
    runtime.sequence.held_static_adapter(folder/'static/held_static_readback.json')
    packet = copy.deepcopy(packet)
    packet['input_sha256'][str(Path(__file__).resolve())] = runtime.sha(Path(__file__))
    packet['prePlay_connector_correction'] = dict(
        prior_error=str(folder/'workflow_error.json'), prior_route_reservations=0,
        original_static_preserved=True, original_super_class_closure_preserved=True,
        physical_initializer_AST_unchanged_except_current_authority_identities=True)
    path = output/'sdk_binding_closure_fix_r01.json'
    if path.exists():
        raise FileExistsError('Corrected admission already consumed; no retry')
    runtime.evidence.write_exclusive(path,packet)
    runtime.lc.freeze_workflow_sources(output/'closure_fix_sources.zip',packet['input_sha256'])
    plan = runtime.read(output/packet['plan_filename'])
    geometry = runtime.read(packet['geometry_file'])
    context = runtime.context_for(folder, packet['actor'], geometry)
    context.stationary.RESULT = Path(packet['original_static_record'])
    context.stationary.TRACE = Path(packet['original_static_trace'])
    context.operations.replay_static = lambda ignored: runtime.sequence.held_static_adapter(
        folder/'static/held_static_readback.json')
    sdk = runtime.retained.runtime.live_sdk()
    with np.load(packet['grid_file']) as grid:
        port = ControlledRouteHost(sdk,packet,path,context=context,plan=plan,geometry=geometry,
            occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),
            resolution=float(grid['resolution_m']),authorized=True)
    controller = runtime.sequence.unified_geometry_lifecycle_controller(plan,geometry,
        profile=plan['geometry_profile'],simulation_budget_s=packet['simulation_budget_s'])
    return await runtime.lc.run_lifecycle_once(port,controller,authorized=True,
        sdk_bridge_class=runtime.sequence.RetainFinalTargetsBridge)

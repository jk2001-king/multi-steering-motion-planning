"""Typed camera-default handoff, using the unchanged shared source checks.

The previous exact-value GET raced with another Top pan/zoom. Permit ONLY
the same reviewed path/field roster and the original typed value rules.
All light identities, source/physics checks, and original strict runtime
monitoring stay unchanged. No controller or experiment setting is changed.
"""
import ast
import asyncio
import copy
import inspect
from pathlib import Path
import textwrap
from types import FunctionType,SimpleNamespace

import hospital_revision_display_handoff as base

runtime=base.runtime
OUTPUT=base.experiment.OUTPUT/'display_handoff_r02'
KEY='ijcas_revision_n3_display_handoff_r02'


def changed_roster(current,reviewed):
    return set(current)!=set(reviewed) or any(
        {(r['path'],r['field']) for r in current[k]}!={(r['path'],r['field']) for r in reviewed[k]}
        for k in current)


def checked_adopter(scope):
    source=ast.parse(textwrap.dedent(inspect.getsource(base.adopt_same_source)))
    tree=copy.deepcopy(source)
    target=ast.parse("deltas!=runtime.read(OUTPUT/'definition.json')['expected_display_differences']",mode='eval').body
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(target)]
    if len(candidates)!=1:raise ValueError('One exact numeric-GET equality guard required')
    candidates[0].test=ast.Call(func=ast.Name(id='_changed_roster',ctx=ast.Load()),
        args=[target.left,target.comparators[0]],keywords=[])
    undo=copy.deepcopy(tree)
    next(n for n in ast.walk(undo) if isinstance(n,ast.If) and isinstance(n.test,ast.Call)
        and isinstance(n.test.func,ast.Name) and n.test.func.id=='_changed_roster').test=target
    if ast.dump(undo)!=ast.dump(source):raise ValueError('No physical/actor/setting or write statement may change')
    scope['_changed_roster']=changed_roster
    exec(compile(ast.fix_missing_locations(tree),'<same_source_typed_display_defaults_only>','exec'),scope)
    return scope['adopt_same_source']


def namespace():
    scope=dict(vars(base),OUTPUT=OUTPUT,KEY=KEY,__file__=__file__,__name__=__name__)
    for name,function in vars(base).items():
        if isinstance(function,FunctionType) and function.__module__==base.__name__:
            bound=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
            bound.__kwdefaults__=function.__kwdefaults__;scope[name]=bound
    checked_adopter(scope)
    original_definition=scope['definition']
    def definition():
        result=original_definition()
        result.update(reviewed_typed_defaults_NOT_fixed_numeric_camera_coordinates=True,
            original_value_race_preserved=str(base.OUTPUT/'stopped.json'),
            physical_checks_identical_AST_undo_verified=True)
        result['input_sha256'].update({str(Path(base.__file__).resolve()):runtime.sha(base.__file__),
            str((base.OUTPUT/'stopped.json').resolve()):runtime.sha(base.OUTPUT/'stopped.json')})
        return result
    scope['definition']=definition
    value=SimpleNamespace(**scope)
    scope['dispatch']=lambda:dispatch(value);value.dispatch=scope['dispatch']
    return value


async def ready_run(parent,*,authorized=False):
    import json
    value=namespace();value.verify();block=base.experiment.campaign()
    cell=next(c for c in runtime.read(block.EXECUTION_PROTOCOL)['main_cells'] if c['cell_id']==base.CELL)
    original=block.OUTPUT/'live'/base.CELL
    prep=runtime.read(original/'preparation.json')
    geom=runtime.read(block.folder_for(base.CELL)/'task_geometry.json')
    factory=block.bound_cell(cell).__globals__['runtime'].context_for
    context=factory(OUTPUT/'GET_preview_NOT_a_trial',prep['new_robot_root'].removesuffix('_final_n3_v7'),geom)
    preview=await value.adopt_same_source(context,runtime.retained.runtime.live_sdk(),geom,preview=True)
    runtime.evidence.write_exclusive(OUTPUT/'same_actor_physical_preview_GET.json',dict(
        preview,new_static_trials=0,new_route_trials=0,no_scene_or_timeline_or_motor_command=True))
    return await value.run(parent,authorized=authorized)


def dispatch(value):
    definition=value.definition();runtime.evidence.write_exclusive(OUTPUT/'definition.json',definition)
    runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',definition['input_sha256'])
    source=("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_display_handoff_current as _typed\n"
        f"assert not hasattr(builtins,{KEY!r}),'Already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_typed.ready_run(builtins.{base.KEY},authorized=True))\n"
        "print({'same_B1_first_static_typed_display_handoff_scheduled':True,'physical_policy_changed':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,user_authority=base.AUTHORITY))
    return base.experiment.original.predecessor.control.base.shared.runner.remote(source)


if __name__=='__main__':namespace().all_steps()

"""Correct anonymous COPY parsing for the recorded native #sdf Session.

Two proved operational deltas only: typed display defaults rather than a
stale numeric GET, and matching the native COPY file format. Original source,
physics, controller, strict static/route and success checks are untouched.
"""
import ast
import asyncio
import copy
import inspect
from pathlib import Path
import textwrap

import hospital_revision_display_handoff_current as prior

runtime=prior.runtime
OUTPUT=prior.base.experiment.OUTPUT/'display_handoff_r03'
KEY='ijcas_revision_n3_display_handoff_r03'


def checked_adopter(scope):
    source=ast.parse(textwrap.dedent(inspect.getsource(prior.base.adopt_same_source)))
    tree=copy.deepcopy(source)
    target=ast.parse("deltas!=runtime.read(OUTPUT/'definition.json')['expected_display_differences']",mode='eval').body
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(target)]
    formats=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='CreateAnonymous' and len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='.usda']
    if len(matches)!=1 or len(formats)!=1:raise ValueError('Exact two operational anchors required')
    matches[0].test=ast.Call(ast.Name('_changed_roster',ast.Load()),[target.left,target.comparators[0]],[])
    formats[0].args[0]=ast.parse("'.usda' if current['Session'].lstrip().startswith('#usda') else '.sdf'",mode='eval').body
    undo=copy.deepcopy(tree)
    next(n for n in ast.walk(undo) if isinstance(n,ast.If) and isinstance(n.test,ast.Call)
        and isinstance(n.test.func,ast.Name) and n.test.func.id=='_changed_roster').test=target
    next(n for n in ast.walk(undo) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='CreateAnonymous').args[0]=ast.Constant('.usda')
    if ast.dump(undo)!=ast.dump(source):raise ValueError('All physical and source checks must be AST-identical')
    scope['_changed_roster']=prior.changed_roster
    exec(compile(ast.fix_missing_locations(tree),'<same_source_native_layer_COPY_only>','exec'),scope)
    return scope['adopt_same_source']


def namespace():
    factory=runtime.bind(prior.namespace,OUTPUT=OUTPUT,KEY=KEY,checked_adopter=checked_adopter,
        __file__=__file__,__name__=__name__)
    value=factory()
    return value


async def preview_current():
    # Use the already frozen r02 definition only for this read-only proof.
    scope=dict(prior.namespace().adopt_same_source.__globals__)
    adopter=checked_adopter(scope)
    block=prior.base.experiment.campaign()
    cell=next(c for c in runtime.read(block.EXECUTION_PROTOCOL)['main_cells'] if c['cell_id']==prior.base.CELL)
    old=block.OUTPUT/'live'/prior.base.CELL;prep=runtime.read(old/'preparation.json')
    geom=runtime.read(block.folder_for(prior.base.CELL)/'task_geometry.json')
    factory=block.bound_cell(cell).__globals__['runtime'].context_for
    context=factory(OUTPUT/'GET_preview_NOT_a_trial',prep['new_robot_root'].removesuffix('_final_n3_v7'),geom)
    result=await adopter(context,runtime.retained.runtime.live_sdk(),geom,preview=True)
    runtime.evidence.write_exclusive(OUTPUT/'same_actor_physical_preview_GET.json',dict(result,
        no_scene_timeline_motor_static_or_route_command=True))
    return result


async def ready_run(parent,*,authorized=False):
    value=namespace()
    return await value.run(parent,authorized=authorized)


def dispatch(value):
    raw=value.definition()
    raw.update(native_Session_COPY_format_corrected=True,physical_checks_AST_identical=True)
    raw['input_sha256'].update({str(Path(prior.__file__).resolve()):runtime.sha(prior.__file__),
        str((OUTPUT/'same_actor_physical_preview_GET.json').resolve()):runtime.sha(OUTPUT/'same_actor_physical_preview_GET.json')})
    runtime.evidence.write_exclusive(OUTPUT/'definition.json',raw)
    runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',raw['input_sha256'])
    source=("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_display_handoff_native as _native\n"
        f"assert not hasattr(builtins,{KEY!r}),'Already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_native.ready_run(builtins.{prior.base.KEY},authorized=True))\n"
        "print({'same_B1_first_static_native_format_handoff_scheduled':True,'physical_policy_changed':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,user_authority=prior.base.AUTHORITY))
    return prior.base.experiment.original.predecessor.control.base.shared.runner.remote(source)


def all_steps():
    value=namespace()
    # The shared monitor invokes this function through its private globals.
    value.all_steps.__globals__['dispatch']=lambda:dispatch(value)
    return value.all_steps()


if __name__=='__main__':all_steps()

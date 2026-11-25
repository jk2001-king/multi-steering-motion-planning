"""Sequential initial-angle supplement after the repaired primary queue.

Preserves the original finite design, old blocked waiter and all negative
outcomes. This does not dispatch a second concurrent physical experiment.
"""
import asyncio
from pathlib import Path

import hospital_revision_placement as placement
import hospital_revision_initial_conditions as initial

runtime=placement.runtime


def initial_context(context,source_asset):
    context=initial.private_source_context(context,source_asset)
    placement.source_context(context)
    context.initial_condition_source_helpers=context.placement_source_helpers
    return context


def cell_bindings(cell):
    return runtime.bind(initial.cell_bindings,private_source_context=initial_context,
        __file__=__file__)(cell)


async def after_primary(parent):
    import builtins
    import hospital_revision_execution as execution
    try:
        result=await asyncio.shield(parent)
        if result.get('status')!='FINITE_QUEUE_FINISHED':
            raise ValueError('Normally finished primary queue required')
        if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not parent:
            raise ValueError('A different experiment replaced the primary task')
        last=Path(result['last_actual_report'])
        if not execution.restored_boundary(runtime.read(last)):
            raise ValueError('Verified actual final Pause/restoration required')
        current=asyncio.current_task()
        builtins.ijcas_hospital_latest_task_experiment=current
        builtins.ijcas_revision_primary_task=current
        run=runtime.bind(initial.run,cell_bindings=cell_bindings,__file__=__file__)
        result=await run(prior_result=last,authorized=True)
        runtime.evidence.write_exclusive(initial.OUTPUT/'after_primary_result.json',result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(initial.OUTPUT/'after_primary_wrapped_blocked.json',dict(
            error=type(error).__name__+': '+str(error),old_blocked_waiter_preserved=True,
            no_physical_retry_or_replacement=True))
        raise


def dispatch():
    from hospital_revision_runner import remote
    import hospital_revision_reporting as reporting
    path=Path(__file__).resolve()
    files=[Path(placement.__file__).resolve(),path,Path(reporting.__file__).resolve()]
    command="import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
    command+=("parent=getattr(builtins,'ijcas_revision_primary_task',None)\n"
        "assert parent is not None and not parent.done(),'Active repaired primary queue required'\n"
        "assert not hasattr(builtins,'ijcas_revision_wrapped_supplement_waiter'),'Continuation already attached'\n")
    # Placement is already executing: do not replace its live module/objects.
    for file in files[1:]:
        command+=(f"p=Path({str(file)!r})\nm=types.ModuleType({file.stem!r});m.__file__=str(p);sys.modules[m.__name__]=m\n"
            "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n")
    command+=("follow=sys.modules['hospital_revision_followthrough']\n"
        "waiter=asyncio.ensure_future(follow.after_primary(parent))\n"
        "builtins.ijcas_revision_wrapped_supplement_waiter=waiter\n"
        "builtins.ijcas_revision_joint_supplement_waiter=waiter\n"
        "analysis=sys.modules['hospital_revision_reporting']\n"
        "def wrapped_block_tables(ignored):\n"
        "    try:print('[REVIEW BLOCK TABLES]',analysis.export(),flush=True)\n"
        "    except Exception as error:print('[REVIEW BLOCK TABLES ERROR]',repr(error),flush=True)\n"
        "parent.add_done_callback(wrapped_block_tables)\nwaiter.add_done_callback(wrapped_block_tables)\n"
        "print({'initial_angle_block_waiting':True,'concurrent_motion':False,'physical_retries':False,'tables_on_each_block_end':True})\n")
    runtime.evidence.write_exclusive(initial.OUTPUT/'dispatch_after_wrapped_primary.json',dict(
        command=command,protocol_sha256=runtime.sha(initial.PROTOCOL),old_blocked_receipt_preserved=True))
    runtime.lc.freeze_workflow_sources(initial.OUTPUT/'wrapped_connector_sources.zip',
        {str(p):runtime.sha(p) for p in (*files,Path(initial.__file__).resolve())})
    return remote(command)


if __name__=='__main__':
    print(dispatch())

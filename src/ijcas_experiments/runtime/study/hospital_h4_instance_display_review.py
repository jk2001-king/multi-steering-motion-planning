"""GET-only USD instance roots and EXISTING Fabric caches, not population.

    Visual instance-proxy leaves can legitimately be absent from Fabric. Read
    their instance-root caches separately; do not label missing leaves as a
    renderer fault. Never populate a missing prim or update its transforms.
"""

import hospital_h4_robot_render_review as render


def usd_instances(stage,geometry):
    actor=render.display.ROBOT;instances={};noninstance=[]
    for row in geometry['geometry']:
        if row['purpose'] not in ('default','render','proxy'):continue
        leaf=stage.GetPrimAtPath(row['path']);parent=leaf;chain=[]
        while parent and str(parent.GetPath()).startswith(actor+'/'):
            if parent.IsInstance():
                proto=parent.GetPrototype()
                chain.append(str(parent.GetPath()))
                instances.setdefault(str(parent.GetPath()),dict(
                    instance_path=str(parent.GetPath()),
                    USD_prototype_path=str(proto.GetPath()) if proto else None,
                    visual_proxy_leaves=[]))['visual_proxy_leaves'].append(str(leaf.GetPath()))
            parent=parent.GetParent()
        if not chain:noninstance.append(str(leaf.GetPath()))
    return dict(instance_roots=[instances[k] for k in sorted(instances)],
        noninstanced_render_leaves=noninstance,
        missing_Fabric_proxy_leaf_is_NOT_a_render_failure_diagnosis=True)


def existing_instance_cache(stage,instances):
    from pxr import UsdUtils
    try:
        from usdrt import Usd as RtUsd,Sdf as RtSdf
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not RtUsd.Stage.SimStageWithHistoryExists(stage_id):
            return dict(existing_sim_Fabric_stage=False,cache_created_or_populated=False)
        rt=RtUsd.Stage.Attach(stage_id)
        paths=[row['instance_path'] for row in instances['instance_roots']]
        return dict(existing_sim_Fabric_stage=True,
            instance_root_rows=render.inspect_cache_rows(stage,rt,paths,RtSdf.Path),
            cache_created_or_populated=False,
            missing_or_differing_cache_is_NOT_alone_an_isolated_render_cause=True)
    except Exception as error:
        return dict(optional_cache_error=type(error).__name__+': '+str(error),
            cache_created_or_populated=False)

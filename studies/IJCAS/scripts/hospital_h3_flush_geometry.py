"""Explicit H3 flush-floor variant; never silently authorize a terrain obstacle."""

import copy
import hashlib
import json


CONDITION="H3_flush_fully_open_continuous_floor"
THRESHOLD="/World/Hospital/Geo_M_DoorFloor20"
MARKER="/World/IJCAS_HospitalH3FlushEnvironment"
COMPONENT_MARKER="/World/IJCAS_HospitalH3FlushComponent"
SUPPORT_TOP_M=.001


def variant_inventory(fully_open_audit,*,live_threshold_inactive=False,other_assets_verified=False):
    variant=fully_open_audit.get("geometry_variant",{})
    if (variant.get("condition")!="fully_open_baseline_session_only"
            or not variant.get("actual_new_leaf_vertices_exported") or not variant.get("collider_flags_unchanged")):
        raise ValueError("Need an actual fully-open door export, not a predicted/partial-open asset")
    if not live_threshold_inactive or not other_assets_verified:
        raise ValueError("Live visual/collision removal and unchanged other assets must be verified")
    result=copy.deepcopy(fully_open_audit)
    matches=[a for a in result["local_assets"] if a["path"]==THRESHOLD]
    if len(matches)!=1 or not matches[0]["collision"]["collision_prim_paths"]:
        raise ValueError("Expected exactly the selected threshold asset with exported collision evidence")
    result["local_assets"]=[a for a in result["local_assets"] if a["path"]!=THRESHOLD]
    result["explicitly_suppressed_assets"]=[matches[0]]
    for row in result.get("candidate_paths",[]):
        if row["path"]==THRESHOLD or row["path"].startswith(THRESHOLD+"/"):
            row.update(active=False,effective_collision_enabled=False,suppressed_by_explicit_flush_condition=True)
    for row in result.get("targets",{}).values():
        if row.get("path")==THRESHOLD:
            row.update(active=False,effective_collision_enabled=False,suppressed_by_explicit_flush_condition=True)
    result["geometry_variant"]={"condition":CONDITION,"base_door_condition":"fully_open_baseline_session_only",
        "only_selected_threshold_suppressed_after_fully_open_export":True,
        "threshold_root":THRESHOLD,"visual_and_collision_suppressed_together":True,
        "live_geometry_verified":True,"source_USD_saved":False,"PhysX_executed":False,
        "is_a_new_environment_condition_not_relabelled_original_trial":True,
        "non_threshold_asset_sha256":{a["path"]:hashlib.sha256(json.dumps(a,sort_keys=True,allow_nan=False).encode()).hexdigest()
            for a in fully_open_audit["local_assets"] if a["path"]!=THRESHOLD},
        "canonical_continuous_support_preparation_and_static_gate_pending":True}
    return result


def verify_variant(audit):
    v=audit.get("geometry_variant",{})
    if (v.get("condition")!=CONDITION or not v.get("live_geometry_verified")
            or not v.get("visual_and_collision_suppressed_together")
            or v.get("source_USD_saved") is not False):
        raise ValueError("Need a live verified explicit flush environment variant")
    removed=audit.get("explicitly_suppressed_assets",[])
    if len(removed)!=1 or removed[0]["path"]!=THRESHOLD:
        raise ValueError("No other hospital obstacle may be removed")
    if any(a["path"]==THRESHOLD for a in audit["local_assets"]):
        raise ValueError("Selected threshold still appears in active collision inventory")
    expected=v.get("non_threshold_asset_sha256",{})
    if (not expected or len(audit["local_assets"])!=len(expected)
            or {a["path"] for a in audit["local_assets"]}!=set(expected)):
        raise ValueError("The remaining hospital asset inventory changed")
    for a in audit["local_assets"]:
        if hashlib.sha256(json.dumps(a,sort_keys=True,allow_nan=False).encode()).hexdigest()!=expected[a["path"]]:
            raise ValueError("A non-threshold asset changed: "+a["path"])
    return True

"""PAUSED live-state readback + NEW measured suffix overlay, NO motion.

Uses only trusted JSON/standard helpers in embedded Python (no optimizer or
ROS/model package import). Existing simulation tensor handles are READ ONLY.
Only annotations in a new Session Layer root and a new receipt are written.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _show_h3b_measured_suffix():
    import csv
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Usd, UsdGeom

    workspace=_ijcas_workspace()
    scripts=_ijcas_path(workspace, 'studies/IJCAS/scripts');gate=_ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate')
    path=gate/'hospital_h3b_measured_suffix_reviews/review_r02/measured_suffix_review.json'
    support=runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    review=json.loads(path.read_bytes());support['verify_files'](review['input_sha256'])
    if ((path.parent/'INCOMPLETE_REVIEW.json').exists()
            or review.get('version')!='H3B_measured_suffix_first_reverse_model_review_v1'
            or review.get('all_first_reverse_model_cases_admitted') is not True
            or review.get('automatic_reverse_motion_authorized') is not False
            or review.get('reverse_PhysX_executed') is not False):
        raise RuntimeError('Need the complete admitted NEW suffix MODEL review, not an incomplete report or motion receipt')
    suffix_path=Path(review['suffix_candidate']);support['verify_files']({suffix_path:review['suffix_candidate_sha256']})
    suffix=json.loads(suffix_path.read_bytes());support['verify_files'](suffix['input_sha256'])
    if suffix['geometry_audit']['verdict']!='PASS' or not all(suffix['geometry_audit']['checks'].values()):
        raise RuntimeError('Measured suffix geometry is not admitted')
    receipt=review['first_component'];first_path=Path(receipt['first_result'])
    support['verify_files']({first_path:receipt['first_result_sha256']});first=json.loads(first_path.read_bytes())
    if first.get('verdict')!='PASS' or first.get('original_motor_gains_restored') is not True:
        raise RuntimeError('Original actual first-forward PASS/gain restoration missing')
    if sorted(c['dt_s'] for c in review['model_cases'])!=[.01,.02]:
        raise RuntimeError('Both integration resolutions required')
    for case in review['model_cases']:
        support['verify_files']({case['trace_file']:case['trace_sha256']})
        if not all(case['checks'].values()):raise RuntimeError('First reverse MODEL case not admitted')
    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Keep the SAME first-forward PASS PAUSED. Do not Stop/reopen/reset or manually Play')
    for name,task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('A hospital trial is still running')
    context=omni.usd.get_context();stage=context.get_stage()
    marker=stage.GetPrimAtPath(support['COMPONENT']) if stage else None
    if (not marker or not marker.IsValid()
            or marker.GetCustomDataByKey('status')!='h3b_course_first_leg_pilot_finished_fresh_setup_required'
            or marker.GetCustomDataByKey('last_h3b_course_first_leg_pilot_gate')!=str(first_path)):
        raise RuntimeError('Scene is not the SAME completed first-forward PASS; do not substitute another start')
    _,manifest,geometry,inventory=support['load_setup'](Path(first['geometry_manifest']).parent,workspace)
    support['live_geometry_check'](stage,manifest,geometry,inventory,
        set(first['physics_preparation_snapshot']['disabled_legacy_floor_colliders']))
    root_before=stage.GetRootLayer().ExportToString();session=stage.GetSessionLayer();session_before=session.ExportToString()
    sim_view=SimulationManager.get_physics_sim_view()
    if sim_view is None:raise RuntimeError('Existing simulation view unavailable; no initialization/reset attempted')
    view=sim_view.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count!=1:raise RuntimeError('Need exactly one existing live articulation')
    names=list(view.shared_metatype.dof_names)
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        elif hasattr(value,'numpy'):value=value.numpy()
        result=np.asarray(value,dtype=float).reshape(-1)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite tensor readback')
        return result
    positions=array(view.get_dof_positions());transform=array(view.get_root_transforms())
    root_velocity=array(view.get_root_velocities())
    if len(positions)!=12 or len(names)!=12 or len(transform)!=7 or len(root_velocity)!=6:
        raise RuntimeError('Unexpected tensor shape; no state mutation attempted')
    retained=receipt['retained_all_joint_positions_rad']
    if set(names)!=set(retained):raise RuntimeError('Runtime joint names differ from saved twelve joints')
    joint_difference=max(abs(float(q)-retained[name]) for name,q in zip(names,positions))
    x,y,z,w=transform[3:];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    saved=first['final_world_xyyaw'];position_difference=math.hypot(transform[0]-saved[0],transform[1]-saved[1])
    yaw_difference=math.atan2(math.sin(yaw-saved[2]),math.cos(yaw-saved[2]))
    speed=math.hypot(root_velocity[0],root_velocity[1])
    if position_difference>.001 or abs(yaw_difference)>math.radians(.1) or joint_difference>.001 or speed>.005:
        raise RuntimeError(f'Live/saved state differs: {position_difference*1000:.3f}mm/{math.degrees(yaw_difference):.4f}deg, joint={joint_difference:.6f}rad, speed={speed*1000:.3f}mm/s. Keep PAUSED; do not reset to force a match')
    if stage.GetRootLayer().ExportToString()!=root_before or session.ExportToString()!=session_before:
        raise RuntimeError('Read-only tensor/geometry check unexpectedly changed the stage')
    root_path='/World/IJCAS_HospitalH3BMeasuredSuffixReview'
    if stage.GetPrimAtPath(root_path).IsValid():raise RuntimeError('Overlay already exists; do not stack or rerun')
    sx,sy,syaw=review['original_mission_start_xyyaw'];cs,sn=math.cos(syaw),math.sin(syaw)
    def world(p):
        x,y,yaw=p
        return sx+cs*x-sn*y,sy+sn*x+cs*y,syaw+yaw
    data=suffix['geometry_audit'];poses=[world(p) for p in data['poses_local_xyyaw']]
    stops=[0]+suffix['nominal_timing']['cusp_sample_indices']+[len(poses)-1]
    fine=next(c for c in review['model_cases'] if c['dt_s']==.01)
    with Path(fine['trace_file']).open(newline='') as stream:
        predicted=[world([float(r[k]) for k in ('x_m','y_m','yaw_rad')]) for r in csv.DictReader(stream)]
    def curve(name,points,color,width=.010):
        obj=UsdGeom.BasisCurves.Define(stage,root_path+'/'+name)
        obj.CreateTypeAttr('linear');obj.CreateWrapAttr('nonperiodic')
        obj.CreatePointsAttr([Gf.Vec3f(*p) for p in points]);obj.CreateCurveVertexCountsAttr([len(points)])
        obj.CreateWidthsAttr([width]);obj.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    def rectangle(name,pose,color,z=3.30,clearance=.010):
        x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw);a,b=.815+clearance,.4825+clearance
        curve(name,[(x+c*u-s*v,y+s*u+c*v,z) for u,v in ((a,b),(a,-b),(-a,-b),(-a,b),(a,b))],color,.006)
    output_base=gate/'hospital_h3b_measured_suffix_live_reviews';output_base.mkdir(parents=True,exist_ok=True);number=1
    while (output:=output_base/f'live_r{number:02d}').exists():number+=1
    output.mkdir()
    try:
        with Usd.EditContext(stage,session):
            annotation=UsdGeom.Xform.Define(stage,root_path).GetPrim()
            annotation.SetCustomDataByKey('status','NEW_MEASURED_SUFFIX_GEOMETRY_FIRST_REVERSE_MODEL_ONLY_NO_DRIVE')
            annotation.SetCustomDataByKey('review_path',str(path))
            for i,(a,b) in enumerate(zip(stops[:-1],stops[1:])):
                color=(.95,.2,.65) if data['leg_directions'][i]<0 else (.05,.8,.95)
                curve('SuffixLeg_'+str(i),[(x,y,3.37) for x,y,_ in poses[a:b+1]],color,.019)
            for i in range(0,len(poses),max(1,len(poses)//28)):
                rectangle('FullEnvelope10mm_'+str(i),poses[i],(.15,.6,.9),3.28)
            rectangle('ACTUAL_FirstPASS_Start',saved,(.1,.85,.25),3.32)
            rectangle('UNCHANGED_FinalMissionGoal',review['original_mission_goal_xyyaw'],(1.,.6,.05),3.33)
            curve('FirstReverseMODEL_NOT_PhysX',[(x,y,3.40) for x,y,_ in predicted],(.95,.85,.1),.012)
            if stage.GetRootLayer().ExportToString()!=root_before or support['sha'](geometry['source_stage'])!=geometry['source_stage_sha256']:
                raise RuntimeError('Source hospital USD changed')
        if timeline.is_playing() or timeline.is_stopped():raise RuntimeError('Timeline changed during read-only review')
        report=dict(status='PAUSED_SAVED_STATE_CONTINUITY_AND_NEW_SUFFIX_OVERLAY_RECORDED_NO_DRIVE_AUTHORITY',
            first_result=str(first_path),first_result_sha256=support['sha'](first_path),
            suffix_candidate=str(suffix_path),suffix_candidate_sha256=support['sha'](suffix_path),
            model_review=str(path),model_review_sha256=support['sha'](path),
            position_difference_m=position_difference,yaw_difference_deg=math.degrees(yaw_difference),
            maximum_joint_difference_rad=joint_difference,measured_live_speed_m_s=speed,
            live_world_xyyaw=[float(transform[0]),float(transform[1]),yaw],
            retained_all_joint_positions_rad=dict(zip(names,map(float,positions))),
            original_first_verdict_unchanged='PASS',physical_state_reset=False,source_USD_unchanged=True,
            annotations_only_in_session_layer=True,automatic_reverse_motion_authorized=False,
            full_suffix_dynamics_or_PhysX_verified=False,reverse_PhysX_executed=False,
            script_sha256=support['sha'](_ijcas_path(workspace, 'isaac/scripts/isaac_show_ijcas_hospital_h3b_measured_suffix.py')))
        target=output/'live_suffix_review.json'
        with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    except Exception:
        session.ImportFromString(session_before)
        raise
    context.get_selection().set_selected_prim_paths([root_path],True)
    print('=== H3-B MEASURED-STATE SUFFIX LIVE REVIEW (NO MOTION) ===')
    print('saved/live continuity:',round(position_difference*1000,4),'mm /',round(math.degrees(yaw_difference),6),'deg; all 12 joints retained')
    print('original first-forward verdict: PASS; NEW suffix geometry + nominal timing: PASS')
    print('first reverse MODEL 10/20ms: admitted; NOT reverse/full H3 PhysX success')
    print('green=ACTUAL saved start; orange=UNCHANGED final goal; cyan=forward; magenta=reverse; blue=envelope +10mm; yellow=first reverse MODEL')
    print('NEW measured-state replanning, not unchanged frozen four-method comparison. F then Top View.')
    print('saved:',target)
    print('Keep PAUSED. Do not reset/reopen/Play/save or run old controllers. Dedicated reverse handoff/controller validation is still pending.')


_show_h3b_measured_suffix()

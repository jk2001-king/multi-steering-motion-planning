"""Shared bidirectional mission execution prototype; NO live adapter or authority.

Compose the frozen forward/v3 controller, native reference and measured
stop/neutral/alignment/hold ACK protocol. This is a new common executor
version, not a copy of a task/run script or a physical hold/contact model.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import math
import numpy as np
from hospital_path_execution import MODES, ramp, wrap
from hospital_timed_path_execution import steering_angles


def _mission_tracking_implementation():
    """Private original tracking code; replace ONLY its end-handoff predicate.

    Reverse travel uses a VIRTUAL heading+pi and curvature sign, never edits
    a native pose/state. Thus the unchanged feedback gains give physical
    k=k_ref+direction*1.5*yaw_error+2*cross_error. Stop timing at an interior
    cusp is explicitly different from the old final25mm handoff.
    """
    import ast
    import hashlib
    import inspect
    import textwrap
    import hospital_h4_path_execution as original
    function=original.common._v3['_v2']['_legacy']['H1TrackingController'].step
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    expected=ast.dump(ast.parse('remaining <= .025 and math.hypot(pose[0]-self.reference.poses[-1][0], pose[1]-self.reference.poses[-1][1]) <= .05',mode='eval').body,include_attributes=False)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.dump(node.test,include_attributes=False)==expected:
            changed.append((node,node.test))
            node.test=ast.parse('_mission_handoff(self,pose,measured_speed,remaining)',mode='eval').body
    if len(changed)!=1:raise ValueError('Exactly one original tracking handoff required; no silent controller fork')
    def handoff(controller,pose,measured_speed,remaining):
        if controller.mission_final_leg:
            return remaining<=.025 and math.hypot(pose[0]-controller.reference.poses[-1][0],pose[1]-controller.reference.poses[-1][1])<=.05
        threshold=.005+.5*measured_speed*max(measured_speed,abs(controller.speed))/.020+controller.mission_tau_u_s*measured_speed
        return (remaining<=threshold and math.hypot(pose[0]-controller.reference.poses[-1][0],pose[1]-controller.reference.poses[-1][1])<=max(.05,threshold+.025))
    namespace=dict(function.__globals__,_mission_handoff=handoff)
    exec(compile(ast.fix_missing_locations(tree),'<private_original_mission_tracking>','exec'),namespace)
    result=namespace[function.__name__]
    changed[0][0].test=changed[0][1]
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise ValueError('Undo must recover the entire original tracking AST')
    result.mission_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        AST_handoff_substitutions=1,undo_reproduces_entire_original_AST=True,
        feedback_gains_path_guards_and_all_other_statements_unchanged=True,
        cusp_stop_threshold_is_nominal_response_timing_NOT_robust_braking_proof=True)
    return result


class MissionPathReference:
    """Frozen native endpoint polyline, split ONLY at direction changes.

    Projection is the existing local monotone method. Endpoint interpolation
    is a control reference, not an exact repropagation/continuous collision
    certificate. Native substep collision audits and actual guards remain
    separate. No physical body, joint or caster state is constructed here.
    """
    def __init__(self,plan):
        import hospital_h4_path_execution as original
        self.actions=[dict(a) for a in plan['selected_actions']]
        self.poses=[tuple(plan['start_world_xyyaw'])]+[tuple(s['world_xyyaw']) for s in plan['selected_states']]
        self.arc=[0.]
        for a,b in zip(self.poses,self.poses[1:]):
            distance=math.hypot(b[0]-a[0],b[1]-a[1])
            if not .001<=distance<=.10:raise ValueError('Finite nonzero original reference edges required')
            self.arc.append(self.arc[-1]+distance)
        self.progress=0.;self.legs=[]
        start=0
        for i in range(1,len(self.actions)+1):
            if i==len(self.actions) or self.actions[i]['direction']!=self.actions[start]['direction']:
                direction=self.actions[start]['direction']
                view=object.__new__(original.PathReference)
                view.actions=[dict(a,direction=1,curvature_1_m=direction*a['curvature_1_m']) for a in self.actions[start:i]]
                view.poses=[(p[0],p[1],wrap(p[2]+(math.pi if direction<0 else 0.))) for p in self.poses[start:i+1]]
                view.arc=[v-self.arc[start] for v in self.arc[start:i+1]];view.progress=0.
                self.legs.append(dict(first_action=start,last_action_exclusive=i,direction=direction,
                    offset_m=self.arc[start],view=view))
                start=i
        self._original_at=original.PathReference.at

    def at(self,distance):return self._original_at(self,distance)


def validate_mission_plan(plan,geometry):
    """Validate actual native H6 schema, NOT a relabelled forward H4 plan."""
    import hospital_h4_path_execution as original
    from patient_initial_stability.steering_lattice import CURVATURES
    measured=plan.get('measured_replan_start')
    nominal=(plan.get('initial_front_rad')==0. and plan.get('initial_rear_rad')==0.
        and plan.get('initial_u_v_r')==[0.,0.,0.])
    if measured is not None:
        from hospital_sensor_navigation import checked_h5_execution_replan_start
        checked_h5_execution_replan_start(plan,geometry)
    if (geometry.get('scenario') not in ('H6_integrated_transfer_mission','H3_doorway_immediate_turn','H1_bedside_departure','H2_bedside_lateral_alignment','H4_ward_corridor_turn','H5_obstacle_bypass_return') or plan.get('scenario')!=geometry['scenario']
            or geometry.get('full_body_rectangle_m')!=[1.630,.965] or geometry.get('full_body_clearance_m')!=.010
            or plan.get('plan_success') is not True or not plan.get('selected_actions')
            or len(plan['selected_actions'])!=len(plan.get('selected_states',[]))
            or plan.get('start_world_xyyaw')!=geometry['start_xyyaw'] or plan.get('goal_world_xyyaw')!=geometry['goal_xyyaw']
            or plan.get('planning_occupancy_digest')!=geometry['planning_occupancy_digest']
            or plan.get('full_envelope_clearance_m')!=.010 or plan.get('common_max_command_speed_m_s')!=.05
            or not nominal and measured is None
            or plan.get('rigid_body_parameter_bounds',{}).get('max_longitudinal_accel_m_s2')!=.020):
        raise ValueError('Same H6 fixed task, original body/limits and complete native MODEL plan required')
    for value in (plan,geometry):
        if value.get('actual_motion_authorized') is not False or value.get('use_for_final_statistics') is not False:
            raise ValueError('Prospective MODEL references grant no drive or final-statistics admission')
    if (plan.get('physical_execution_verified') is not False or plan.get('Nav2_execution_verified') is not False
            or plan.get('integrated_stage_portal_bypass_and_continuous_executor_validation_pending') is not True):
        raise ValueError('Do not claim integrated physical/Nav2 validation from a model path')
    settings=original.method_settings(plan['method'])
    allowed=['coordinated_counterphase'] if plan['method']==original.METHODS[0] else list(MODES)
    if plan['allowed_modes']!=allowed or plan['planner_dynamics']!=settings['planner_dynamics']:
        raise ValueError('Original method dynamics and mode family required')
    for name in ('start_world_xyyaw','goal_world_xyyaw','goal_local_xyyaw'):
        if len(plan[name])!=3 or not all(math.isfinite(v) for v in plan[name]):raise ValueError('Finite task poses required')
    if any(abs(a-b)>1e-10 for a,b in zip(original.relative_pose(plan['goal_world_xyyaw'],plan['start_world_xyyaw']),plan['goal_local_xyyaw'])):
        raise ValueError('Fixed world/local H6 goal frames disagree')
    for action,state in zip(plan['selected_actions'],plan['selected_states']):
        if (action.get('mode') not in allowed or state.get('mode')!=action['mode']
                or type(action.get('direction')) is not int or action['direction'] not in (-1,1)
                or action.get('curvature_1_m') not in CURVATURES
                or type(action.get('target_speed_m_s')) not in (int,float)
                or not math.isfinite(action['target_speed_m_s']) or not 0.<action['target_speed_m_s']<=.05):
            raise ValueError('Original finite bounded native actions required')
        pose=state['world_xyyaw'];local=(state['x_m'],state['y_m'],math.radians(state['yaw_deg']))
        if len(pose)!=3 or not all(math.isfinite(v) for v in (*pose,*local)):
            raise ValueError('Finite original endpoint poses required')
        if any(abs(a-b)>1e-9 for a,b in zip(original.relative_pose(pose,plan['start_world_xyyaw']),local)):
            raise ValueError('Native endpoint frames disagree')


class MissionTrackingController:
    """NEW method-blind bidirectional controller PROTOTYPE, no live adapter.

    Reuse original tracking code/gains through a virtual reverse frame, and
    original v3 terminal code unchanged. Cusp stop/neutral/alignment/release
    are explicit measured/ACK gates. Changing a software leg index never
    resets a body, wheel, caster, velocity or measured steering angle.
    """
    def __init__(self,plan,geometry,*,simulation_budget_s):
        from types import FunctionType
        import hospital_h4_path_execution as original
        validate_mission_plan(plan,geometry)
        if type(simulation_budget_s) not in (int,float) or simulation_budget_s not in (180.,240.,300.,600.):
            raise ValueError('Explicit prospective common300/600s MODEL-only budget required')
        self.reference=MissionPathReference(plan);self.goal=tuple(plan['goal_world_xyyaw'])
        self.budget=float(simulation_budget_s);self.tau_u=float(plan['rigid_body_parameter_bounds']['longitudinal_response_time_s'])
        if not math.isfinite(self.tau_u) or self.tau_u<=0.:raise ValueError('Original positive response time required')
        self.previous_time=self.started=None;self.leg_index=0;self.speed=0.
        self.mode=plan['selected_actions'][0]['mode'];self.curvature=0.;self.phase='path_tracking'
        self.stage='initial_stop';self.stage_started=self.aligned_since=None;self.advance_after_neutral=False
        self.hold_entry_candidate=None;self.pulse_count=0;self.events=[];self.terminal=None
        self.tracking_step=_mission_tracking_implementation()
        def raw(reference,goal):
            init=original.H4TrackingController.__init__
            bound=FunctionType(init.__code__,dict(init.__globals__,PathReference=lambda p:reference),init.__name__,init.__defaults__)
            cls=type('MissionOriginalV3Instance',(original.H4TrackingController,),{'__init__':bound})
            return cls(dict(selected_actions=reference.actions,goal_world_xyyaw=goal))
        self.raw=raw;self._make_tracker()
        if plan.get('measured_replan_start') is not None:
            # The captured state has ALREADY passed the own measured stop/ACK
            # gate. Do not neutralize or repeat a fresh-start preparation.
            captured=plan['measured_replan_start']
            from hospital_sensor_navigation import h5_captured_command_curvature
            self.mode=captured['retained_mode'];self.curvature=h5_captured_command_curvature(captured,plan['geometry_profile'])
            self.stage='release_hold'

    def _make_tracker(self):
        self.leg=self.reference.legs[self.leg_index];view=self.leg['view']
        self.tracker=self.raw(view,view.poses[-1]);self.tracker.mission_final_leg=self.leg_index==len(self.reference.legs)-1
        self.tracker.mission_tau_u_s=self.tau_u

    def _stage(self,stage,timestamp,pose):
        self.stage=stage;self.stage_started=timestamp;self.aligned_since=None
        self.events.append(dict(stage=stage,time_s=timestamp,leg_index=self.leg_index,
            actual_observed_pose=list(pose),command_speed_m_s=self.speed))

    def _dwell(self,ready,timestamp):
        if not ready:self.aligned_since=None;return False
        if self.aligned_since is None:self.aligned_since=timestamp
        return timestamp-self.aligned_since>=.25

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        import hospital_h4_path_execution as original
        if (type(drive_hold_confirmed) is not bool or len(pose)!=3 or len(measured_steer)!=2
                or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))
                or measured_speed<0. or max(abs(v) for v in measured_steer)>math.radians(60.)):
            raise ValueError('Finite actual body/steering/speed and explicit hold ACK required')
        previous_time=self.previous_time;dt=0. if previous_time is None else timestamp-previous_time
        if not 0.<=dt<=.25:raise ValueError('Original0.25s control-gap guard')
        self.previous_time=timestamp
        if self.started is None:self.started=timestamp;self.stage_started=timestamp
        if timestamp-self.started>self.budget:raise TimeoutError('Prospective whole-mission MODEL time cap')
        hold=False;ready=False;info={}
        if self.stage=='tracking':
            if drive_hold_confirmed:raise RuntimeError('Drive hold must be released before tracking motion')
            direction=self.leg['direction'];virtual=(pose[0],pose[1],wrap(pose[2]+(math.pi if direction<0 else 0.)))
            before=(self.mode,self.curvature,self.speed)
            self.tracker.previous_time=previous_time;self.tracker.speed=abs(self.speed)
            self.tracker.mode=self.mode;self.tracker.curvature=direction*self.curvature
            info=self.tracking_step(self.tracker,timestamp,virtual,measured_speed)
            self.reference.progress=max(self.reference.progress,self.leg['offset_m']+self.leg['view'].progress)
            if info['phase']=='path_tracking':
                self.mode=info['mode'];self.curvature=direction*info['curvature_1_m'];self.speed=direction*info['speed_m_s']
            else:
                self.mode,self.curvature,self.speed=before
                if self.tracker.mission_final_leg:
                    self.terminal=self.raw(self.reference,self.goal)
                    for name,value in dict(mode=self.mode,curvature=self.curvature,speed=self.speed,
                            previous_time=previous_time,terminal_started=timestamp,stage_started=timestamp,
                            terminal_stage='entry_braking',phase='terminal_braking').items():setattr(self.terminal,name,value)
                    self._stage('terminal',timestamp,pose)
                else:self._stage('cusp_braking',timestamp,pose)
        if self.stage in ('initial_stop','cusp_braking'):
            self.speed=ramp(self.speed,0.,dt,.020)
            if timestamp-self.stage_started>8.:raise TimeoutError('Measured cusp/initial stopping timeout8s')
            if abs(self.speed)<1e-7 and measured_speed<=.005:
                self.advance_after_neutral=self.stage=='cusp_braking'
                hold=True;self._stage('neutral',timestamp,pose)
        if self.stage=='neutral':
            self.speed=0.;self.curvature=0.;hold=True
            if measured_speed>.005:raise RuntimeError('Hold requested while actual body is not stopped')
            if timestamp-self.stage_started>16.:raise TimeoutError('Neutral/hold ACK timeout16s')
            if self._dwell(drive_hold_confirmed and max(abs(v) for v in measured_steer)<=math.radians(2.5),timestamp):
                interior_handoff=self.advance_after_neutral
                if interior_handoff:
                    # A completed interior leg advances only after a measured
                    # stop/neutral ACK; never by restoring its nominal state.
                    self.leg_index+=1;self._make_tracker();self.advance_after_neutral=False
                action=self.leg['view'].actions[0];self.mode=action['mode']
                # Native plans all start from the SAME nominal zero steering.
                # Do not pre-align each method's first curvature and silently
                # replace the common initial condition. Only P propagates its
                # finite-rate initial transition; baseline idealizations stay
                # explicitly different in their planning model, not the plant.
                self.curvature=self.leg['direction']*action['curvature_1_m'] if interior_handoff else 0.
                self._stage('alignment',timestamp,pose)
        if self.stage=='alignment':
            self.speed=0.;hold=True
            if measured_speed>.005:raise RuntimeError('Alignment requires measured stop')
            if timestamp-self.stage_started>16.:raise TimeoutError('Original steering alignment timeout16s')
            target=tuple(float(v) for v in steering_angles(self.curvature,self.mode))
            if self._dwell(drive_hold_confirmed and max(abs(a-b) for a,b in zip(measured_steer,target))<=math.radians(2.5),timestamp):
                hold=False;self._stage('release_hold',timestamp,pose)
        if self.stage=='release_hold':
            self.speed=0.;hold=False
            if timestamp-self.stage_started>16.:raise TimeoutError('Drive hold release ACK timeout16s')
            if not drive_hold_confirmed:
                self._stage('tracking',timestamp,pose)
        if self.stage=='terminal':
            self.phase=self.terminal.phase
            if timestamp-self.terminal.terminal_started>self.terminal.config['terminal_timeout_s']:
                raise TimeoutError('Original90s terminal budget includes hold ACK waiting')
            waiting=(self.terminal.drive_hold_requested and not drive_hold_confirmed or
                drive_hold_confirmed and not self.terminal.drive_hold_requested)
            if waiting:
                if drive_hold_confirmed and abs(self.terminal.speed)>1e-7:
                    raise RuntimeError('Unexpected drive hold ACK during commanded terminal motion')
                self.terminal.previous_time=timestamp;self.speed=0.;hold=self.terminal.drive_hold_requested
            else:
                info=self.terminal.step(timestamp,pose,measured_speed,measured_steer)
                self.speed=info['speed_m_s'];self.mode=info['mode'];self.curvature=info['curvature_1_m']
                hold=info['drive_hold_requested'];self.phase=self.terminal.phase;self.pulse_count=self.terminal.pulse_count
                if info['hold_ready']:
                    self.hold_entry_candidate=self.terminal.hold_entry_candidate
                    self._stage('final_neutral',timestamp,pose)
        if self.stage=='final_neutral':
            self.speed=0.;self.curvature=0.;hold=True;self.phase='terminal_braking'
            error=original.relative_pose(self.goal,pose)
            if measured_speed>.005 or math.hypot(error[0],error[1])>.030 or abs(error[2])>math.radians(2.):
                raise RuntimeError('First terminal capture left original30mm/2deg/5mm_s before fixed hold; no retry')
            if timestamp-self.stage_started>16.:raise TimeoutError('Final neutral/hold ACK timeout16s')
            ready=self._dwell(drive_hold_confirmed and max(abs(v) for v in measured_steer)<=math.radians(2.5),timestamp)
        if self.stage not in ('terminal','final_neutral'):self.phase='path_tracking'
        index,reference=self.reference.at(self.reference.progress);error=original.relative_pose(self.goal,pose)
        return dict(info,phase=self.phase,terminal_stage=self.stage,speed_m_s=self.speed,mode=self.mode,
            curvature_1_m=self.curvature,action_index=index,progress_m=self.reference.progress,
            path_error_m=math.hypot(pose[0]-reference[0],pose[1]-reference[1]),
            path_heading_error_deg=math.degrees(wrap(reference[2]-pose[2])),
            goal_position_error_m=math.hypot(error[0],error[1]),goal_heading_error_deg=math.degrees(error[2]),
            hold_ready=ready,drive_hold_requested=hold,pulse_count=self.pulse_count,
            leg_index=self.leg_index,direction=self.leg['direction'],physical_hold_ACK=drive_hold_confirmed)


def mission_execution_contract(*,simulation_budget_s):
    if type(simulation_budget_s) not in (int,float) or simulation_budget_s not in (180.,240.,300.,600.):
        raise ValueError('Explicit prospective common300/600s MODEL-only duration required')
    return dict(version='shared_bidirectional_v3_measured_handoff_PROTOTYPE_v2',
        role='NEW_COMMON_OFFLINE_EXECUTOR_NOT_AN_ADOPTED_LIVE_PROTOCOL',simulation_limit_s=float(simulation_budget_s),
        method_identifier_selects_control_gains_or_policy=False,
        tracking_binding=_mission_tracking_implementation().mission_binding,
        reverse_tracking_is_virtual_heading_pi_and_curvature_sign_only=True,
        initial_alignment_is_common_nominal_zero_NOT_each_methods_first_curvature=True,
        physical_states_joints_velocities_casters_never_reset_or_invented=True,
        speed_cap_m_s=.050,acceleration_m_s2=.020,steering_slew_deg_s=8.,
        original_tracking_guards_m_deg=[.080,10.],original_terminal_m_deg_speed=[.030,2.,.005],
        cusp_stopping_timeout_s=8.,neutral_alignment_or_release_timeout_s=16.,alignment_dwell_s=.25,
        hold_request_requires_zero_command_AND_measured_speed_under_5mm_s=True,
        next_motion_requires_hold_ACK_then_neutral_then_alignment_then_released_ACK=True,
        original_v3_terminal_pulses_and90s_budget_unchanged=True,
        first_terminal_capture_additional_neutral_gate_does_NOT_allow_retry_or_better_hold_window=True,
        ONE_fixed5s_hold_and_fixed_last3s_remain_external_runtime_checks=True,
        reference_endpoint_interpolation_is_NOT_exact_substep_or_continuous_collision_proof=True,
        frozen_plan_model_parameters_gains_material_and_physical_safety_limits_unchanged=True,
        different_executor_versions_must_NOT_be_pooled_as_same_comparison=True,
        model_timing_and_ACK_assumptions_NOT_physical_stop_or_hold_force_guarantees=True,
        runtime_adapter_prepared=False,actual_motion_authorized=False,use_for_final_statistics=False)


def mission_model_case_binding(geometry,plan,*,simulation_budget_s):
    """Bind the ORIGINAL closed-loop NE MODEL case, not its plant/hold force.

    ACK is deliberately a one-tick software echo of the previous request.
    It tests controller sequencing only; the old zero-target NE plant still
    integrates every velocity/pose and contains NO captured-wheel hold force.
    A live adapter must replace this assumption with actual hold/release ACKs.
    """
    import copy
    from types import FunctionType,SimpleNamespace
    import screen_hospital_h4_common_executor as original
    from hospital_mission_geometry import disjoint_support_rectangles
    from hospital_h4_subfloor_compact_path_launch import normalize_runtime_evidence
    validate_mission_plan(plan,geometry);contract=mission_execution_contract(simulation_budget_s=simulation_budget_s)
    design=copy.deepcopy(geometry);covers=design['same_three_authored_support_rectangles']
    if len(covers)!=3:raise ValueError('All three original authored H6 support rectangles required')
    disjoint_support_rectangles(covers)
    instances=[]
    class Controller(MissionTrackingController):
        def __init__(self,candidate):
            if candidate!=plan:raise ValueError('The same frozen native plan is required')
            super().__init__(candidate,design,simulation_budget_s=simulation_budget_s)
            self.software_previous_request_ACK=candidate.get('measured_replan_start') is not None
            self.ack_events=[];instances.append(self)
        def step(self,timestamp,pose,measured_speed,measured_steer):
            ack=self.software_previous_request_ACK;previous=self.stage
            info=super().step(timestamp,pose,measured_speed,measured_steer,drive_hold_confirmed=ack)
            self.software_previous_request_ACK=info['drive_hold_requested']
            if ack!=info['drive_hold_requested'] or self.stage!=previous:
                self.ack_events.append(dict(time_s=timestamp,assumed_previous_request_ACK=ack,
                    hold_requested=info['drive_hold_requested'],stage=self.stage,
                    measured_model_speed_m_s=measured_speed,command_speed_m_s=info['speed_m_s']))
            return info
    fields={k:v for k,v in vars(original.execution).items() if not k.startswith('__')}
    fields.update(H4TrackingController=Controller,TrackingController=Controller,
        GUARDS=dict(original.execution.GUARDS,maximum_simulation_duration_s=float(simulation_budget_s)))
    execution=SimpleNamespace(**fields)
    def support_covers(candidate):
        if candidate!=design:raise ValueError('Model support geometry changed')
        return copy.deepcopy(covers)
    cohort=SimpleNamespace(support_covers=support_covers,support_area_missing=original.bundle.cohort.support_area_missing)
    function=original.case
    bound=FunctionType(function.__code__,dict(function.__globals__,execution=execution,
        bundle=SimpleNamespace(cohort=cohort)),function.__name__,function.__defaults__)
    bound.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    def run(candidate,bounds,dt,occupied,origin,resolution,support):
        if bounds!=plan['rigid_body_parameter_bounds'] or support!=design:
            raise ValueError('Original same-plant bounds/support required')
        if instances:raise ValueError('One MODEL integration per isolated case binding; no automatic retry')
        report,rows=bound(candidate,bounds,dt,occupied,origin,resolution,support)
        controller=instances[0]
        report.update(new_common_executor_contract=contract,
            control_leg_transitions=normalize_runtime_evidence(controller.events),
            software_ACK_events=normalize_runtime_evidence(controller.ack_events),
            assumed_ACK_is_previous_request_echo_NOT_physical_drive_hold=True,
            original_NE_plant_integration_and_collision_support_checks_code_unchanged=True,
            nominal_model_is_NOT_PhysX_contact_caster_hold_force_or_drive_admission=True,
            full_goal_or_H6_actual_execution_complete=False)
        return report,rows
    run.original_case_code=bound.__code__
    run.plant_code_and_guards_unchanged=bound.__code__ is original.case.__code__
    return run,execution,design


def export_mission_executor_model_screen(planning_review_path,*,simulation_budget_s,output):
    """ONE predeclared all-four/two-resolution MODEL screen; append only."""
    import csv
    import hashlib
    import inspect
    import json
    import sys
    import zipfile
    from pathlib import Path
    from hospital_runtime_evidence_io import write_exclusive
    from hospital_h4_subfloor_compact_path_launch import normalize_runtime_evidence
    import screen_hospital_h4_common_executor as original
    if any(name in sys.modules for name in ('isaacsim','omni.kit.app','omni.physx')):
        raise RuntimeError('Offline-only model screen; no Kit/timeline entry')
    planning_review_path=Path(planning_review_path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Existing complete/partial screen preserved; no retry or overwrite')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    review=json.loads(planning_review_path.read_bytes());contract_path=Path(review['protocol'])
    native=json.loads(contract_path.read_bytes());geometry_path=Path(native['common_geometry'])
    geometry=json.loads(geometry_path.read_bytes())
    if ([r['method'] for r in review['methods']]!=list(original.execution.METHODS)
            or not all(r['accepted_offline_start_goal_plan'] is True for r in review['methods'])
            or sha(geometry_path)!=native['common_geometry_sha256']):
        raise ValueError('All four independently accepted same-task model plans required')
    inputs={str(planning_review_path):sha(planning_review_path)}
    for collection in (review['input_sha256'],review['output_sha256']):
        for p,digest in collection.items():
            if sha(p)!=digest:raise ValueError('Frozen model input changed: '+p)
            inputs[p]=digest
    inputs[str(geometry_path)]=sha(geometry_path)
    plans=[json.loads(Path(r['plan_file']).read_bytes()) for r in review['methods']]
    for plan in plans:
        validate_mission_plan(plan,geometry)
        if plan['rigid_body_parameter_bounds']!=native['rigid_body_parameter_bounds']:
            raise ValueError('All four cases must use the same original plant parameters')
    definition=mission_execution_contract(simulation_budget_s=simulation_budget_s)
    definition.update(planning_review=str(planning_review_path),same_common_geometry=str(geometry_path),
        cases=[dict(method=p['method'],integration_dt_s=dt) for p in plans for dt in original.TIMESTEPS_S],
        hold_ACK='ONE_TICK_PREVIOUS_REQUEST_ECHO_SOFTWARE_SEQUENCE_ASSUMPTION_NOT_PHYSICAL_FORCE',
        resolution_cases_NOT_independent_experimental_repetitions=True,
        no_current_live_static_release_or_drive_authority=True,new_PhysX_trial_count=0)
    root=_ijcas_workspace()
    sources=[Path(__file__).resolve(),_ijcas_path(root, 'src/patient_initial_stability/test/test_hospital_timed_path_execution.py'),
        Path(original.__file__).resolve(),Path(inspect.getsourcefile(original.bundle.cohort.support_area_missing)).resolve()]
    sources += [_ijcas_path(root, 'studies/IJCAS/scripts', name) for name in (
        'hospital_h4_path_execution.py','hospital_h2_path_execution.py','hospital_path_execution.py',
        'hospital_path_execution_v2.py','hospital_path_execution_v3.py','hospital_timed_path_execution.py','hospital_geometry_tools.py',
        'hospital_mission_geometry.py','hospital_runtime_evidence_io.py','hospital_h4_subfloor_compact_path_launch.py')]
    sources += [_ijcas_path(root, 'src/patient_initial_stability/patient_initial_stability', name) for name in ('rigid_body_edge.py','steering_lattice.py')]
    sources=list(dict.fromkeys(sources));source_hashes={str(p.relative_to(root)):sha(p) for p in sources}
    output.mkdir(parents=True)
    write=lambda p,d:write_exclusive(p,normalize_runtime_evidence(d))
    write(output/'prospective_MODEL_definition.json',definition)
    archive=output/'executed_model_sources.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,str(p.relative_to(root)))
    grid_path=planning_review_path.parent/'planning_occupancy.npz'
    with np.load(grid_path) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if occupied.dtype!=bool or resolution!=.010:raise ValueError('Same10mm native frozen raster required')
    cases=[];files=[output/'prospective_MODEL_definition.json',archive]
    for plan in plans:
        for dt in original.TIMESTEPS_S:
            print('[H6 EXECUTOR MODEL] START',plan['method'],dt,'no Isaac or physical hold-force model',flush=True)
            run,_,design=mission_model_case_binding(geometry,plan,simulation_budget_s=simulation_budget_s)
            report,rows=run(plan,native['rigid_body_parameter_bounds'],dt,occupied,origin,resolution,design)
            trace=output/(plan['method']+'_'+str(round(dt*1000))+'ms.csv')
            with trace.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            report.update(trace=str(trace),trace_sha256=sha(trace));cases.append(report);files.append(trace)
            write(output/(plan['method']+'_'+str(round(dt*1000))+'ms.json'),report)
            files.append(output/(plan['method']+'_'+str(round(dt*1000))+'ms.json'))
            print('[H6 EXECUTOR MODEL] END',plan['method'],dt,report['model_status'],report['error'],flush=True)
    result=dict(status='PROSPECTIVE_COMMON_BIDIRECTIONAL_MODEL_DIAGNOSTIC_NOT_PHYSICAL_ADMISSION',
        cases=cases,case_count=len(cases),MODEL_PASS_count=sum(r['model_status']=='MODEL_SCREEN_PASS' for r in cases),
        model_and_ACK_limitations=definition,original_case_plant_function_code_unchanged=True,
        source_members_sha256=source_hashes,input_sha256=inputs,output_sha256={str(p):sha(p) for p in files},
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    write(output/'common_bidirectional_MODEL_screen.json',result)
    print('[H6 EXECUTOR MODEL] SAVED',output/'common_bidirectional_MODEL_screen.json',flush=True)
    return output/'common_bidirectional_MODEL_screen.json'


def retained_tensor_encoder(backend_utils,device):
    """Use the installed SDK backend's conversion, not CPU arrays on a GPU view."""
    if not callable(getattr(backend_utils,'convert',None)):
        raise ValueError('Existing SDK backend conversion required')
    def encode(value,kind):
        if kind not in ('float32','int32'):raise ValueError('Only explicit target/index tensor types supported')
        array=np.ascontiguousarray(value,dtype=np.float32 if kind=='float32' else np.int32)
        return backend_utils.convert(array,device=device,dtype=kind,indexed=False)
    return encode


class RetainedTensorDriveAdapter:
    """Scoped existing PhysX tensor view ONLY; no wrapper/scene construction.

    Confirmation is BACKEND target/gain readback at a LATER physics stamp,
    not a requested flag, motor-force proof or standstill substitute. SDK
    tensor setters write full arrays; all noncontrolled entries are preserved
    exactly. No instantaneous position/velocity, max-force, USD, camera,
    timeline or passive-caster state setter is ever used.

    The caller MUST supply a singleton view created for the admitted actor,
    same-stage/actor guard, physics-step clock and actual Pause predicate.
    Default construction is read-only. A model request echo is not supported.
    """
    STEERING=('base_link_to_front_active_steering','base_link_to_rear_active_steering')
    DRIVES=('front_active_steering_to_drive_wheel','rear_active_steering_to_drive_wheel')
    READERS=dict(kp='get_dof_stiffnesses',kd='get_dof_dampings',limits='get_dof_max_forces',
        positions='get_dof_position_targets',velocities='get_dof_velocity_targets')

    def __init__(self,view,expected_motor_baseline,*,encode,check_context,paused_check,
            physics_stamp,authorized=False):
        if not all(callable(v) for v in (encode,check_context,paused_check)):
            raise ValueError('Explicit backend/context/observed-Pause bindings required')
        self.view=view;self.encode=encode;self.check_context=check_context;self.paused_check=paused_check
        self.authorized=authorized is True;self.closed=False;self.dirty=False;self.pending_stamp=None
        self.desired_hold=False;self.confirmed_hold=False;self.capture_count=0;self.events=[]
        self.last_stamp=float(physics_stamp)
        if not math.isfinite(self.last_stamp):raise ValueError('Finite actual physics stamp required')
        self._scope()
        self.names=tuple(view.shared_metatype.dof_names);links=tuple(view.shared_metatype.link_names)
        if (view.count!=1 or view.max_dofs!=12 or view.max_links!=13 or len(self.names)!=12
                or len(set(self.names))!=12 or len(links)!=13 or len(set(links))!=13
                or self.names!=tuple(expected_motor_baseline['joint_names'])):
            raise ValueError('One own12joint/13link articulation and exact source joint ordering required')
        indices={name:i for i,name in enumerate(self.names)}
        if any(name not in indices for name in (*self.STEERING,*self.DRIVES)):
            raise ValueError('Complete original active steering/drive mapping required')
        self.steering=np.asarray([indices[n] for n in self.STEERING],dtype=int)
        self.drive=np.asarray([indices[n] for n in self.DRIVES],dtype=int)
        self.active=np.r_[self.steering,self.drive]
        self.passive=np.asarray([i for i in range(12) if i not in self.active],dtype=int)
        self.original=self._motor()
        for key,source in (('kp','kp'),('kd','kd'),('limits','max_efforts')):
            expected=np.asarray(expected_motor_baseline[source],np.float32)
            if expected.shape!=(12,) or not np.array_equal(self.original[key],expected):
                raise ValueError('Actual retained motor baseline differs from own static: '+key)
        if (np.any(self.original['kp']<0.) or np.any(self.original['kd']<0.)
                or np.any(self.original['kp'][self.drive]!=0.) or np.any(self.original['velocities'][self.active]!=0.)
                or np.any(self.original['kp'][self.steering]<=0.)
                or not np.isfinite(self.original['limits'][self.active]).all()
                or np.any(self.original['limits'][self.active]<=0.)):
            raise ValueError('Original velocity-drive, zero active targets and finite actuator limits required')
        self.expected={k:v.copy() for k,v in self.original.items()}
        self.captured_targets=None;self.last_command_stamp=self.last_stamp

    def _scope(self):
        context=self.check_context()
        if (context is not None and not bool(context)) or not bool(self.view.check()):
            raise RuntimeError('Retained stage/actor/physics handle changed; no recovery or USD fallback')

    def _array(self,reader,*,allow_infinity=False):
        self._scope();value=getattr(self.view,reader)()
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        elif hasattr(value,'numpy'):value=value.numpy()
        value=np.asarray(value,dtype=np.float32)
        if (value.shape!=(1,12) or np.isnan(value).any() or np.isneginf(value).any()
                or not allow_infinity and not np.isfinite(value).all()):
            raise ValueError('Complete finite single-articulation tensor readback required: '+reader)
        return value[0].copy()

    def _motor(self):
        return {k:self._array(reader,allow_infinity=k=='limits') for k,reader in self.READERS.items()}

    def _write_permit(self):
        if not self.authorized or self.closed:raise RuntimeError('Separate live authority and open adapter required')
        self._scope()

    def _set(self,setter,value,*,paused=False):
        if setter not in ('set_dof_stiffnesses','set_dof_dampings','set_dof_position_targets','set_dof_velocity_targets'):
            raise ValueError('Only PD gains and desired targets may be written; never physical state/force limits')
        self._write_permit()
        if paused and self.paused_check() is not True:
            raise RuntimeError('Actual retained PAUSED state required for baseline restoration')
        value=np.asarray(value,np.float32)
        if value.shape!=(12,) or not np.isfinite(value).all():raise ValueError('Finite complete actuator targets/gains required')
        data=self.encode(value.reshape(1,12),'float32');index=self.encode(np.asarray([0]),'int32')
        self.dirty=True;getattr(self.view,setter)(data,index)

    def _verify(self,current):
        for key in self.READERS:
            if not np.array_equal(current[key],self.expected[key]):
                raise RuntimeError('Actual SDK gain/target/limit readback differs: '+key)
        self._scope()

    def confirmation(self,physics_stamp):
        if self.closed:raise RuntimeError('Closed actuator adapter cannot confirm a new motion')
        stamp=float(physics_stamp)
        if not math.isfinite(stamp) or stamp<self.last_stamp:raise ValueError('Monotonic actual physics samples required')
        self.last_stamp=stamp;current=self._motor()
        self._verify(current)
        if self.pending_stamp is not None:
            if stamp<=self.pending_stamp:return None
            self.confirmed_hold=self.desired_hold;self.pending_stamp=None
            self.events.append(dict(event='BACKEND_HOLD_CONFIRMED' if self.confirmed_hold else 'BACKEND_RELEASE_CONFIRMED',physics_stamp=stamp))
        return self.confirmed_hold

    def measured_steering(self):
        result=tuple(float(v) for v in self._array('get_dof_positions')[self.steering])
        if max(abs(v) for v in result)>math.radians(60.):raise ValueError('Original actual steering guard60deg')
        return result

    def request_hold(self,enabled,physics_stamp,*,command_speed_m_s,measured_speed_m_s):
        self._write_permit()
        if (type(enabled) is not bool or not math.isfinite(command_speed_m_s)
                or not math.isfinite(measured_speed_m_s) or measured_speed_m_s<0.):
            raise ValueError('Explicit requested hold and measured finite speed required')
        ack=self.confirmation(physics_stamp)
        if enabled==self.desired_hold:return
        if ack is None:raise RuntimeError('No reversal of a pending unconfirmed hold transaction')
        if abs(command_speed_m_s)>1e-7 or measured_speed_m_s>.005:
            raise RuntimeError('Hold/release transition requires zero command AND measured stop')
        self.desired_hold=enabled;self.pending_stamp=float(physics_stamp)
        self.expected['velocities'][self.drive]=0.
        if enabled:
            from hospital_drive_hold import drive_hold_gain_arrays
            self.captured_targets=self._array('get_dof_positions')[self.drive].copy();self.capture_count+=1
            self.expected['positions'][self.drive]=self.captured_targets
            kp,_=drive_hold_gain_arrays(self.original['kp'],self.original['kd'],self.drive)
            self.expected['kp']=np.asarray(kp,np.float32)
            # Zero velocity and captured targets BEFORE enabling stiffness.
            self._set('set_dof_velocity_targets',self.expected['velocities'])
            self._set('set_dof_position_targets',self.expected['positions'])
            self._set('set_dof_stiffnesses',self.expected['kp'])
        else:
            # Restore velocity-only gain BEFORE clearing the unused hold target.
            self.expected['kp']=self.original['kp'].copy()
            self._set('set_dof_stiffnesses',self.expected['kp'])
            self.expected['positions'][self.drive]=self.original['positions'][self.drive]
            self._set('set_dof_position_targets',self.expected['positions'])
            self._set('set_dof_velocity_targets',self.expected['velocities'])
        self.events.append(dict(event='HOLD_REQUESTED' if enabled else 'RELEASE_REQUESTED',
            physics_stamp=float(physics_stamp),capture_count=self.capture_count))
        self._verify(self._motor())

    def command(self,steering_targets,wheel_velocities,physics_stamp):
        self._write_permit();stamp=float(physics_stamp);ack=self.confirmation(stamp)
        steer=np.asarray(steering_targets,float);wheels=np.asarray(wheel_velocities,float)
        dt=stamp-self.last_command_stamp
        if (steer.shape!=(2,) or wheels.shape!=(2,) or not np.isfinite(steer).all() or not np.isfinite(wheels).all()
                or not 0.<=dt<=.25 or np.max(abs(steer))>math.radians(60.)
                or np.max(abs(steer-self.expected['positions'][self.steering]))>math.radians(8.)*dt+1e-7):
            raise ValueError('Original finite command/steering-slew/control-gap bounds required')
        actual=self.measured_steering()
        equivalent=wheels*.074*np.cos(actual)
        if (max(abs(v) for v in equivalent)>.0500001 or abs(equivalent[0]-equivalent[1])>1e-6
                or (ack is None or ack is True) and np.max(abs(wheels))>1e-7):
            raise RuntimeError('No movement while held/pending; bounded consistent original drive targets required')
        self.expected['positions'][self.steering]=steer.astype(np.float32)
        self.expected['velocities'][self.drive]=wheels.astype(np.float32)
        self._set('set_dof_position_targets',self.expected['positions'])
        self._set('set_dof_velocity_targets',self.expected['velocities'])
        self.last_command_stamp=stamp;self._verify(self._motor())

    def emergency_zero(self):
        """Best-effort drive velocity target zero; NEVER an instantaneous stop."""
        self._write_permit();velocity=self._array('get_dof_velocity_targets');velocity[self.drive]=0.
        self._set('set_dof_velocity_targets',velocity);self.expected['velocities']=velocity

    def restore_original(self):
        """After ACTUAL Pause: attempt each independent restoration, no retry.

        Preserve current passive entries even if an unexpected external edit
        occurred; do not silently repair/assign passive targets or force limits.
        Such differences make the final complete-baseline check FAIL.
        """
        self._write_permit()
        if self.paused_check() is not True:raise RuntimeError('Confirmed actual Pause required; no restoration during motion')
        errors=[]
        for key,setter in (('velocities','set_dof_velocity_targets'),('kp','set_dof_stiffnesses'),
                ('kd','set_dof_dampings'),('positions','set_dof_position_targets')):
            try:
                current=self._array(self.READERS[key]);current[self.active]=self.original[key][self.active]
                self._set(setter,current,paused=True)
            except Exception as failure:errors.append(key+': '+type(failure).__name__+': '+str(failure))
        checks={}
        try:
            current=self._motor()
            checks={key:bool(np.array_equal(current[key],value)) for key,value in self.original.items()}
        except Exception as failure:errors.append('readback: '+type(failure).__name__+': '+str(failure))
        self.closed=True;self.pending_stamp=None
        self.restoration_report=dict(restored=not errors and bool(checks) and all(checks.values()),
            checks=checks,errors=errors,physical_positions_velocities_and_passive_states_never_set=True,
            max_effort_limits_never_written=True,source_USD_never_written=True)
        return self.restoration_report


class MissionSDKActuationBridge:
    """Controller-to-retained-view bridge, NOT a complete live route runner.

    The outer admitted workflow still owns exact live continuity, contacts,
    raster/support/tilt/resource guards, clock callbacks, fixed hold, confirmed
    Pause and durable closeout. A supplied guard is mandatory before writes.
    """
    def __init__(self,controller,actuator,*,guard):
        if not callable(guard):raise ValueError('The admitted workflow safety guard is mandatory')
        if not actuator.authorized:raise RuntimeError('Separate explicit actuation authority required for SDK bridge')
        self.controller=controller;self.actuator=actuator;self.guard=guard
        self.last_stamp=actuator.last_command_stamp
        self.steering_command=tuple(float(v) for v in actuator.original['positions'][actuator.steering])
        self.last_info=None;self.failed=False

    def step(self,physics_stamp,pose,measured_speed):
        if self.failed:raise RuntimeError('SDK bridge already failed; no automatic retry')
        try:return self._step(physics_stamp,pose,measured_speed)
        except Exception:
            self.failed=True
            raise

    def _step(self,physics_stamp,pose,measured_speed):
        stamp=float(physics_stamp);ack=self.actuator.confirmation(stamp)
        if ack is None or self.last_info is not None and stamp==self.last_stamp:
            if self.last_info is None or ack is None and abs(self.last_info['speed_m_s'])>1e-7:
                raise RuntimeError('Pending actuator transaction cannot conceal motion')
            self.guard(pose,measured_speed,self.last_info)
            return dict(self.last_info,SDK_hold_confirmation=ack,SDK_physics_sample_not_advanced=True)
        actual=self.actuator.measured_steering()
        if self.controller.stage=='release_hold' and ack is False:
            target=tuple(float(v) for v in steering_angles(self.controller.curvature,self.controller.mode))
            if measured_speed>.005 or max(abs(a-b) for a,b in zip(actual,target))>math.radians(2.5):
                raise RuntimeError('Measured stop/alignment lost before actual release ACK; no resume')
        info=self.controller.step(stamp,pose,measured_speed,actual,drive_hold_confirmed=ack)
        self.guard(pose,measured_speed,info)
        self.actuator.request_hold(info['drive_hold_requested'],stamp,
            command_speed_m_s=info['speed_m_s'],measured_speed_m_s=measured_speed)
        from hospital_path_execution import direct_joint_targets
        steer,wheels=direct_joint_targets(info['speed_m_s'],info['mode'],info['curvature_1_m'],
            self.steering_command,stamp-self.last_stamp,actual)
        self.actuator.command(steer,wheels,stamp)
        self.steering_command=steer;self.last_stamp=stamp
        self.last_info=dict(info,SDK_hold_confirmation=self.actuator.confirmation(stamp),
            SDK_controller_input_hold_confirmation=ack,SDK_physics_sample_not_advanced=False)
        return dict(self.last_info)


def inspect_installed_tensor_sdk(api_path,converter_paths):
    """Read installed primary SDK source/ABI ONLY, never import/start Kit."""
    import ast
    import hashlib
    import warnings
    from pathlib import Path
    parse_warnings=[]
    def parse(path):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always');tree=ast.parse(path.read_text())
        parse_warnings.extend(dict(path=str(path),category=w.category.__name__,message=str(w.message)) for w in observed)
        return tree
    api_path=Path(api_path);tree=parse(api_path)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArticulationView')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    getters=(*RetainedTensorDriveAdapter.READERS.values(),'get_dof_positions','check')
    setters=('set_dof_stiffnesses','set_dof_dampings','set_dof_position_targets','set_dof_velocity_targets')
    for name in getters:
        if [a.arg for a in methods[name].args.args]!=['self']:raise ValueError('Unexpected installed getter ABI: '+name)
    for name in setters:
        method=methods[name]
        if [a.arg for a in method.args.args]!=['self','data','indices']:
            raise ValueError('Unexpected installed tensor setter ABI: '+name)
        called={n.func.attr for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        if not {name,'as_contiguous_float32','as_contiguous_uint32'}<=called:
            raise ValueError('Expected direct typed backend setter, no high-level USD fallback: '+name)
    sources={str(api_path):hashlib.sha256(api_path.read_bytes()).hexdigest()}
    for path in converter_paths:
        path=Path(path);module=parse(path)
        function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='convert')
        if [a.arg for a in function.args.args]!=['data','device','dtype','indexed']:
            raise ValueError('Unexpected installed SDK backend conversion ABI')
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(status='INSTALLED_SDK_SOURCE_ABI_INSPECTED_NOT_LIVE_BACKEND_VALIDATION',
        getters=list(getters),setters=list(setters),input_sha256=sources,
        source_parse_warnings_recorded_not_runtime_API_errors=parse_warnings,
        no_Kit_import_or_physics_view_or_control_call=True,actual_motion_authorized=False)

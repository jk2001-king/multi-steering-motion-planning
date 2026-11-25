"""Pure measured-encoder/AMCL frontend for the ORIGINAL platform controller.

No SDK, ROS publisher, actor, timeline or hardware operation exists here.
The actuation-shaped endpoint below STORES PROPOSALS ONLY. Its hold ACK and
joint measurements come verbatim from the real native adapter, never from
the requested hold or a simulated/fabricated backend confirmation.
"""

import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import hospital_sensor_runtime as native
import hospital_sensor_messages as messages
import hospital_mission_sequence as sequence


VERSION='MEASURED_AMCL_ENCODER_ORIGINAL_P_PLATFORM_PROPOSAL_v1'
TASK_LIMITS={'H5_obstacle_bypass_return':(240.,900.),
    'H6_integrated_transfer_mission':(600.,3000.)}
H6_SENSOR_WALL_REVIEW=dict(approval_source='USER_QUESTION_REPLY',
    question_item_id='["request_user_input_async","call_10e7e20f9340442dad04ee6df19c5ba7",0]',
    answer_ko='H6 센서 검증 실제시간 3,000초·1회 승인',
    sensor_orchestration_wall_s=3000.,simulation_budget_s=600.,
    original_main_comparison1500s_NOT_changed=True)
SENSOR_CAPTURE_POLICY=dict(version='SENSOR_ONLY_STRICT15MM_STATIONARY_CAPTURE_v1',
    stationary_hold_entry_position_m=.015,original_world_scoring_position_m=.030,
    unchanged_heading_deg=2.,unchanged_stop_m_s=.005,unchanged_fixed_hold_s=5.,
    goal_or_estimated_pose_shifted=False,localization_error_bound_or_guarantee_claimed=False,
    original_main_comparison_controller_NOT_changed=True)


def task_limits(scenario):
    if scenario not in TASK_LIMITS:raise ValueError('Only the declared H5/H6 P integration cases are supported')
    simulation,wall=TASK_LIMITS[scenario]
    return dict(simulation_budget_s=simulation,maximum_wall_time_s=wall,
        maximum_sensor_frames=int(simulation*5)+10,maximum_control_records=int(simulation*20)+10)


def review_amcl_covariance(fix):
    """Keep RAW AMCL values; never use them as a calibrated safety bound.

    Humble pf.c forms covariance as E[xx]-E[x]E[x]. Particle collapse at
    nonzero map coordinates can produce tiny negative cancellation residuals.
    This diagnostic allowance is for numerical consistency ONLY: no PSD
    projection, clipping, pose correction, or physical tolerance change.
    https://github.com/ros-navigation/navigation2/blob/humble/nav2_amcl/src/pf/pf.c
    """
    covariance=np.array(messages.frames.vector(fix['covariance'],36,'actual AMCL covariance')).reshape(6,6)
    if not np.allclose(covariance,covariance.T,rtol=0.,atol=1e-12):
        raise ValueError('Actual AMCL covariance is not symmetric')
    epsilon=float(np.finfo(float).eps);operations=4*10000+16
    gamma=operations*epsilon/(1-operations*epsilon)
    scale=max(1.,max(abs(float(v)) for v in fix['xyyaw'][:2])**2,float(np.max(abs(covariance))))
    allowance=8*gamma*scale;minimum=float(np.linalg.eigvalsh(covariance).min())
    if minimum < -allowance:raise ValueError('AMCL covariance negative beyond cancellation allowance: '+str(minimum))
    return dict(actual_minimum_eigenvalue=minimum,cancellation_diagnostic_allowance=allowance,
        raw_covariance_modified=False,covariance_NOT_used_as_calibrated_control_or_safety_bound=True,
        tiny_negative_roundoff_present=minimum<0.,physical_goal_stop_and_collision_limits_NOT_changed=True)


def checked_definition(value):
    value=sequence.retained.evidence.json_value(value)
    limits=task_limits(value.get('scenario'))
    if (value.get('version')!=VERSION
            or value.get('method')!='P_dynamics_full_envelope' or value.get('sensor_speed_window_s')!=.2
            or value.get('control_minimum_period_s')!=.05 or value.get('maximum_control_gap_s')!=.25
            or any(value.get(k)!=limits[k] for k in ('simulation_budget_s','maximum_wall_time_s'))
            or value.get('maximum_native_route_trials')!=1 or value.get('automatic_retry') is not False
            or value.get('ground_truth_control_pose_or_velocity_inputs') is not False):
        raise ValueError('Exact bounded H5/H6 P measured-input proposal definition required')
    policy=value.get('sensor_stationary_capture_policy')
    if policy is not None and policy!=SENSOR_CAPTURE_POLICY:
        raise ValueError('Only the separately declared stricter15mm SENSOR control configuration is supported')
    if value['scenario']=='H6_integrated_transfer_mission' and value.get('sensor_wall_budget_review')!=H6_SENSOR_WALL_REVIEW:
        raise ValueError('Separate actual H6 sensor3000s approval required; original comparison remains1500s')
    for key in ('controller_definition','frozen_plan','geometry','encoder_geometry','encoder_wrap_contract'):
        file=Path(value[key])
        if not file.is_absolute() or value['input_sha256'].get(str(file))!=hashlib.sha256(file.read_bytes()).hexdigest():
            raise ValueError('Pinned original platform input required: '+key)
    names=value['native_joint_names'];targets=value['original_native_position_targets_rad']
    if (len(names)!=12 or len(set(names))!=12 or len(targets)!=12
            or any(not isinstance(name,str) for name in names)
            or not all(math.isfinite(float(x)) for x in targets)
            or not set(native.EncoderOdometrySequence.CHANNELS)<=set(names)):
        raise ValueError('Complete actual twelve-joint original baseline required')
    return value


def reviewed_controller_packet(route_path,packet,plan):
    """Pure compatibility review, not an old outcome or source-pin rewrite.

    H6 predates two saved-evidence readers and OPTIONAL H5 replan support.
    Admit a NEW controller binding only if the sole contract difference is
    the sequence-file SHA, all remaining module AST is identical, the active
    generated tracker is bytecode-identical, and removing ONE provably
    inactive H5-only branch recovers the complete original geometry class.
    """
    expected=sequence.checked_route_lifecycle_contract(packet)
    if packet['common_lifecycle_contract']==expected:
        return packet,dict(current_controller_contract_already_exact=True)
    import ast,types,zipfile
    route_path=Path(route_path).resolve();archive=route_path.parent/'route_sources.zip'
    if packet['input_sha256'].get(str(Path(sequence.__file__).resolve())) is None:
        raise ValueError('Original sequence source pin required')
    if packet['scenario']!='H6_integrated_transfer_mission' or plan.get('measured_replan_start',{}).get('goal_distance_handoff_policy') is not None:
        raise ValueError('No unreviewed controller-contract migration or H5 replan branch admission')
    sequence.lifecycle.verify_workflow_sources(archive,packet['input_sha256'])
    source_path=Path(sequence.__file__).resolve();current=source_path.read_bytes()
    with zipfile.ZipFile(archive) as saved:
        old=saved.read('studies/IJCAS/scripts/hospital_mission_sequence.py')
    if hashlib.sha256(old).hexdigest()!=packet['input_sha256'][str(source_path)]:
        raise ValueError('Exact original sequence code inside its own source archive required')
    normalized=copy.deepcopy(packet['common_lifecycle_contract'])
    normalized['original_motion_contract']['candidate_source_sha256']=hashlib.sha256(current).hexdigest()
    if normalized!=expected:raise ValueError('Controller contract differs beyond the reviewed source-file digest')
    old_tree=ast.parse(old);new_tree=ast.parse(current)
    indexed=lambda tree:{n.name:n for n in tree.body if hasattr(n,'name')}
    before=indexed(old_tree);after=indexed(new_tree)
    if set(before)!=set(after):raise ValueError('Unexpected sequence definition set change')
    changed={k for k in before if ast.dump(before[k],include_attributes=False)!=ast.dump(after[k],include_attributes=False)}
    allowed={'saved_route_boundary_review','saved_route_preparation_handoff',
        '_tracking_floor_candidate_implementation','unified_geometry_motion_class'}
    if changed!=allowed or set(before)!=set(after):raise ValueError('Unexpected sequence implementation change')
    def unchanged_module(tree):
        tree=copy.deepcopy(tree);tree.body=[n for n in tree.body if getattr(n,'name',None) not in allowed]
        return ast.dump(tree,include_attributes=False)
    if unchanged_module(old_tree)!=unchanged_module(new_tree):raise ValueError('Other source/import/global changes are not reviewed')
    namespace=dict(vars(sequence))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[before['_tracking_floor_candidate_implementation']],type_ignores=[])),
        '<verified_original_H6_tracking_builder>','exec'),namespace)
    original=namespace['_tracking_floor_candidate_implementation'](.020,physical_direction_conditioned=True)
    active=sequence._tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True)
    def signature(code):
        return (code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags,code.co_code,
            tuple(signature(x) if isinstance(x,types.CodeType) else x for x in code.co_consts),
            code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars)
    if signature(original.__code__)!=signature(active.__code__) or original.mission_binding!=active.mission_binding:
        raise ValueError('Actual generated H6 tracking code/binding changed')
    condition=ast.dump(ast.parse("plan.get('measured_replan_start',{}).get('goal_distance_handoff_policy') is not None",mode='eval').body,include_attributes=False)
    class RemoveInactive(ast.NodeTransformer):
        count=0
        def visit_If(self,node):
            if ast.dump(node.test,include_attributes=False)==condition:
                self.count+=1;return None
            return self.generic_visit(node)
    remove=RemoveInactive();reviewed=remove.visit(copy.deepcopy(after['unified_geometry_motion_class']))
    if remove.count!=1 or ast.dump(reviewed,include_attributes=False)!=ast.dump(before['unified_geometry_motion_class'],include_attributes=False):
        raise ValueError('Inactive H5-only branch removal does not recover the complete original H6 geometry class')
    bound=copy.deepcopy(packet);bound['common_lifecycle_contract']=expected
    return bound,dict(original_packet_and_result_NOT_modified=True,only_contract_source_SHA_representation_rebound=True,
        source_archive=str(archive),source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        original_sequence_sha256=hashlib.sha256(old).hexdigest(),active_sequence_sha256=hashlib.sha256(current).hexdigest(),
        exact_active_generated_tracking_code_and_binding=True,inactive_H5_branch_removed_recovers_original_AST=True,
        all_other_module_AST_exact=True,original_frozen_path_goals_gains_and_physical_limits_NOT_changed=True)


class ProposalActuatorPort:
    """Data-only implementation of the shared bridge endpoint.

    `authorized` means explicit authorization to COMPUTE a proposal here;
    this endpoint has no SDK/setter and cannot actuate anything. The real
    native SDK host separately checks motion authority and every write.
    """
    def __init__(self,names,targets):
        self.authorized=True;self.original=dict(positions=np.array(targets,dtype=np.float32))
        self.steering=[names.index(name) for name in native.EncoderOdometrySequence.CHANNELS[2:]]
        self.last_command_stamp=0.;self.observation=None;self.commanded=None;self.hold_requested=None

    def load(self,encoder,actual_hold_ack):
        if actual_hold_ack is not None and type(actual_hold_ack) is not bool:
            raise ValueError('Actual native hold ACK or explicit pending None required')
        self.observation=encoder;self.actual_hold_ack=actual_hold_ack
        self.commanded=None;self.hold_requested=None

    def confirmation(self,stamp):
        if self.observation is None:raise ValueError('No actual native readback supplied')
        return self.actual_hold_ack

    def measured_steering(self):
        return tuple(float(self.observation['positions_rad'][i]) for i in self.steering)

    def request_hold(self,enabled,stamp,*,command_speed_m_s,measured_speed_m_s):
        if type(enabled) is not bool or not all(math.isfinite(x) for x in (stamp,command_speed_m_s,measured_speed_m_s)):
            raise ValueError('Finite explicit platform hold PROPOSAL required')
        self.hold_requested=enabled

    def command(self,steering_targets,wheel_velocities,stamp):
        values=messages.frames.vector([*steering_targets,*wheel_velocities],4,'platform proposal targets')
        self.commanded=dict(steering_targets_rad=values[:2].tolist(),wheel_velocity_targets_rad_s=values[2:].tolist())
        self.last_command_stamp=stamp


class MeasuredPlatformController:
    """Original controller+geometry bridge, with CAUSAL measured inputs only."""
    def __init__(self,definition):
        self.definition=checked_definition(definition)
        self.definition_digest=native._digest(self.definition)
        route=json.loads(Path(definition['controller_definition']).read_text())
        self.plan=json.loads(Path(definition['frozen_plan']).read_text())
        self.geometry=json.loads(Path(definition['geometry']).read_text())
        if (self.plan['method']!=definition['method'] or self.plan['scenario']!=definition['scenario']
                or self.geometry['start_xyyaw']!=self.plan['start_world_xyyaw']
                or self.geometry['goal_xyyaw']!=self.plan['goal_world_xyyaw']
                or any(a['direction']!=1 for a in self.plan['selected_actions'])):
            raise ValueError('Same frozen forward H5/H6 P task/path required; no rebasing or replacement edge')
        route,self.controller_source_review=reviewed_controller_packet(definition['controller_definition'],route,self.plan)
        self.controller=sequence.route_controller(route,self.plan,self.geometry)
        self.sensor_capture_policy=definition.get('sensor_stationary_capture_policy')
        if self.sensor_capture_policy is not None:
            motion=self.controller.motion;original_raw=motion.raw
            def configure(controller):
                config=copy.deepcopy(controller.config)
                if config.get('stationary_hold_entry_position_m')!=.030:
                    raise ValueError('Exact original30mm stationary control entry required before stricter SENSOR binding')
                controller.config=dict(config,stationary_hold_entry_position_m=.015)
                return controller
            motion.tracker=configure(motion.tracker)
            def strict_raw(*args,**kwargs):return configure(original_raw(*args,**kwargs))
            motion.raw=strict_raw
        self.endpoint=ProposalActuatorPort(definition['native_joint_names'],definition['original_native_position_targets_rad'])
        self.bridge=sequence.RetainFinalTargetsBridge(self.controller,self.endpoint,guard=self._guard_proposal)
        self.encoder_geometry=json.loads(Path(definition['encoder_geometry']).read_text())['geometry']
        wrap=json.loads(Path(definition['encoder_wrap_contract']).read_text())
        self.encoder=native.EncoderOdometrySequence(actor_root=definition['actor_root'],geometry=self.encoder_geometry,
            wrap_period_rad=wrap['wrap_period_rad'],maximum_abs_rate_rad_s=wrap['maximum_abs_rate_rad_s'],
            position_roundoff_rad=wrap['position_roundoff_rad'],maximum_sample_gap_s=.25)
        self.previous_encoder=None;self.last_estimate=None;self.odom_history={};self.intervals=deque()
        self.fix=None;self.first_stamp=None;self.last_control_stamp=None;self.failed=False
        self.current_feedback=None;self.current_speed=None

    def observe_encoder(self,sample):
        if self.failed:raise ValueError('Terminal measured platform stream; no reset')
        try:
            if native._digest(self.definition)!=self.definition_digest:raise ValueError('Frozen platform definition changed')
            sample=sequence.retained.evidence.json_value(sample)
            if set(sample)&{'base_xyz','base_wxyz','world_pose','ground_truth_pose','control_pose'}:
                raise ValueError('Ground-truth pose fields are NOT encoder/controller inputs')
            if sample['joint_names']!=self.definition['native_joint_names']:
                raise ValueError('Exact declared native twelve-joint order required')
            stamp=sample['stamp_ns']
            if self.previous_encoder is not None and stamp==self.previous_encoder['stamp_ns']:
                if sample!=self.previous_encoder:raise ValueError('Different native encoder values at SAME acquisition stamp')
                return self.last_estimate
            result=self.encoder.accept(sample)
            if self.first_stamp is None:self.first_stamp=stamp
            if self.previous_encoder is not None:
                begin=self.previous_encoder['stamp_ns']
                vector=result['measured_interval_twist']['body_twist_u_v_r']
                self.intervals.append((begin,stamp,list(vector)))
            self.previous_encoder=sample;self.last_estimate=result;self.odom_history[stamp]=result['odom_xyyaw']
            while len(self.odom_history)>128:del self.odom_history[next(iter(self.odom_history))]
            while self.intervals and self.intervals[0][1]<=stamp-200000000:self.intervals.popleft()
            return result
        except BaseException:
            self.failed=True;raise

    def accept_amcl_fix(self,fix):
        fix=sequence.retained.evidence.json_value(fix)
        messages.frames.vector(fix['xyyaw'],3,'actual AMCL map pose')
        self.covariance_review=review_amcl_covariance(fix)
        stamp=messages.stamp_value(fix['stamp'])
        if stamp not in self.odom_history:raise ValueError('EXACT actual measured encoder anchor required for AMCL fix')
        if self.fix is not None and stamp<messages.stamp_value(self.fix['stamp']):raise ValueError('Regressed AMCL fix')
        if self.fix is not None and stamp==messages.stamp_value(self.fix['stamp']) and fix!=self.fix:
            raise ValueError('Contradictory AMCL fixes at the SAME acquisition stamp')
        self.fix=fix

    def feedback(self):
        # Lazy import avoids any ROS import/initialization in this pure module.
        from hospital_sensor_ros_bridge import matched_current_pose
        return matched_current_pose(self.fix,self.odom_history,self.previous_encoder['stamp_ns'],self.last_estimate['odom_xyyaw'])

    def filtered_encoder_twist(self):
        stamp=self.previous_encoder['stamp_ns'];weighted=np.zeros(3);duration=0
        for begin,end,twist in self.intervals:
            length=max(0,end-max(begin,stamp-200000000));weighted+=length*np.asarray(twist);duration+=length
        if not duration:
            sample=self.previous_encoder;names=sample['joint_names'];selected=[names.index(k) for k in native.EncoderOdometrySequence.CHANNELS]
            q=sample['positions_rad'];qd=sample['velocities_rad_s']
            weighted=np.asarray(messages.frames.centerline_encoder_twist(qd[selected[0]],qd[selected[1]],q[selected[2]],q[selected[3]],geometry=self.encoder_geometry)['body_twist_u_v_r']);duration=1
        return (weighted/duration).tolist()

    def filtered_encoder_speed(self):
        mean=self.filtered_encoder_twist()
        return float(math.hypot(mean[0],mean[1]))

    def _guard_proposal(self,pose,speed,info):
        if (self.current_feedback is None or list(pose)!=self.current_feedback['estimated_map_xyyaw']
                or speed!=self.current_speed or self.current_feedback['no_ground_truth_pose_input'] is not True
                or not math.isfinite(info['speed_m_s']) or abs(info['speed_m_s'])>.0500001):
            raise ValueError('Only actual matched AMCL/encoder input may produce a bounded P proposal')

    def propose(self,sample,*,relative_physics_time_s,actual_native_hold_ack):
        if self.failed:raise ValueError('Terminal measured P controller; no retry')
        try:
            estimate=self.observe_encoder(sample);stamp=sample['stamp_ns']
            if (self.last_control_stamp is not None and not 0<stamp-self.last_control_stamp<=250000000
                    or abs(relative_physics_time_s-(stamp-self.first_stamp)/1e9)>.001):
                raise ValueError('Actual monotone bounded source physics clock required, not a new origin')
            self.last_control_stamp=stamp;self.current_feedback=self.feedback();self.current_speed=self.filtered_encoder_speed()
            self.endpoint.load(sample,actual_native_hold_ack)
            info=self.bridge.step(relative_physics_time_s,self.current_feedback['estimated_map_xyyaw'],self.current_speed)
            generated=self.endpoint.commanded is not None
            targets=(self.endpoint.commanded if generated else dict(
                steering_targets_rad=list(self.bridge.steering_command),wheel_velocity_targets_rad_s=[0.,0.]))
            # These are only SOURCE-BRIDGE input field names, not actual SDK
            # confirmations after a write: no write exists in this component.
            source_info=copy.deepcopy(info)
            for key in list(source_info):
                if key.startswith('SDK_'):source_info.pop(key)
            twist=messages.frames.centerline_encoder_twist(*targets['wheel_velocity_targets_rad_s'],
                *self.endpoint.measured_steering(),geometry=self.encoder_geometry)['body_twist_u_v_r']
            return sequence.retained.evidence.json_value(dict(version=VERSION,stamp_ns=stamp,
                relative_physics_time_s=relative_physics_time_s,controller_info=source_info,
                source_original_bridge_input_ack=actual_native_hold_ack,requested_hold=info['drive_hold_requested'],
                actuator_write_required=generated,proposed_joint_targets=targets,proposed_body_twist_u_v_r=twist,
                measured_encoder_speed_m_s=self.current_speed,matched_sensor_feedback=self.current_feedback,
                raw_AMCL_covariance_numerical_review=self.covariance_review,
                complete=self.controller.complete,final_hold_started_s=self.controller.hold_started,
                fixed_hold_attempts=self.controller.fixed_hold_attempts,
                original_P_path_states_modes_and_controller_math_NOT_changed=self.sensor_capture_policy is None,
                sensor_stationary_capture_policy=self.sensor_capture_policy,
                original_controller_source_compatibility_review=self.controller_source_review,
                original_main_comparison_controller_NOT_changed=True,
                proposal_only_NO_SDK_actuation_or_physical_admission=True))
        except BaseException:
            self.failed=True;raise

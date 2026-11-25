"""ROS-ABI-isolated native sensor/encoder -> AMCL bridge, no actuator writes.

Import is pure. Only run_bridge() starts isolated ROS and a loopback socket.
Raw native world poses are needed to validate virtual ray provenance but are
NEVER used as encoder odometry, map->odom TF, or localization corrections.
The declared map prior is separate; neither a task goal nor its path moves
to the first AMCL estimate. Replay and live acquisition stay distinct.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import time

import hospital_runtime_evidence_io as evidence
import hospital_sensor_messages as messages
import hospital_sensor_runtime as native
import hospital_sensor_transport as transport
import hospital_sensor_platform as platform_math


MAX_WIRE_BYTES = 4 * 1024 * 1024


def costmap_guard_log_record(log_text):
    """Read-only ORIGINAL locked guard snapshot, never a rebuilt costmap."""
    marker='IJCAS_COSTMAP_GUARD_DIAGNOSTIC '
    lines=[line.split(marker,1)[1] for line in log_text.splitlines() if marker in line]
    if not lines:return None
    if len(lines)!=1:raise ValueError('One fatal original guard diagnostic expected, never select a best snapshot')
    raw=lines[0]
    if len(raw.encode())>MAX_WIRE_BYTES:raise ValueError('Bounded guard snapshot required')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate guard diagnostic field')
            result[key]=value
        return result
    decoder=json.JSONDecoder(object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
    value,end=decoder.raw_decode(raw)
    import re
    if re.sub(r'\x1b\[[0-9;]*m','',raw[end:]).strip():raise ValueError('Unexpected trailing diagnostic data')
    value=evidence.json_value(value)
    if value.get('version')!='ACTUAL_LOCKED_NAV2_COSTMAP_REJECTION_v1':raise ValueError('Original locked diagnostic version required')
    messages.frame_id(value['frame_id'])
    rows,columns=value['shape_rows_columns'];cells=rows*columns
    if (type(rows) is not int or type(columns) is not int or rows<0 or columns<0
            or type(value['projection_step']) is not int or not 0<=value['projection_step']<=5
            or any(type(value[k]) is not int or value[k]<0 for k in ('proposal_stamp_ns','pose_stamp_ns'))
            or type(value['complete_grid_snapshot']) is not bool
            or value['complete_grid_snapshot']!=(cells<=65536)):
        raise ValueError('Bounded original grid/projection/stamps required')
    messages.frames.vector(value['projected_xyyaw'],3,'actual projected pose')
    messages.frames.vector(value['origin_xy_m'],2,'actual costmap origin')
    if not 0<value['resolution_m']<1.:raise ValueError('Actual finite costmap resolution required')
    if value['cost'] is not None and (type(value['cost']) not in (int,float) or not math.isfinite(value['cost'])):
        raise ValueError('Original finite or explicit nonfinite-null rejection cost required')
    if (len(value['padded_footprint_xy_m'])<3 or len(value['padded_footprint_xy_m'])>32
            or len(value['cost_data'])!=(cells if cells<=65536 else 0)
            or any(type(x) is not int or not 0<=x<=255 for x in value['cost_data'])):
        raise ValueError('Exact original padded footprint and every native byte cell required')
    for point in value['padded_footprint_xy_m']:messages.frames.vector(point,2,'actual padded footprint vertex')
    if 'geometry_guard_version' in value:
        if (value['geometry_guard_version']!='CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1'
                or len(value.get('rejected_cell_column_row',[]))!=2
                or any(type(x) is not int for x in value['rejected_cell_column_row'])):
            raise ValueError('Exact closed-cell geometry diagnostic required')
        column,row=value['rejected_cell_column_row']
        if not ((column,row)==(-1,-1) or 0<=column<columns and 0<=row<rows):
            raise ValueError('Original in-grid rejection cell or invalid-geometry sentinel required')
    return value


def matched_current_pose(localization, history, current_stamp_ns, current_odom_xyyaw):
    """AMCL fix + EXACT encoder anchor, not GT, interpolation or shifted goal."""
    if localization is None:raise ValueError('Actual AMCL fix required')
    stamp = messages.stamp_value(localization['stamp'])
    if not 0 <= current_stamp_ns - stamp <= 1000000000 or stamp not in history:
        raise ValueError('Fresh AMCL fix and exact original encoder anchor required')
    inverse = messages.frames.transform_pose(history[stamp], [0., 0., 0.], inverse=True)
    map_from_odom = messages.frames.transform_pose(localization['xyyaw'], inverse)
    return dict(stamp_ns=current_stamp_ns, fix_stamp_ns=stamp,
        estimated_map_xyyaw=messages.frames.transform_pose(map_from_odom, current_odom_xyyaw),
        map_correction_NOT_used_as_velocity=True, unknown_encoder_uncertainty_NOT_a_calibrated_bound=True,
        no_ground_truth_pose_input=True)


def short_nav2_parameters(prefix):
    """NEW integration-controller config, not the frozen comparison controller."""
    return {
        prefix + '/controller_server': {'ros__parameters': dict(
            use_sim_time=True, controller_frequency=20., odom_topic=prefix + '/measured_encoder_odom',
            min_x_velocity_threshold=.0001, min_y_velocity_threshold=.0001, min_theta_velocity_threshold=.0001,
            failure_tolerance=0., publish_zero_velocity=True, progress_checker_plugin='progress_checker',
            goal_checker_plugins=['goal_checker'], controller_plugins=['FollowPath'],
            **{'progress_checker.plugin':'nav2_controller::SimpleProgressChecker',
               'progress_checker.required_movement_radius':.02,'progress_checker.movement_time_allowance':20.,
               'goal_checker.plugin':'nav2_controller::SimpleGoalChecker','goal_checker.stateful':True,
               'goal_checker.xy_goal_tolerance':.005,'goal_checker.yaw_goal_tolerance':math.radians(2.),
               'FollowPath.plugin':'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
               'FollowPath.desired_linear_vel':.03,'FollowPath.lookahead_dist':.15,
               'FollowPath.min_lookahead_dist':.1,'FollowPath.max_lookahead_dist':.2,
               'FollowPath.lookahead_time':1.5,'FollowPath.transform_tolerance':.3,
               'FollowPath.use_velocity_scaled_lookahead_dist':False,
               'FollowPath.min_approach_linear_velocity':.02,'FollowPath.approach_velocity_scaling_dist':.2,
               'FollowPath.use_collision_detection':True,'FollowPath.use_regulated_linear_velocity_scaling':True,
               'FollowPath.use_cost_regulated_linear_velocity_scaling':False,
               'FollowPath.regulated_linear_scaling_min_radius':.8,'FollowPath.regulated_linear_scaling_min_speed':.02,
               'FollowPath.use_rotate_to_heading':False,'FollowPath.allow_reversing':True,'FollowPath.max_angular_accel':.5})},
        prefix + '/local_costmap/local_costmap': {'ros__parameters':dict(
            use_sim_time=True, update_frequency=10., publish_frequency=2., global_frame='ijcas_sensor_dev_odom',
            robot_base_frame='base_link', rolling_window=True, width=4, height=4, resolution=.02,
            transform_tolerance=.3, footprint='[[0.815,0.4825],[0.815,-0.4825],[-0.815,-0.4825],[-0.815,0.4825]]',
            footprint_padding=.01, plugins=['static_layer','obstacle_layer','inflation_layer'],
            **{'static_layer.plugin':'nav2_costmap_2d::StaticLayer','static_layer.map_topic':prefix+'/map',
               'static_layer.map_subscribe_transient_local':True,
               'obstacle_layer.plugin':'nav2_costmap_2d::ObstacleLayer','obstacle_layer.enabled':True,
               'obstacle_layer.footprint_clearing_enabled':True,'obstacle_layer.observation_sources':'scan',
               'obstacle_layer.scan.topic':prefix+'/scan','obstacle_layer.scan.data_type':'LaserScan',
               'obstacle_layer.scan.marking':True,'obstacle_layer.scan.clearing':True,
               'obstacle_layer.scan.inf_is_valid':True,'obstacle_layer.scan.obstacle_min_range':.1,
               'obstacle_layer.scan.obstacle_max_range':3.5,'obstacle_layer.scan.raytrace_min_range':.1,
               'obstacle_layer.scan.raytrace_max_range':4.,'obstacle_layer.scan.max_obstacle_height':1.8,
               'inflation_layer.plugin':'nav2_costmap_2d::InflationLayer','inflation_layer.inflation_radius':.5,
               'inflation_layer.cost_scaling_factor':3.})}}


def platform_nav2_parameters(prefix,definition):
    """Actual Nav2 controller plugins, with the ORIGINAL P math outside SDK.

    Do not map stock differential-drive pure pursuit blindly to front/rear
    steering or replace the true task goal by P's nearby lattice endpoint.
    """
    definition=platform_math.checked_definition(definition)
    plan=json.loads(Path(definition['frozen_plan']).read_text())
    params=short_nav2_parameters(prefix)
    node=params[prefix+'/controller_server']['ros__parameters']
    for key in list(node):
        if key.startswith('FollowPath.') or key.startswith('goal_checker.'):node.pop(key)
    node.update({'progress_checker.movement_time_allowance':45.,
        'FollowPath.plugin':'ijcas_nav2_sensor_adapter/MeasuredPlatformController',
        'FollowPath.proposal_topic':prefix+'/platform_proposal',
        'FollowPath.acknowledgement_topic':prefix+'/platform_command_ack',
        'FollowPath.reported_pose_topic':prefix+'/actual_nav2_encoder_pose',
        'FollowPath.maximum_proposal_age_s':.25,'FollowPath.collision_projection_s':.25,
        'FollowPath.footprint_guard_version':'CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1',
        'goal_checker.plugin':'ijcas_nav2_sensor_adapter/DeclaredTaskGoalChecker',
        # ROS parameter arrays must be homogeneous. H6's declared yaw is
        # JSON integer0 while XY are doubles; this is representation ONLY.
        'goal_checker.declared_task_goal_xyyaw':[float(x) for x in plan['goal_world_xyyaw']],
        'goal_checker.declared_map_frame':'ijcas_sensor_dev_map',
        'goal_checker.fixed_observation_complete_topic':prefix+'/platform_fixed_observation_complete'})
    return params


def initial_prior_covariance(configuration):
    """Declared map prior, NEVER a correction derived from current truth.

    Whole missions start at their controlled fixed docking pose. This is
    NOT global relocalization from an unknown initial location. The separate
    short probe retains its displaced broad prior and recovery evidence.
    """
    if configuration.get('known_fixed_start_local_prior'):
        definition=configuration.get('mission_platform_definition')
        if (definition is None or configuration.get('known_fixed_start_local_prior')!=dict(position_std_m=.003,yaw_std_deg=.1)
                or configuration['initial_prior_xyyaw']!=json.loads(Path(definition['frozen_plan']).read_text())['start_world_xyyaw']):
            raise ValueError('Explicit common known fixed-task start prior required; never current GT reanchoring')
        position=.003;heading=math.radians(.1)
    else:position=.15;heading=math.radians(10.)
    covariance=[0.]*36;covariance[0]=covariance[7]=position**2;covariance[35]=heading**2
    return covariance


def receive_object(connection):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            piece = connection.recv(size - len(parts))
            if not piece:raise EOFError('Native bridge connection closed; no stale feedback')
            parts.extend(piece)
        return bytes(parts)
    size = struct.unpack('!I', exact(4))[0]
    if not 0 < size <= MAX_WIRE_BYTES:raise ValueError('Bounded complete native bridge frame required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ValueError('Duplicate native bridge key')
            result[key] = value
        return result
    return json.loads(exact(size), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite wire value')))


def send_object(connection, value):
    payload = json.dumps(evidence.json_value(value), separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(payload) <= MAX_WIRE_BYTES:raise ValueError('Native bridge frame exceeds bounded wire size')
    connection.sendall(struct.pack('!I', len(payload)) + payload)


def checked_configuration(value):
    import re
    value = evidence.json_value(value)
    if (value.get('version') != 'ISOLATED_NATIVE_SENSOR_AMCL_BRIDGE_v1'
            or value.get('authorized_data_only') is not True or value.get('actuator_connection') is not False
            or value.get('capture_kind') not in ('REPLAY_ACTUAL_ZERO_OBSERVATIONS', 'LIVE_ACTUAL_ACQUISITION')
            or type(value.get('domain_id')) is not int or not 1 <= value['domain_id'] <= 100
            or not re.fullmatch(r'/ijcas_sensor_dev/[a-z0-9_]+', value.get('topic_prefix', ''))):
        raise ValueError('Explicit isolated DATA-ONLY bridge scope required, never default ROS domain')
    messages.frame_id(value['map_frame_id']); messages.frame_id(value['odom_frame_id'])
    prior = messages.frames.vector(value['initial_prior_xyyaw'], 3, 'explicit fixed-map prior')
    if (value['map_frame_id'] == value['odom_frame_id'] or len(value['input_sha256']) < 3
            or type(value['force_stationary_updates']) is not bool):
        raise ValueError('Complete explicit map/encoder/prior and stationary-update policy required')
    for name in ('map_report', 'encoder_geometry', 'encoder_wrap_contract'):
        path = Path(value[name])
        if not path.is_absolute() or value['input_sha256'].get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Exact pinned sensor bridge input required: ' + name)
    for name, digest in value['input_sha256'].items():
        path = Path(name)
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen bridge source/input changed: ' + name)
    if type(value['maximum_frames']) is not int or not 1 <= value['maximum_frames'] <= 3010:
        raise ValueError('Bounded sensor bridge observation count required')
    if value.get('short_nav2_path') is not None:
        native.checked_short_nav2_definition(value['short_nav2_path'])
        if value['capture_kind'] != 'LIVE_ACTUAL_ACQUISITION':
            raise ValueError('Short Nav2 controller requires explicit LIVE acquisition, not replay-as-motion')
    if value.get('mission_platform_definition') is not None:
        definition=platform_math.checked_definition(value['mission_platform_definition'])
        if value['maximum_frames']!=platform_math.task_limits(definition['scenario'])['maximum_sensor_frames']:
            raise ValueError('Exact declared task-duration/5Hz sensor record bound required')
        if (value.get('short_nav2_path') is not None or value['capture_kind']!='LIVE_ACTUAL_ACQUISITION'
                or definition['actor_root']!=value['actor_root'] or value.get('compact_raw_source_frames') is not True
                or value['map_frame_id']!='ijcas_sensor_dev_map' or value['odom_frame_id']!='ijcas_sensor_dev_odom'):
            raise ValueError('Separate actual full-P sensor/Nav2 scope and fixed registered frames required')
        if any(definition[key]!=value[key] for key in ('encoder_geometry','encoder_wrap_contract')):
            raise ValueError('Same measured encoder geometry/wrap for control and scan diagnostics required')
        if any(value['input_sha256'].get(name)!=digest for name,digest in definition['input_sha256'].items()):
            raise ValueError('Outer frozen bridge must include ALL platform math/data inputs')
        prefix=Path(value['nav2_plugin_prefix']).resolve()
        for file in (prefix/'lib/libijcas_sensor_nav2_plugins.so',prefix/'share/ijcas_nav2_sensor_adapter/plugins.xml'):
            if value['input_sha256'].get(str(file))!=hashlib.sha256(file.read_bytes()).hexdigest():
                raise ValueError('Exact independently built Nav2 plugin binary/description required')
    if not value['actor_root'].startswith('/World/') or not value['capture_id']:
        raise ValueError('Exact native actor and acquisition identity required')
    initial_prior_covariance(value)
    return value


class MeasuredFrameConsumer:
    """Recompute causal wheel odometry; ignore world pose as estimator input."""
    def __init__(self, configuration):
        self.configuration = checked_configuration(configuration)
        self.sequence = messages.PacketSequence()
        geometry = json.loads(Path(configuration['encoder_geometry']).read_text())['geometry']
        wrap = json.loads(Path(configuration['encoder_wrap_contract']).read_text())
        self.encoder = native.EncoderOdometrySequence(actor_root=configuration['actor_root'], geometry=geometry,
            wrap_period_rad=wrap['wrap_period_rad'], maximum_abs_rate_rad_s=wrap['maximum_abs_rate_rad_s'],
            position_roundoff_rad=wrap['position_roundoff_rad'], maximum_sample_gap_s=.25)
        self.count = 0
        self.failed = False; self.identity = native._digest(self.configuration)

    def accept(self, frame):
        if self.failed:raise ValueError('Terminal native bridge consumer; no automatic reset')
        try:
            if native._digest(self.configuration) != self.identity:raise ValueError('Frozen bridge configuration changed')
            return self._accept(frame)
        except BaseException:
            self.failed = True
            raise

    def _accept(self, frame):
        reconstructed=isinstance(frame,transport.VerifiedFrame)
        if reconstructed:frame=frame.frame
        if self.count >= self.configuration['maximum_frames']:raise ValueError('Fixed bridge frame budget exceeded')
        packet = frame['scan_packet']; acquisition = packet['acquisition']
        if (acquisition['source_role'] != 'UNVALIDATED_NATIVE_QUERY_CAPTURE'
                or acquisition['actor_root'] != self.configuration['actor_root']
                or acquisition['capture_id'] != self.configuration['capture_id']):
            raise ValueError('Actual native source/actor/capture required, not a software scan fixture')
        if reconstructed:self.sequence._accept_immediately_validated(packet)
        else:self.sequence.accept(packet)
        encoder = frame['native_encoder']
        if encoder['stamp_ns'] != acquisition['stamp_ns']:raise ValueError('Exact acquired scan/encoder timestamp required')
        estimate = self.encoder.accept(encoder)
        if estimate != frame['causal_encoder_estimate']:
            raise ValueError('Independent receiver encoder recomputation differs; no GT or target substitution')
        self.count += 1
        return packet, encoder, estimate


class BridgeClient:
    """Bounded loopback peer; no ROS or SDK imports in the native interpreter."""
    def __init__(self, ready_path, *, maximum_exchange_wall_s=5., expected_configuration=None):
        self.ready = json.loads(Path(ready_path).read_text())
        if (self.ready.get('status') != 'DATA_ONLY_AMCL_BRIDGE_READY'
                or self.ready.get('host') != '127.0.0.1'
                or not 1024 <= self.ready['port'] <= 65535
                or not 0 < maximum_exchange_wall_s <= 5.):
            raise ValueError('Owned bounded loopback AMCL bridge endpoint required')
        self.configuration=None
        if expected_configuration is not None:
            expected_configuration = Path(expected_configuration).resolve()
            self.configuration=checked_configuration(json.loads(expected_configuration.read_text()))
            if (self.ready.get('configuration_path') != str(expected_configuration)
                    or self.ready.get('configuration_sha256') != hashlib.sha256(expected_configuration.read_bytes()).hexdigest()):
                raise ValueError('Actual bridge must ACK the exact frozen configuration before connection')
        self.connection = socket.create_connection(('127.0.0.1', self.ready['port']), timeout=maximum_exchange_wall_s)
        self.connection.settimeout(maximum_exchange_wall_s); self.closed = False

    def frame(self, frame):
        if self.closed:raise ValueError('Closed bridge, no automatic reconnection')
        stamp = frame['native_encoder']['stamp_ns']; sequence = frame['scan_packet']['sequence_number']
        transmitted=(transport._pack_immediately_verified_frame(frame) if self.configuration is not None
            and self.configuration.get('compact_raw_source_frames') else frame)
        send_object(self.connection, dict(token=self.ready['token'], operation='frame', frame=transmitted))
        result = receive_object(self.connection)
        if result.get('status') != 'EXACT_NATIVE_FRAME_PUBLISHED' or result.get('stamp_ns') != stamp or result.get('sequence_number') != sequence:
            raise RuntimeError('AMCL bridge did not ACK exact native acquisition; no stale feedback')
        return result

    def platform_control(self,encoder,relative_physics_time_s,actual_native_hold_ack):
        if self.closed or self.configuration is None or not self.configuration.get('mission_platform_definition'):
            raise ValueError('Own configured full-platform session required')
        send_object(self.connection,dict(token=self.ready['token'],operation='platform_control',
            native_encoder=encoder,relative_physics_time_s=relative_physics_time_s,actual_native_hold_ack=actual_native_hold_ack))
        result=receive_object(self.connection)
        if result.get('status')!='EXACT_MEASURED_PLATFORM_PROPOSAL' or result.get('stamp_ns')!=encoder['stamp_ns']:
            raise RuntimeError('Actual Nav2 platform proposal missing/ambiguous, no stale fallback')
        return result

    def zero_plugin_smoke(self):
        if self.closed or self.configuration is None or self.configuration.get('diagnostic_zero_plugin_smoke') is not True:
            raise ValueError('Explicit data-only zero-plugin diagnostic required')
        send_object(self.connection,dict(token=self.ready['token'],operation='zero_plugin_smoke'))
        result=receive_object(self.connection)
        if result.get('status')!='ACTUAL_NAV2_ZERO_PLUGIN_DIAGNOSTIC_NO_NATIVE_ACTUATION':
            raise RuntimeError('Actual plugin zero-output diagnostic failed')
        return result

    def close(self):
        if self.closed:return
        self.closed = True
        try:
            send_object(self.connection, dict(token=self.ready['token'], operation='finish'))
            return receive_object(self.connection)
        finally:self.connection.close()

    def start_follow_path(self):
        if self.closed:raise ValueError('Closed bridge; no motion request')
        # Lifecycle startup is BEFORE native Play, not a live feedback wait.
        # Give DDS discovery a separate bounded25s startup budget. Actual
        # moving-frame feedback keeps the original <=5s exchange timeout.
        original_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(30.)
            send_object(self.connection, dict(token=self.ready['token'], operation='start_follow_path'))
            result = receive_object(self.connection)
        finally:self.connection.settimeout(original_timeout)
        if result.get('status') != 'OWNED_NAV2_PATH_ACCEPTED':raise RuntimeError('Actual Nav2 FollowPath not accepted')
        return result

    def prepare_controller(self):
        if self.closed:raise ValueError('Closed bridge; no controller preparation')
        original_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(30.)
            send_object(self.connection, dict(token=self.ready['token'], operation='prepare_controller'))
            result = receive_object(self.connection)
        finally:self.connection.settimeout(original_timeout)
        if result.get('status') != 'OWNED_NAV2_CONTROLLER_READY_NO_GOAL':
            raise RuntimeError('Actual Nav2 preparation failed before native Play')
        return result


def run_bridge(configuration_path):
    """Own one AMCL/data-only session; terminate only its own process/context."""
    import sys
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use matching system ROS Humble Python3.10, never import its bindings into Kit3.11')
    path = Path(configuration_path).resolve(); config = checked_configuration(json.loads(path.read_text()))
    import os
    import secrets
    import select
    import subprocess
    import tempfile
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseStamped, Twist, TwistStamped
    from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from sensor_msgs.msg import LaserScan, JointState
    from tf2_msgs.msg import TFMessage
    from rosgraph_msgs.msg import Clock
    from lifecycle_msgs.srv import ChangeState, GetState
    from rcl_interfaces.srv import GetParameters
    from std_srvs.srv import Empty
    from std_msgs.msg import Bool

    output = path.parent
    if (output / 'bridge_ready.json').exists() or (output / 'bridge_result.json').exists():
        raise ValueError('Existing bridge session preserved; no automatic restart')
    consumer = MeasuredFrameConsumer(config)
    platform=(platform_math.MeasuredPlatformController(config['mission_platform_definition'])
        if config.get('mission_platform_definition') else None)
    wall_budget=(platform.definition['maximum_wall_time_s'] if platform is not None else
        config['short_nav2_path']['maximum_wall_time_s'] if config.get('short_nav2_path') else 180.)
    map_report = json.loads(Path(config['map_report']).read_text())
    prefix = config['topic_prefix']; map_frame = config['map_frame_id']; odom_frame = config['odom_frame_id']
    parameters = dict(use_sim_time=True, set_initial_pose=False, global_frame_id=map_frame,
        odom_frame_id=odom_frame, base_frame_id='base_link', tf_broadcast=True,
        scan_topic=prefix + '/scan', robot_model_type='nav2_amcl::OmniMotionModel',
        min_particles=500, max_particles=10000, max_beams=90, laser_min_range=.1, laser_max_range=12.,
        laser_model_type='likelihood_field_prob', do_beamskip=False, sigma_hit=.03,
        z_hit=.8, z_rand=.2, update_min_d=.02, update_min_a=.02, transform_tolerance=.5, resample_interval=1)
    command = ['/opt/ros/humble/lib/nav2_amcl/amcl', '--ros-args', '-r', '__node:=amcl', '-r', '__ns:=' + prefix,
        '-r', '/tf:=' + prefix + '/tf', '-r', '/tf_static:=' + prefix + '/tf_static', '-r', '/clock:=' + prefix + '/clock']
    for name, value in parameters.items():
        command.extend(['-p', name + ':=' + (str(value).lower() if type(value) is bool else str(value))])
    context = Context(); node = executor = process = server = connection = None
    poses = []; receipts = []; error = None; phase = 'initialization'; log_file = tempfile.TemporaryFile(mode='w+')
    receipt_archive=None
    navigation = dict(process=None, goal=None, result_future=None, parameter_readback=None,
        command=None, command_count=0,plugin_ack=None,plugin_pose=None,platform_requests=0)
    odom_history = {} if platform is None else platform.odom_history
    latest_native_tf = None; latest_amcl_tf = None; last_tf_relay_wall = 0.
    try:
        os.environ['ROS_LOCALHOST_ONLY']='1'  # Owned process only; no LAN actuator/topic connection.
        rclpy.init(args=[], context=context, domain_id=config['domain_id'])
        node = rclpy.create_node('native_data_bridge', namespace=prefix, context=context)
        executor = SingleThreadedExecutor(context=context); executor.add_node(node)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pubs = dict(map=node.create_publisher(OccupancyGrid, prefix + '/map', latched),
            static=node.create_publisher(TFMessage, prefix + '/tf_static', latched),
            dynamic=node.create_publisher(TFMessage, prefix + '/tf', 10),
            clock=node.create_publisher(Clock, prefix + '/clock', 10),
            scan=node.create_publisher(LaserScan, prefix + '/scan', 10),
            odometry=node.create_publisher(Odometry, prefix + '/measured_encoder_odom', 10),
            joints=node.create_publisher(JointState, prefix + '/native_joint_state', 10),
            platform_proposal=node.create_publisher(TwistStamped,prefix+'/platform_proposal',10),
            platform_complete=node.create_publisher(Bool,prefix+'/platform_fixed_observation_complete',10),
            prior=node.create_publisher(PoseWithCovarianceStamped, prefix + '/initialpose', 10))
        subscription = node.create_subscription(PoseWithCovarianceStamped, prefix + '/amcl_pose', poses.append, 50)
        def receive_command(message):
            values = [message.linear.x, message.linear.y, message.linear.z,
                message.angular.x, message.angular.y, message.angular.z]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite actual Nav2 command')
            navigation['command'] = dict(linear_m_s=values[:3], angular_rad_s=values[3:],
                received_native_clock_ns=clock.clock.sec * 10**9 + clock.clock.nanosec)
            navigation['command_count'] += 1
        command_subscription = node.create_subscription(Twist, prefix + '/nav2_cmd_vel', receive_command, 50)
        def receive_plugin_ack(message):navigation['plugin_ack']=message
        def receive_plugin_pose(message):navigation['plugin_pose']=message
        plugin_ack_subscription=node.create_subscription(TwistStamped,prefix+'/platform_command_ack',receive_plugin_ack,20)
        plugin_pose_subscription=node.create_subscription(PoseStamped,prefix+'/actual_nav2_encoder_pose',receive_plugin_pose,20)
        def retain_actual_amcl_transform(message):
            nonlocal latest_amcl_tf
            for transform in message.transforms:
                if transform.header.frame_id == map_frame and transform.child_frame_id == odom_frame:
                    latest_amcl_tf = transform
        # A controller starts AFTER the first real observation. Standard
        # dynamic TF is volatile, so a late listener misses that publication.
        # Relay only exact already received transforms, with ORIGINAL stamps:
        # no fake clock, scan repetition, extrapolation or GT authority.
        actual_tf_subscription = node.create_subscription(TFMessage, prefix + '/tf', retain_actual_amcl_transform, 50)
        grid = native.plane_map_message(map_report, map_frame_id=map_frame); pubs['map'].publish(grid)
        clock = Clock(); stamp = messages.stamp_parts(config['initial_native_stamp_ns'])
        clock.clock.sec = stamp['sec']; clock.clock.nanosec = stamp['nanosec']
        process = subprocess.Popen(command, env=dict(os.environ, ROS_DOMAIN_ID=str(config['domain_id'])), stdout=log_file, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 40.
        def pump(timeout=.01):
            nonlocal last_tf_relay_wall
            if process.poll() is not None:raise RuntimeError('Owned AMCL process exited')
            if navigation['process'] is not None and navigation['process'].poll() is not None:
                raise RuntimeError('Owned Nav2 controller process exited')
            if time.monotonic() > deadline:raise TimeoutError('Bounded bridge lifecycle/frame wait expired')
            wall = time.monotonic()
            if wall - last_tf_relay_wall >= .05:
                transforms = [value for value in (latest_native_tf, latest_amcl_tf) if value is not None]
                if transforms:pubs['dynamic'].publish(TFMessage(transforms=transforms))
                last_tf_relay_wall = wall
            pubs['clock'].publish(clock); executor.spin_once(timeout_sec=timeout)

        def publish_encoder(encoder,estimate,packet=None):
            nonlocal clock,latest_native_tf,dynamic_tf_count
            stamp=messages.stamp_parts(encoder['stamp_ns']);clock=Clock()
            clock.clock.sec=stamp['sec'];clock.clock.nanosec=stamp['nanosec']
            tf=TransformStamped();tf.header.stamp=clock.clock
            tf.header.frame_id=odom_frame;tf.child_frame_id='base_link'
            x,y,yaw=estimate['odom_xyyaw'];tf.transform.translation.x=x;tf.transform.translation.y=y
            tf.transform.rotation.z=math.sin(yaw/2.);tf.transform.rotation.w=math.cos(yaw/2.)
            latest_native_tf=tf;pubs['clock'].publish(clock);pubs['dynamic'].publish(TFMessage(transforms=[tf]))
            dynamic_tf_count+=1;odom_history[encoder['stamp_ns']]=estimate['odom_xyyaw']
            if len(odom_history)>128:del odom_history[next(iter(odom_history))]
            # ROS Python message assignment ALIASES nested objects. The
            # JointState frame below must never mutate the retained TF to
            # base_link -> base_link during subsequent listener relays.
            odom=Odometry();odom.header=copy.deepcopy(tf.header);odom.child_frame_id='base_link'
            odom.pose.pose.position.x=x;odom.pose.pose.position.y=y;odom.pose.pose.orientation=tf.transform.rotation
            odom.pose.covariance=[1e6 if i%7==0 else 0. for i in range(36)]  # UNKNOWN, not calibrated.
            odom.twist.covariance=list(odom.pose.covariance)
            if platform is not None:twist=platform.filtered_encoder_twist()
            elif estimate['measured_interval_twist'] is not None:twist=estimate['measured_interval_twist']['body_twist_u_v_r']
            else:
                names=encoder['joint_names'];selected=[names.index(name) for name in native.EncoderOdometrySequence.CHANNELS]
                q=encoder['positions_rad'];qd=encoder['velocities_rad_s']
                twist=messages.frames.centerline_encoder_twist(qd[selected[0]],qd[selected[1]],q[selected[2]],q[selected[3]],
                    geometry=consumer.encoder.geometry)['body_twist_u_v_r']
            odom.twist.twist.linear.x=twist[0];odom.twist.twist.linear.y=twist[1];odom.twist.twist.angular.z=twist[2]
            pubs['odometry'].publish(odom)
            if packet is not None:joints=native._native_joint_state_from_validated(packet,encoder)
            else:
                joints=JointState();joints.header=copy.deepcopy(tf.header);joints.header.frame_id='base_link'
                joints.name=encoder['joint_names'];joints.position=encoder['positions_rad'];joints.velocity=encoder['velocities_rad_s'];joints.effort=[]
            pubs['joints'].publish(joints)
        change = node.create_client(ChangeState, prefix + '/amcl/change_state')
        state = node.create_client(GetState, prefix + '/amcl/get_state')
        get_parameters = node.create_client(GetParameters, prefix + '/amcl/get_parameters')
        no_motion = node.create_client(Empty, prefix + '/request_nomotion_update')
        phase = 'AMCL_configure_activate_actual_parameter_readback'
        for client in (change, state, get_parameters):
            while not client.service_is_ready():pump()
        for transition in (1, 3):
            request = ChangeState.Request(); request.transition.id = transition; future = change.call_async(request)
            while not future.done():pump()
            if not future.result().success:raise RuntimeError('AMCL lifecycle transition rejected')
        future = state.call_async(GetState.Request())
        while not future.done():pump()
        if future.result().current_state.id != 3:raise RuntimeError('Actual AMCL ACTIVE required')
        request = GetParameters.Request(); request.names = list(parameters); future = get_parameters.call_async(request)
        while not future.done():pump()
        fields = {1: 'bool_value', 2: 'integer_value', 3: 'double_value', 4: 'string_value'}
        actual = {key: getattr(value, fields[value.type]) for key, value in zip(request.names, future.result().values)}
        if actual != parameters:raise ValueError('Actual AMCL parameters differ from frozen bridge definition')
        def configure_nav2():
            nonlocal deadline
            deadline = min(deadline, time.monotonic() + 25.)
            if navigation['process'] is not None:raise ValueError('ONE declared Nav2 goal only; no replacement or retry')
            task = config.get('short_nav2_path')
            if platform is None and (task is None or config['capture_kind'] != 'LIVE_ACTUAL_ACQUISITION'
                    or task['scope'] != 'SHORT_STRAIGHT_NAV2_INTERFACE_NOT_HOSPITAL_MISSION'
                    or task['mode'] != 'coordinated_counterphase' or len(task['states_xyyaw']) < 2):
                raise ValueError('Explicit frozen SHORT Nav2 path and live acquisition required')
            params = (short_nav2_parameters(prefix) if platform is None else platform_nav2_parameters(prefix,platform.definition))
            params_path = output / 'nav2_controller_parameters.json'; evidence.write_exclusive(params_path, params)
            controller_command = ['/opt/ros/humble/lib/nav2_controller/controller_server', '--ros-args',
                '-r', '__node:=controller_server', '-r', '__ns:=' + prefix,
                '-r', '/tf:=' + prefix + '/tf', '-r', '/tf_static:=' + prefix + '/tf_static',
                '-r', '/clock:=' + prefix + '/clock', '-r', 'cmd_vel:=' + prefix + '/nav2_cmd_vel',
                '--params-file', str(params_path)]
            owned_environment=dict(os.environ,ROS_DOMAIN_ID=str(config['domain_id']))
            if platform is not None:
                plugin=Path(config['nav2_plugin_prefix']).resolve()
                owned_environment['AMENT_PREFIX_PATH']=str(plugin)+':'+owned_environment.get('AMENT_PREFIX_PATH','')
                owned_environment['LD_LIBRARY_PATH']=str(plugin/'lib')+':'+owned_environment.get('LD_LIBRARY_PATH','')
            navigation['process'] = subprocess.Popen(controller_command, env=owned_environment,
                stdout=log_file, stderr=subprocess.STDOUT, text=True)
            lifecycle_client = node.create_client(ChangeState, prefix + '/controller_server/change_state')
            while not lifecycle_client.service_is_ready():pump()
            for transition in (1, 3):
                request = ChangeState.Request(); request.transition.id = transition
                future = lifecycle_client.call_async(request)
                while not future.done():pump()
                if not future.result().success:raise RuntimeError('Nav2 controller lifecycle transition failed')
            # Prove actual controller and full-envelope/collision parameters,
            # not just the YAML intended for a differently named child node.
            readbacks = {}
            for target, expected in params.items():
                client = node.create_client(GetParameters, target + '/get_parameters')
                while not client.service_is_ready():pump()
                request = GetParameters.Request(); request.names = list(expected['ros__parameters'])
                future = client.call_async(request)
                while not future.done():pump()
                type_fields = {1:'bool_value',2:'integer_value',3:'double_value',4:'string_value',8:'double_array_value',9:'string_array_value'}
                values = {}
                for name, value in zip(request.names, future.result().values):
                    if value.type not in type_fields:raise ValueError('Missing/incorrect actual Nav2 parameter: ' + name)
                    decoded = getattr(value, type_fields[value.type]); values[name] = list(decoded) if value.type in (8,9) else decoded
                if values != expected['ros__parameters']:raise ValueError('Actual Nav2 controller/costmap settings differ: ' + target)
                readbacks[target] = values
            navigation['parameter_readback'] = readbacks
            if platform is not None:
                library=(Path(config['nav2_plugin_prefix'])/'lib/libijcas_sensor_nav2_plugins.so').resolve(strict=True)
                maps=Path('/proc/'+str(navigation['process'].pid)+'/maps').read_text()
                if str(library) not in maps or str(library)+' (deleted)' in maps:
                    raise ValueError('Actual owned Nav2 process did not load the pinned platform plugin binary')
                navigation['actual_plugin_binary']=dict(path=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest())
            deadline = time.monotonic() + wall_budget
            return dict(status='OWNED_NAV2_CONTROLLER_READY_NO_GOAL',
                actual_controller_and_costmap_parameters=readbacks, actual_goal_count=0)

        def arm_nav2():
            if navigation['process'] is None or navigation['goal'] is not None:
                raise ValueError('One already prepared Nav2 controller and ONE first goal required')
            states=(config['short_nav2_path']['states_xyyaw'] if platform is None else
                [platform.plan['start_world_xyyaw']]+[row['world_xyyaw'] for row in platform.plan['selected_states']])
            goal=(states[-1] if platform is None else platform.plan['goal_world_xyyaw'])
            client = ActionClient(node, FollowPath, prefix + '/follow_path')
            while not client.server_is_ready():pump()
            path = RosPath(); path.header.frame_id = map_frame; path.header.stamp = clock.clock
            for x, y, heading in states:
                pose = PoseStamped(); pose.header = path.header; pose.pose.position.x = float(x); pose.pose.position.y = float(y)
                pose.pose.orientation.z = math.sin(heading / 2.); pose.pose.orientation.w = math.cos(heading / 2.)
                path.poses.append(pose)
            request = FollowPath.Goal(); request.path = path; request.controller_id = 'FollowPath'; request.goal_checker_id = 'goal_checker'
            future = client.send_goal_async(request)
            while not future.done():pump()
            handle = future.result()
            if not handle.accepted:raise RuntimeError('Actual Nav2 FollowPath goal rejected')
            navigation['action_client'] = client; navigation['goal'] = handle; navigation['result_future'] = handle.get_result_async()
            until = time.monotonic() + 1.
            while navigation['command'] is None and time.monotonic() < until:pump()
            return dict(status='OWNED_NAV2_PATH_ACCEPTED', fixed_goal_xyyaw=goal,
                frozen_reference_endpoint_xyyaw=states[-1],unplanned_goal_edge_appended=False,
                actual_controller_and_costmap_parameters=navigation['parameter_readback'],
                actual_nav2_command=navigation['command'], task_goal_NOT_reanchored_to_AMCL=True)

        def proposal_message(proposal):
            message=TwistStamped();message.header.frame_id='base_link'
            stamp=messages.stamp_parts(proposal['stamp_ns'])
            message.header.stamp.sec=stamp['sec'];message.header.stamp.nanosec=stamp['nanosec']
            u,v,rate=proposal['proposed_body_twist_u_v_r']
            message.twist.linear.x=u;message.twist.linear.y=v;message.twist.angular.z=rate
            return message

        def actual_platform_command(request):
            if platform is None or config.get('diagnostic_zero_plugin_smoke'):
                raise ValueError('An actual full-platform control definition, not a smoke test, is required')
            encoder=request['native_encoder'];estimate=platform.observe_encoder(encoder)
            publish_encoder(encoder,estimate)
            proposal=platform.propose(encoder,relative_physics_time_s=request['relative_physics_time_s'],
                actual_native_hold_ack=request['actual_native_hold_ack'])
            navigation['last_original_platform_proposal']=proposal
            navigation['platform_requests']+=1
            if navigation['platform_requests']>platform_math.task_limits(platform.definition['scenario'])['maximum_control_records']:
                raise ValueError('Declared task-duration/20Hz platform command budget exceeded')
            if navigation['goal'] is None and platform.controller.lifecycle_stage!='pre_hold':
                # Warm TF/history through the ORIGINAL2s pre-hold before
                # asking the actual Nav2 server for ONE frozen path.
                pubs['platform_proposal'].publish(proposal_message(dict(proposal,proposed_body_twist_u_v_r=[0.,0.,0.])))
                navigation['path_acceptance']=arm_nav2()
            if navigation['goal'] is None:
                if max(abs(value) for value in proposal['proposed_body_twist_u_v_r'])>1e-10:
                    raise RuntimeError('No direct forward fallback before actual Nav2 path admission')
                return dict(status='EXACT_MEASURED_PLATFORM_PROPOSAL',stamp_ns=encoder['stamp_ns'],proposal=proposal,
                    prehold_ZERO_only_before_Nav2=True,actual_nav2_command=None,actual_nav2_action_status=None,
                    actual_Nav2_path_acceptance=None)
            before_count=navigation['command_count'];pubs['platform_proposal'].publish(proposal_message(proposal))
            if proposal['complete']:
                if max(abs(value) for value in proposal['proposed_body_twist_u_v_r'])>1e-10:
                    raise RuntimeError('Fixed-hold completion may only send zero velocity')
                pubs['platform_complete'].publish(Bool(data=True))
                until=time.monotonic()+3.
                while not navigation['result_future'].done() and time.monotonic()<until:pump()
                if not navigation['result_future'].done() or navigation['result_future'].result().status!=4:
                    raise RuntimeError('Actual Nav2 action did not accept the declared task goal/fixed-hold completion')
                # Preserve the RECEIVED actual server output, not a constructed
                # expected-zero tuple labelled as an observation.
                actual_zero=navigation['command']
                if actual_zero is None or max(abs(v) for v in actual_zero['linear_m_s']+actual_zero['angular_rad_s'])>1e-10:
                    raise RuntimeError('Actual received Nav2 completion output is not zero')
                return dict(status='EXACT_MEASURED_PLATFORM_PROPOSAL',stamp_ns=encoder['stamp_ns'],proposal=proposal,
                    prehold_ZERO_only_before_Nav2=False,actual_nav2_command=actual_zero,
                    actual_nav2_action_status=4,actual_Nav2_path_acceptance=navigation.get('path_acceptance'),
                    fixed_hold_complete_ZERO_only=True)
            until=time.monotonic()+1.
            while time.monotonic()<until:
                if navigation['result_future'].done():raise RuntimeError('Nav2 path ended before the ORIGINAL P fixed hold')
                ack=navigation['plugin_ack'];actual_command=navigation['command']
                ack_stamp=(None if ack is None else ack.header.stamp.sec*10**9+ack.header.stamp.nanosec)
                if ack_stamp==encoder['stamp_ns'] and navigation['command_count']>before_count and actual_command is not None:
                    expected=[ack.twist.linear.x,ack.twist.linear.y,ack.twist.linear.z,ack.twist.angular.x,ack.twist.angular.y,ack.twist.angular.z]
                    observed=actual_command['linear_m_s']+actual_command['angular_rad_s']
                    if max(abs(a-b) for a,b in zip(expected,observed))<=1e-10:break
                pump()
            else:raise TimeoutError('Actual Nav2 server did not publish its matching platform-proposal output')
            actual_pose=navigation['plugin_pose']
            if actual_pose is None:raise RuntimeError('Actual Nav2 encoder-TF pose receipt missing')
            pose_stamp=actual_pose.header.stamp.sec*10**9+actual_pose.header.stamp.nanosec
            q=actual_pose.pose.orientation
            pose_xyyaw=[actual_pose.pose.position.x,actual_pose.pose.position.y,
                math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
            if (actual_pose.header.frame_id!=odom_frame or pose_stamp not in odom_history
                    or not 0<=encoder['stamp_ns']-pose_stamp<=100000000
                    or max(abs(a-b) for a,b in zip(pose_xyyaw,odom_history[pose_stamp]))>1e-6):
                raise RuntimeError('Actual Nav2 pose does not match its EXACT original encoder-TF anchor')
            return dict(status='EXACT_MEASURED_PLATFORM_PROPOSAL',stamp_ns=encoder['stamp_ns'],proposal=proposal,
                actual_nav2_command=actual_command,actual_nav2_action_status=None,
                actual_plugin_ack_source_stamp_ns=ack_stamp,
                actual_nav2_encoder_pose=dict(stamp_ns=pose_stamp,odom_xyyaw=pose_xyyaw),
                prehold_ZERO_only_before_Nav2=False,actual_Nav2_path_acceptance=navigation.get('path_acceptance'))
        server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen(1); token = secrets.token_hex(32)
        evidence.write_exclusive(output / 'bridge_ready.json', dict(status='DATA_ONLY_AMCL_BRIDGE_READY',
            host='127.0.0.1', port=server.getsockname()[1], token=token, pid=os.getpid(),
            configuration_path=str(path), configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            domain_id=config['domain_id'], topic_prefix=prefix, actual_AMCL_parameters=actual,
            capture_kind=config['capture_kind'], no_actuator_or_Nav2_controller_publishers=True))
        phase = 'await_owned_native_connection'; deadline = time.monotonic() + 180.
        while not select.select([server], [], [], .01)[0]:pump()
        connection, peer = server.accept(); connection.settimeout(3.)
        prior_sent = False; dynamic_tf_count = stationary_requests = 0
        phase = 'actual_scan_and_measured_encoder_processing'
        deadline = time.monotonic() + wall_budget
        receipt_archive=(transport.BoundedJSONLArchive(output/'actual_frame_receipts.jsonl.gz',maximum_records=config['maximum_frames'])
            if platform is not None else None)
        while True:
            while not select.select([connection], [], [], .005)[0]:pump()
            request = receive_object(connection)
            if request.get('token') != token:raise ValueError('Wrong owned loopback bridge token')
            if request.get('operation') == 'finish':
                if navigation['goal'] is not None and not navigation['result_future'].done():
                    future = navigation['goal'].cancel_goal_async()
                    while not future.done():pump()
                send_object(connection, dict(status='OWNED_DATA_BRIDGE_FINISHED', native_frames=len(receipts))); break
            if request.get('operation') == 'start_follow_path':
                send_object(connection, arm_nav2()); continue
            if request.get('operation') == 'prepare_controller':
                send_object(connection, configure_nav2()); continue
            if request.get('operation')=='platform_control':
                send_object(connection,actual_platform_command(request));continue
            if request.get('operation')=='zero_plugin_smoke':
                if config.get('diagnostic_zero_plugin_smoke') is not True or platform is None or not receipts:
                    raise ValueError('One actual scan plus explicit no-actuation diagnostic required')
                estimate=platform.last_estimate;encoder=platform.previous_encoder
                zero=dict(stamp_ns=encoder['stamp_ns'],proposed_body_twist_u_v_r=[0.,0.,0.])
                pubs['platform_proposal'].publish(proposal_message(zero))
                accepted=arm_nav2();until=time.monotonic()+2.
                while time.monotonic()<until:
                    ack=navigation['plugin_ack'];command=navigation['command']
                    if ack is not None and ack.header.stamp.sec*10**9+ack.header.stamp.nanosec==encoder['stamp_ns'] and command is not None:break
                    pump()
                else:raise TimeoutError('Actual compiled Nav2 plugin acknowledgement missing')
                if any(abs(v)>1e-12 for v in command['linear_m_s']+command['angular_rad_s']):
                    raise RuntimeError('Data-only diagnostic must remain exact zero')
                send_object(connection,dict(status='ACTUAL_NAV2_ZERO_PLUGIN_DIAGNOSTIC_NO_NATIVE_ACTUATION',
                    actual_plugin_binary=navigation['actual_plugin_binary'],actual_path_acceptance=accepted,
                    actual_zero_command=command,native_actuator_connection=False,simulator_Play_calls=0))
                continue
            if request.get('operation') != 'frame':raise ValueError('Unknown bridge operation, no command dispatch')
            incoming=(transport.unpack_frame(request['frame']) if config.get('compact_raw_source_frames') else request['frame'])
            packet, encoder, estimate = consumer.accept(incoming)
            # Consumer.accept fully validates the packet + encoder before
            # returning. Immediate synchronous conversion, no skipped input
            # boundary check and no duplicate multi-MiB reconstruction.
            objects = messages._ros_message_objects_from_validated(packet); stamp_ns = encoder['stamp_ns']
            control_estimate=estimate if platform is None else platform.observe_encoder(encoder)
            publish_encoder(encoder,control_estimate,packet)
            first = not prior_sent
            if first:
                pubs['static'].publish(objects['base_to_sensor_tf'])
                prior = PoseWithCovarianceStamped(); prior.header.stamp = objects['scan'].header.stamp; prior.header.frame_id = map_frame
                px, py, heading = config['initial_prior_xyyaw']; prior.pose.pose.position.x = px; prior.pose.pose.position.y = py
                prior.pose.pose.orientation.z = math.sin(heading / 2.); prior.pose.pose.orientation.w = math.cos(heading / 2.)
                prior.pose.covariance=initial_prior_covariance(config)
                while not pubs['prior'].get_subscription_count():pump()
                pubs['prior'].publish(prior); prior_sent = True
                until = time.monotonic() + .25
                while time.monotonic() < until:pump()
            elif config['force_stationary_updates']:
                while not no_motion.service_is_ready():pump()
                future = no_motion.call_async(Empty.Request())
                while not future.done():pump()
                future.result(); stationary_requests += 1
            while not pubs['scan'].get_subscription_count():pump()
            pubs['scan'].publish(objects['scan'])  # ONE publication per actual acquisition.
            until = time.monotonic() + 2.
            while (not poses or messages.stamp_value(native.static_pose_message_fields(poses[-1])['stamp']) != stamp_ns) and time.monotonic() < until:
                pump()
            localization = None if not poses else native.static_pose_message_fields(poses[-1])
            if platform is not None and localization is not None:platform.accept_amcl_fix(localization)
            feedback = None
            if localization is not None:
                try:feedback = matched_current_pose(localization, odom_history, stamp_ns, control_estimate['odom_xyyaw'])
                except ValueError:pass  # Explicit absent feedback, never old/latest anchor substitution.
            if navigation['process'] is not None and platform is None:
                until = time.monotonic() + .15
                while (navigation['command'] is None or navigation['command']['received_native_clock_ns'] != stamp_ns) and time.monotonic() < until:pump()
            action_status = None
            if navigation['result_future'] is not None and navigation['result_future'].done():
                action_status = int(navigation['result_future'].result().status)
            result = dict(status='EXACT_NATIVE_FRAME_PUBLISHED', stamp_ns=stamp_ns,
                sequence_number=packet['sequence_number'], localization=localization,
                measured_encoder_estimate=estimate, capture_kind=config['capture_kind'],
                current_matched_feedback=feedback, actual_nav2_command=navigation['command'], actual_nav2_action_status=action_status,
                fresh_fix_matches_current_scan=localization is not None and messages.stamp_value(localization['stamp']) == stamp_ns,
                GT_pose_used_for_odometry_or_localization_correction=False,
                initial_prior_is_separate_explicit_map_prior=True, no_actuator_commands=True)
            if receipt_archive is None:receipts.append(result)
            else:
                receipt_archive(result);receipts.append(dict(stamp_ns=stamp_ns,sequence_number=packet['sequence_number'],
                    fresh_fix_matches_current_scan=result['fresh_fix_matches_current_scan']))
            send_object(connection, result)
        result = dict(status='DATA_ONLY_AMCL_SESSION_FINISHED_NOT_NAVIGATION_PASS', capture_kind=config['capture_kind'],
            frame_count=len(receipts), receipts=receipts, requested_parameters=parameters, actual_parameters=actual,
            one_scan_publication_per_original_acquisition=True, explicit_stationary_update_service_calls=stationary_requests,
            late_listener_TF_relay_retains_original_acquisition_stamps=True,
            encoder_TF_publications=dynamic_tf_count, identity_odom_only_at_first_native_frame=True,
            no_GT_odometry_or_TF_authority=True, no_actuator_or_Nav2_controller_started=navigation['process'] is None,
            actual_nav2_controller_started=navigation['process'] is not None, actual_nav2_command_count=navigation['command_count'],
            actual_nav2_parameters=navigation['parameter_readback'],
            moving_localization_or_Nav2_success=False)
        if receipt_archive is not None:result['original_AMCL_encoder_frame_receipt_archive']=receipt_archive.finish()
        result.update(actual_platform_control_requests=navigation['platform_requests'],
            actual_Nav2_path_acceptance=navigation.get('path_acceptance'),actual_plugin_binary=navigation.get('actual_plugin_binary'),
            actual_Nav2_result_status=(None if navigation['result_future'] is None or not navigation['result_future'].done()
                else int(navigation['result_future'].result().status)))
    except BaseException as failure:
        error = type(failure).__name__ + ': ' + str(failure)
        result = dict(status='DATA_ONLY_AMCL_BRIDGE_ERROR_NOT_NAVIGATION', phase=phase, error=error, receipts=receipts)
        if navigation.get('last_original_platform_proposal') is not None:
            result['last_original_platform_proposal']=navigation['last_original_platform_proposal']
        if poses:result['last_original_received_AMCL_pose']=native.static_pose_message_fields(poses[-1])
    finally:
        if receipt_archive is not None and not receipt_archive.finished:
            try:result['original_AMCL_encoder_frame_receipt_archive']=receipt_archive.finish()
            except Exception as failure:result['receipt_archive_error']=type(failure).__name__+': '+str(failure)
        for sock in (connection, server):
            if sock is not None:sock.close()
        for owned in (navigation['process'], process):
            if owned is not None:
                if owned.poll() is None:owned.terminate()
                try:owned.wait(timeout=3.)
                except subprocess.TimeoutExpired:owned.kill(); owned.wait(timeout=3.)
        if executor is not None:executor.shutdown()
        if node is not None:node.destroy_node()
        if context.ok():context.shutdown()
        log_file.seek(0);log_text=log_file.read(8*1024*1024+1);log_file.close()
        if len(log_text)>8*1024*1024:result['owned_process_log_exceeded_bound']=True
        else:
            target=output/'owned_ROS_process.log'
            with target.open('x',encoding='utf-8') as stream:stream.write(log_text);stream.flush();os.fsync(stream.fileno())
            result['complete_owned_ROS_process_log']=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            try:
                diagnostic=costmap_guard_log_record(log_text)
                if diagnostic is not None:
                    target=output/'actual_costmap_rejection.json';evidence.write_exclusive(target,diagnostic)
                    result['actual_locked_costmap_guard_diagnostic']=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        original_snapshot_NOT_reconstructed=True,guard_NOT_disabled_or_relaxed=True)
            except Exception as failure:result['guard_diagnostic_readback_error']=type(failure).__name__+': '+str(failure)
        result['owned_AMCL_process_log_tail']=log_text[-32768:]
        evidence.write_exclusive(output / 'bridge_result.json', result)
    if error:raise RuntimeError(error)
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:raise SystemExit('Exactly one checked data-only bridge configuration required')
    run_bridge(sys.argv[1])

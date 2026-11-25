"""Hospital scan wire/message and fixed-map estimate adapters, without nodes.

No ROS context, publisher, socket, SDK, timeline or controller is initialized.
The optional message factory imports installed ROS message types lazily; it
does NOT import system Python-3.10 bindings into Isaac Python-3.11. Instant
native rays retain zero inter-beam time, exact raw returns and explicit TF.
This is software plumbing, NOT sensor/localization/navigation validation.
"""

import copy
import hashlib
import json
import math

import numpy as np

import hospital_sensor_navigation as frames
import hospital_runtime_evidence_io as evidence


VERSION='HOSPITAL_INSTANT_SCAN_MESSAGE_AND_FIXED_MAP_ESTIMATE_ADAPTER_v1'
MAX_PACKET_BYTES=4*1024*1024
MAX_STAMP_NS=(2**31)*10**9-1
SOURCE_ROLES=('SOFTWARE_FIXTURE','UNVALIDATED_NATIVE_QUERY_CAPTURE')
FIRST_RETURN_MODEL='FIRST_EXTERNAL_NATIVE_RANGE_ALL_RAW_AUDITED_v1'


def first_external_scan_return(ray,hits,*,own_collision_paths,range_min):
    """First-return transducer, with ALL raw returns retained and audited.

    Scalar/range/identity/normal validity is required for every hit. The
    original100um point-on-ray check is required for every own hit and every
    nearest external tie. A STRICTLY farther external hit is occluded and
    never supplies a measurement; its unmodified barycentric point remains
    in the raw audit, not certified as an observed surface or turned to free.
    The independent physical collision guard still checks the entire scene.
    """
    own=set(own_collision_paths);raw=[]
    for hit in hits:
        path=hit['collision'];distance=float(hit['distance_m'])
        point=frames.vector(hit['position_m'],3,'raw_hit_position')
        normal=frames.vector(hit['normal'],3,'raw_hit_normal')
        if (not isinstance(path,str) or not path.startswith('/World/') or not math.isfinite(distance)
                or not 0.<=distance<=ray['length_m'] or abs(float(np.linalg.norm(normal))-1.)>1e-4):
            raise ValueError('Malformed/nonfinite native hit, including occluded returns')
        raw.append(dict(collision=path,distance_m=distance,position_m=point.tolist(),normal=normal.tolist()))
    external=[hit for hit in raw if hit['collision'] not in own]
    distance=min((hit['distance_m'] for hit in external),default=None)
    visible=[hit for hit in raw if hit['collision'] in own or hit['distance_m']==distance]
    result=frames.scan_return(ray,visible,own_collision_paths=own_collision_paths,range_min=range_min)
    origin=np.asarray(ray['origin_world_m']);direction=np.asarray(ray['direction_world'])
    occluded=[dict(raw_hit_index=i,collision=hit['collision'],distance_m=hit['distance_m'],
        first_external_range_m=distance,native_point_off_ray_m=float(np.max(np.abs(
            np.asarray(hit['position_m'])-origin-hit['distance_m']*direction))),
        raw_position_NOT_repaired=True,NOT_used_as_observed_first_return=True)
        for i,hit in enumerate(raw) if hit['collision'] not in own and hit['distance_m']>distance]
    result.update(raw_hits=raw,exact_self_hits=[hit for hit in raw if hit['collision'] in own],
        all_external_hits=external,first_return_model=FIRST_RETURN_MODEL,occluded_raw_return_audit=occluded)
    return result


def frame_id(value):
    if (not isinstance(value,str) or not value or value.startswith('/')
            or any(ch.isspace() for ch in value)):
        raise ValueError('Explicit unprefixed ROS frame identifier required')
    return value


def stamp_parts(stamp_ns):
    if type(stamp_ns) is not int or not 0<=stamp_ns<=MAX_STAMP_NS:
        raise ValueError('Explicit nonnegative simulation nanoseconds fitting ROS signed seconds required')
    sec,nanosec=divmod(stamp_ns,10**9)
    return dict(sec=sec,nanosec=nanosec)


def stamp_value(stamp):
    if (type(stamp) is not dict or set(stamp)!=set(('sec','nanosec'))
            or type(stamp['sec']) is not int or type(stamp['nanosec']) is not int
            or not 0<=stamp['sec']<2**31 or not 0<=stamp['nanosec']<10**9):
        raise ValueError('Exact canonical ROS seconds/nanoseconds required')
    return stamp['sec']*10**9+stamp['nanosec']


def scan_geometry(acquisition,config):
    """Same geometry for queries and packet verification; no default mount."""
    stamp_parts(acquisition['stamp_ns'])
    actor=acquisition['actor_root']
    if (not isinstance(actor,str) or not actor.startswith('/World/') or actor.endswith('/')
            or acquisition['source_role'] not in SOURCE_ROLES
            or not isinstance(acquisition['capture_id'],str) or not acquisition['capture_id']):
        raise ValueError('Exact actor and explicit UNVALIDATED capture/fixture identity required')
    own=config['own_collision_paths']
    if (type(own) not in (list,tuple) or not own or len(set(own))!=len(own)
            or any(not isinstance(path,str) or not path.startswith(actor+'/') for path in own)):
        raise ValueError('Unique exact actor collision identities required; no prefix self-hit exclusion')
    base=frame_id(config['base_frame_id']);laser=frame_id(config['sensor_frame_id'])
    geometry=frames.scan_geometry(acquisition['base_xyz'],acquisition['base_wxyz'],
        config['sensor_xyz_in_base'],config['sensor_wxyz_in_base'],frame_id=laser,
        ray_count=config['ray_count'],angle_min=config['angle_min_rad'],angle_increment=config['angle_increment_rad'],
        range_min=config['range_min_m'],range_max=config['range_max_m'],rate_hz=config['rate_hz'])
    if base==laser and (np.linalg.norm(config['sensor_xyz_in_base'])>1e-12 or
            not np.allclose(frames.rotation_wxyz(config['sensor_wxyz_in_base']),np.eye(3),rtol=0.,atol=1e-12)):
        raise ValueError('Nonidentity sensor extrinsics require a child sensor frame')
    if config['noise_model']!='ideal_native_raycast_no_noise':
        raise ValueError('Only explicitly disclosed ideal instantaneous ray data are implemented; no invented noise validation')
    return geometry


def scan_packet(acquisition,config,records,sequence_number):
    """All beams/hits retained; any below-min external obstacle blocks encoding.

    The caller supplies raw ALL-hit observations taken at this exact pose and
    simulation stamp. This function verifies internal consistency, not native
    acquisition freshness, cooked geometry or an actual sensor observation.
    """
    if type(sequence_number) is not int or sequence_number<0:
        raise ValueError('Nonnegative explicit acquisition sequence required')
    geometry=scan_geometry(acquisition,config)
    if type(records) not in (list,tuple) or len(records)!=len(geometry['rays']):
        raise ValueError('Every prescribed beam must have an explicit raw observation, not a missing/free substitution')
    decoded=[];ranges=[];blocked=[];saved_records=[]
    for i,(ray,record) in enumerate(zip(geometry['rays'],records)):
        if (type(record) is not dict or set(record)!=set(('index','hits'))
                or type(record['index']) is not int or record['index']!=i or type(record['hits']) not in (list,tuple)):
            raise ValueError('Exactly ordered complete unique beam observations required')
        model=config.get('return_model')
        if model not in (None,FIRST_RETURN_MODEL):raise ValueError('Unreviewed sensor return model')
        decoder=frames.scan_return if model is None else first_external_scan_return
        result=decoder(ray,record['hits'],own_collision_paths=config['own_collision_paths'],
            range_min=config['range_min_m'])
        decoded.append(result)
        normalized_hits=[]
        for original,normalized in zip(record['hits'],result['raw_hits']):
            # Only the four validated numerical fields are representation-
            # normalized. Extra native metadata is retained, not discarded;
            # unsupported extra objects fail before any output is created.
            saved=dict(original);saved.update(normalized)
            normalized_hits.append(saved)
        saved_records.append(dict(index=i,hits=normalized_hits))
        if result['status']=='NO_RETURN':ranges.append(dict(kind='NO_RETURN'))
        elif result['status']=='VALID_RANGE':ranges.append(dict(kind='RANGE',metres=result['valid_range_m']))
        else:
            blocked.append(i);ranges.append(dict(kind='BLOCKED_BELOW_MIN',metres=result['nearest_external_hit']['distance_m']))
    stamp=stamp_parts(acquisition['stamp_ns'])
    scan=dict(header=dict(stamp=stamp,frame_id=config['sensor_frame_id']),
        angle_min=geometry['angle_min_rad'],angle_max=geometry['angle_max_rad'],angle_increment=geometry['angle_increment_rad'],
        time_increment=0.,scan_time=float(geometry['scan_time_s']),range_min=geometry['range_min_m'],range_max=geometry['range_max_m'],
        ranges=ranges,intensities=[])
    # ROS LaserScan uses float32 fields. Do not let finite JSON values become
    # overflow/no-return or zero resolution/range through wire quantization.
    maximum=float(np.finfo(np.float32).max)
    values=[scan[key] for key in ('angle_min','angle_max','angle_increment','scan_time','range_min','range_max')]
    values.extend(item['metres'] for item in ranges if 'metres' in item)
    if any(abs(value)>maximum for value in values):raise ValueError('LaserScan float32 overflow; not a valid no-return')
    converted={key:float(np.float32(scan[key])) for key in ('angle_increment','scan_time','range_min','range_max')}
    if (converted['angle_increment']<=0. or converted['scan_time']<=0.
            or not 0.<converted['range_min']<converted['range_max']):
        raise ValueError('LaserScan range/rate/angular resolution lost in float32; no wire repair')
    # Some native SDK vector wrappers cannot be deep-copied/pickled. Read
    # validated numerical components instead; final json_value recursively
    # copies the resulting ordinary Python representation.
    saved_acquisition=dict(acquisition)
    saved_acquisition.update(base_xyz=frames.vector(acquisition['base_xyz'],3,'base_xyz').tolist(),
        base_wxyz=frames.vector(acquisition['base_wxyz'],4,'base_wxyz').tolist())
    saved_config=dict(config)
    saved_config.update(sensor_xyz_in_base=frames.vector(config['sensor_xyz_in_base'],3,'sensor_xyz').tolist(),
        sensor_wxyz_in_base=frames.vector(config['sensor_wxyz_in_base'],4,'sensor_wxyz').tolist(),
        own_collision_paths=list(config['own_collision_paths']))
    for key in ('angle_min_rad','angle_increment_rad','range_min_m','range_max_m','rate_hz'):
        saved_config[key]=float(config[key])
    packet=dict(version=VERSION,kind='INSTANT_SCAN',sequence_number=sequence_number,
        acquisition=saved_acquisition,configuration=saved_config,raw_records=saved_records,
        decoded_rays=decoded,scan=scan,blocked_ray_indices=blocked,
        status='READY_FOR_MESSAGE_ENCODING_NOT_SENSOR_VALIDATION' if not blocked else 'HOLD_BELOW_MIN_EXTERNAL_OBSTACLE_NOT_FREE',
        publishable_without_silent_near_obstacle_loss=not blocked,
        sensor_origin_world_m=geometry['sensor_origin_world_m'],
        actual_sensor_success=False,localization_or_Nav2_success=False,new_trial_count=0,actual_motion_authorized=False)
    return evidence.json_value(packet)


def validate_packet(packet):
    """Reconstruct against raw observations; metadata cannot relabel a return."""
    if type(packet) is not dict or packet.get('version')!=VERSION or packet.get('kind')!='INSTANT_SCAN':
        raise ValueError('Exact reviewed instantaneous scan packet required')
    expected=scan_packet(packet['acquisition'],packet['configuration'],packet['raw_records'],packet['sequence_number'])
    normalized=evidence.json_value(packet)
    # Type-strict comparison preserves the former JSON comparison (including
    # bool/int, int/float and signed zero) without serializing multi-MiB raw
    # audits twice for every received frame. No validation is omitted.
    pending=[(normalized,expected)];equal=True
    while pending:
        left,right=pending.pop()
        if type(left) is not type(right):equal=False;break
        if isinstance(left,dict):
            if left.keys()!=right.keys():equal=False;break
            pending.extend((left[key],right[key]) for key in left)
        elif isinstance(left,list):
            if len(left)!=len(right):equal=False;break
            pending.extend(zip(left,right))
        elif left!=right or (type(left) is float and left==0. and math.copysign(1.,left)!=math.copysign(1.,right)):
            equal=False;break
    if not equal:
        raise ValueError('Scan metadata/ranges/raw returns differ; no free-space or frame repair')
    return packet


def encode_packet(packet):
    normalized=evidence.json_value(validate_packet(packet))
    payload=json.dumps(normalized,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(payload)>MAX_PACKET_BYTES:raise ValueError('Bounded sensor packet size exceeded; no partial scan')
    return payload


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON sensor field; no last-value replacement')
        result[key]=value
    return result


def decode_packet(payload):
    if type(payload) is not bytes or not 0<len(payload)<=MAX_PACKET_BYTES:
        raise ValueError('Bounded complete byte packet required')
    def reject_constant(value):raise ValueError('Nonstandard/nonfinite JSON sensor value: '+value)
    packet=json.loads(payload,object_pairs_hook=_unique_object,parse_constant=reject_constant)
    return validate_packet(packet)


class PacketSequence:
    """No implicit reset, duplicate, future stamp or auto-restart on a new actor."""

    def __init__(self):self.last=None;self.configuration=None

    def accept(self,packet):
        validate_packet(packet)
        return self._accept_immediately_validated(packet)

    def _accept_immediately_validated(self,packet):
        """Private boundary after raw-source reconstruction did ALL checks."""
        if not packet['publishable_without_silent_near_obstacle_loss']:
            raise ValueError('Nonpublishable near-obstacle observation retained; no free-space scan')
        item=(packet['acquisition']['actor_root'],packet['acquisition']['capture_id'],
            packet['sequence_number'],packet['acquisition']['stamp_ns'])
        if self.last is not None:
            if (item[:2]!=self.last[:2] or item[2]<=self.last[2] or item[3]<=self.last[3]
                    or packet['configuration']!=self.configuration):
                raise ValueError('Acquisition identity/sequence/clock changed; no automatic reset or replay')
        self.last=item;self.configuration=copy.deepcopy(packet['configuration'])
        return item


def ros_message_objects(packet):
    """Construct standard ROS objects ONLY; no context/node/publication.

    Call in the matching ROS interpreter, never inject system rclpy into Kit.
    A later publisher must verify exclusive TF/clock ownership and DDS receipt.
    No-return infinity appears only in the ROS float array, never JSON evidence.
    """
    validate_packet(packet)
    return _ros_message_objects_from_validated(packet)


def _ros_message_objects_from_validated(packet):
    """Private immediate receiver fast path AFTER full PacketSequence.accept.

    No await, callback or mutation may occur between verification and this
    conversion. Public callers must use ros_message_objects(), which checks.
    """
    if not packet['publishable_without_silent_near_obstacle_loss']:
        raise ValueError('Blocked sensor capture cannot become a ROS free-space message')
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    from rosgraph_msgs.msg import Clock
    message=LaserScan();scan=packet['scan'];stamp=scan['header']['stamp']
    message.header.stamp.sec=stamp['sec'];message.header.stamp.nanosec=stamp['nanosec']
    message.header.frame_id=scan['header']['frame_id']
    for field in ('angle_min','angle_max','angle_increment','time_increment','scan_time','range_min','range_max'):
        # Keep raw JSON at its original precision, but construct message
        # fields at their declared ROS float32 precision BEFORE CDR encoding.
        setattr(message,field,float(np.float32(scan[field])))
    message.ranges=[math.inf if item['kind']=='NO_RETURN' else float(item['metres']) for item in scan['ranges']]
    message.intensities=[]
    config=packet['configuration'];transform=None
    if config['base_frame_id']!=config['sensor_frame_id']:
        value=TransformStamped();value.header.stamp.sec=stamp['sec'];value.header.stamp.nanosec=stamp['nanosec']
        value.header.frame_id=config['base_frame_id'];value.child_frame_id=config['sensor_frame_id']
        x,y,z=config['sensor_xyz_in_base'];q=config['sensor_wxyz_in_base']
        value.transform.translation.x=float(x);value.transform.translation.y=float(y);value.transform.translation.z=float(z)
        value.transform.rotation.w=float(q[0]);value.transform.rotation.x=float(q[1])
        value.transform.rotation.y=float(q[2]);value.transform.rotation.z=float(q[3])
        transform=TFMessage(transforms=[value])
    clock=Clock();clock.clock.sec=stamp['sec'];clock.clock.nanosec=stamp['nanosec']
    return dict(scan=message,base_to_sensor_tf=transform,clock=clock)


def localization_feedback(record,*,world_from_map_xyyaw,map_frame_id,now_ns,previous_stamp_ns,
        maximum_age_ns,maximum_position_std_m,maximum_heading_std_rad):
    """Explicit fixed-map AMCL-style estimate, never task rebasing or GT pose.

    Covariance/freshness limits are mandatory caller-declared conditions, not
    fitted from an endpoint. This only validates one message for a FUTURE
    controller interface; no reference, goal or actual command is changed.
    """
    if record['header']['frame_id']!=frame_id(map_frame_id):raise ValueError('Estimate must be in the declared fixed map frame')
    if record['source']!='scan_odometry_localization_estimate':raise ValueError('Explicit scan/odometry estimate required, not ground truth relabelled')
    stamp=stamp_value(record['header']['stamp']);stamp_parts(now_ns)
    if (type(maximum_age_ns) is not int or maximum_age_ns<=0 or stamp>now_ns or now_ns-stamp>maximum_age_ns):
        raise ValueError('Estimate stale or future-dated in the simulation clock')
    if previous_stamp_ns is not None:
        stamp_parts(previous_stamp_ns)
        if stamp<=previous_stamp_ns:raise ValueError('Repeated or regressed localization sample; no automatic restart')
    limits=(maximum_position_std_m,maximum_heading_std_rad)
    if not all(type(value) in (float,int) and math.isfinite(float(value)) and value>0. for value in limits):
        raise ValueError('Explicit finite positive uncertainty limits required')
    pose=frames.vector(record['position_xyz_m'],3,'estimated_map_position')
    rotation=frames.rotation_wxyz(record['orientation_wxyz']);yaw=math.atan2(rotation[1,0],rotation[0,0])
    covariance=np.asarray(record['covariance_6x6'],float)
    if (covariance.shape!=(6,6) or not np.isfinite(covariance).all()
            or not np.allclose(covariance,covariance.T,rtol=0.,atol=1e-12)
            or np.linalg.eigvalsh(covariance).min()<-1e-12 or np.any(np.diag(covariance)<0.)):
        raise ValueError('Finite symmetric positive-semidefinite full localization covariance required')
    position_std=math.sqrt(max(0.,float(np.linalg.eigvalsh(covariance[:2,:2]).max())))
    heading_std=math.sqrt(float(covariance[5,5]))
    if position_std>maximum_position_std_m or heading_std>maximum_heading_std_rad:
        raise ValueError('Estimate exceeds prospectively declared uncertainty limits')
    world=frames.transform_pose(world_from_map_xyyaw,[pose[0],pose[1],yaw])
    return dict(version=VERSION,source=record['source'],stamp_ns=stamp,age_ns=now_ns-stamp,
        fixed_map_frame_id=map_frame_id,declared_world_from_map_xyyaw=list(map(float,world_from_map_xyyaw)),
        estimated_world_xyyaw=world,largest_XY_std_m=position_std,yaw_std_rad=heading_std,
        estimate_used_to_reanchor_reference_or_move_task=False,
        ground_truth_pose_substituted=False,localization_or_navigation_success=False,actual_motion_authorized=False)


FEEDBACK_VERSION='HOSPITAL_FIXED_MAP_ASYNCHRONOUS_ESTIMATE_ODOMETRY_SOFTWARE_v1'
FEEDBACK_LIMITS=('maximum_localization_age_ns','maximum_odometry_age_ns',
    'maximum_position_std_m','maximum_heading_std_rad','maximum_odometry_position_std_m',
    'maximum_odometry_heading_std_rad','maximum_linear_velocity_std_m_s','maximum_angular_velocity_std_rad_s')


def _feedback_source(provenance,kind):
    """Validate explicit claims, NOT DDS identity or a sensor measurement."""
    keys={'kind','role','topic','publisher_node','capture_id','actor_root'}
    if (type(provenance) is not dict or set(provenance)!=keys or provenance['kind']!=kind
            or provenance['role'] not in ('SOFTWARE_FIXTURE','UNVALIDATED_ROS_MESSAGE_CAPTURE')):
        raise ValueError('Explicit unvalidated estimate/encoder source provenance required; not ground truth')
    for key in keys-{'kind','role'}:
        value=provenance[key]
        if not isinstance(value,str) or not value or any(c.isspace() for c in value):
            raise ValueError('Exact source/capture identifiers required')
    if (not provenance['actor_root'].startswith('/World/') or provenance['actor_root'].endswith('/')
            or not provenance['topic'].startswith('/') or not provenance['publisher_node'].startswith('/')
            or any(token in provenance['topic'].lower() for token in ('ground_truth','groundtruth'))):
        raise ValueError('Explicit non-ground-truth topic and actor identities required; claims still need live verification')
    return dict(provenance)


def _covariance(value,name):
    matrix=np.asarray(value,dtype=float)
    if (matrix.shape!=(6,6) or not np.isfinite(matrix).all()
            or not np.allclose(matrix,matrix.T,rtol=0.,atol=1e-12)
            or np.any(np.diag(matrix)<0.) or np.linalg.eigvalsh(matrix).min()<-1e-12):
        raise ValueError('Finite symmetric positive-semidefinite6x6 covariance required: '+name)
    return matrix


def _ros_pose_record(header,pose,provenance,kind):
    """Copy actual standard-message fields without importing ROS into Kit."""
    p,q=pose.pose.position,pose.pose.orientation
    covariance=np.asarray(pose.covariance,dtype=float)
    if covariance.shape!=(36,):raise ValueError('ROS row-major36 covariance entries required')
    record=dict(header=dict(stamp=dict(sec=header.stamp.sec,nanosec=header.stamp.nanosec),
        frame_id=header.frame_id),source=kind,position_xyz_m=[p.x,p.y,p.z],
        orientation_wxyz=[q.w,q.x,q.y,q.z],covariance_6x6=covariance.reshape(6,6).tolist(),
        source_provenance=_feedback_source(provenance,kind))
    return record


def _checked_feedback_record(record,kind):
    required={'header','source','position_xyz_m','orientation_wxyz','covariance_6x6','source_provenance'}
    if kind=='wheel_encoder_odometry_estimate':
        required|={'child_frame_id','linear_velocity_base_m_s','angular_velocity_base_rad_s','twist_covariance_6x6'}
    if type(record) is not dict or set(record)!=required or record['source']!=kind:
        raise ValueError('Exact declared estimate/odometry record required; no fallback or invented channels')
    if type(record['header']) is not dict or set(record['header'])!={'stamp','frame_id'}:
        raise ValueError('Exact source header required')
    stamp_value(record['header']['stamp']);frame_id(record['header']['frame_id'])
    _feedback_source(record['source_provenance'],kind)
    frames.vector(record['position_xyz_m'],3,'source position');frames.rotation_wxyz(record['orientation_wxyz'])
    _covariance(record['covariance_6x6'],'pose')
    if kind=='wheel_encoder_odometry_estimate':
        frame_id(record['child_frame_id']);frames.vector(record['linear_velocity_base_m_s'],3,'child-frame linear velocity')
        frames.vector(record['angular_velocity_base_rad_s'],3,'child-frame angular velocity')
        _covariance(record['twist_covariance_6x6'],'twist')
    return evidence.json_value(record)


def localization_record_from_ros(message,*,source_provenance):
    """PoseWithCovarianceStamped -> raw record, no node/TF/clock inference.

    The explicit source claims are retained but not authenticated here. An
    actual capture must separately prove publisher/TF/clock ownership and
    sensor provenance; accepting a Python message is not localization proof.
    """
    kind='scan_odometry_localization_estimate'
    return _checked_feedback_record(_ros_pose_record(message.header,message.pose,source_provenance,kind),kind)


def odometry_record_from_ros(message,*,source_provenance):
    """Odometry pose is in header.frame_id; TWIST is in child_frame_id.

    No pose finite difference, GT substitution or commanded-speed proxy.
    This branch explicitly requires a declared encoder-odometry source;
    numerical covariance is reported metadata, not measured error accuracy.
    """
    kind='wheel_encoder_odometry_estimate';record=_ros_pose_record(message.header,message.pose,source_provenance,kind)
    linear,angular=message.twist.twist.linear,message.twist.twist.angular
    covariance=np.asarray(message.twist.covariance,dtype=float)
    if covariance.shape!=(36,):raise ValueError('ROS row-major36 twist covariance entries required')
    record.update(child_frame_id=message.child_frame_id,linear_velocity_base_m_s=[linear.x,linear.y,linear.z],
        angular_velocity_base_rad_s=[angular.x,angular.y,angular.z],twist_covariance_6x6=covariance.reshape(6,6).tolist())
    return _checked_feedback_record(record,kind)


def odometry_ros_message_objects(record):
    """Encoder record -> Odometry and SAME-stamp odom<-base TF, no node.

    Only construct installed standard messages in their matching interpreter.
    Preserve full pose, CHILD-frame twist and reported covariances; do not
    infer covariance, calibrate a rolling radius, derive velocity from a map
    jump or substitute simulator ground truth. Source claims and TF ownership
    still need live verification; the caller must retain the original record.
    """
    record=_checked_feedback_record(record,'wheel_encoder_odometry_estimate')
    if record['header']['frame_id']==record['child_frame_id']:
        raise ValueError('Distinct encoder odom/base frames required before message construction')
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    message=Odometry();transform=TransformStamped();header=record['header'];stamp=header['stamp']
    for item in (message,transform):
        item.header.frame_id=header['frame_id'];item.header.stamp.sec=stamp['sec'];item.header.stamp.nanosec=stamp['nanosec']
        item.child_frame_id=record['child_frame_id']
    for axis,value in zip(('x','y','z'),record['position_xyz_m']):
        setattr(message.pose.pose.position,axis,float(value));setattr(transform.transform.translation,axis,float(value))
    for axis,value in zip(('w','x','y','z'),record['orientation_wxyz']):
        setattr(message.pose.pose.orientation,axis,float(value));setattr(transform.transform.rotation,axis,float(value))
    for axis,value in zip(('x','y','z'),record['linear_velocity_base_m_s']):setattr(message.twist.twist.linear,axis,float(value))
    for axis,value in zip(('x','y','z'),record['angular_velocity_base_rad_s']):setattr(message.twist.twist.angular,axis,float(value))
    message.pose.covariance=np.asarray(record['covariance_6x6']).ravel().tolist()
    message.twist.covariance=np.asarray(record['twist_covariance_6x6']).ravel().tolist()
    return dict(odometry=message,odom_to_base_tf=TFMessage(transforms=[transform]))


def _xyyaw(record):
    rotation=frames.rotation_wxyz(record['orientation_wxyz'])
    return [*record['position_xyz_m'][:2],math.atan2(rotation[1,0],rotation[0,0])]


def _odometry_reported_uncertainty(record,limits):
    """Check reported metadata, never claim independently measured accuracy."""
    pose_cov=_covariance(record['covariance_6x6'],'encoder pose');twist_cov=_covariance(record['twist_covariance_6x6'],'encoder twist')
    stds=[math.sqrt(max(0.,float(np.linalg.eigvalsh(pose_cov[:2,:2]).max()))),math.sqrt(pose_cov[5,5]),
        math.sqrt(max(0.,float(np.linalg.eigvalsh(twist_cov[:3,:3]).max()))),
        math.sqrt(max(0.,float(np.linalg.eigvalsh(twist_cov[3:,3:]).max())))]
    names=('maximum_odometry_position_std_m','maximum_odometry_heading_std_rad',
        'maximum_linear_velocity_std_m_s','maximum_angular_velocity_std_rad_s')
    if any(value>limits[key] for key,value in zip(names,stds)):
        raise ValueError('Reported odometry covariance exceeds declared limits')


class FixedMapFeedbackStream:
    """Asynchronous AMCL fixes + fresh odometry, PURE and never a controller.

    Each new localization FIX requires an odometry pose at EXACTLY the fix
    acquisition time (lookup/interpolation must be supplied and later proven
    by the capture). Do not pair it with latest odometry or a post-dated AMCL
    TF expiration stamp. Reusing an unchanged fix is allowed within its
    declared age cap; each odometry/control stamp must advance independently.
    map<-odom is computed from the matched fix, not the first task pose.

    A map correction may jump. Its displacement is reported separately and
    NEVER converted to velocity. Actual estimated twist comes only from the
    odometry child frame, rotated with full attitude for world components.
    No covariance independence, current fused accuracy, exclusive TF source,
    actual delivery, live readiness, or motion admission is established here.
    """
    def __init__(self,*,world_from_map_xyyaw,map_frame_id,odom_frame_id,base_frame_id,limits):
        names=[frame_id(v) for v in (map_frame_id,odom_frame_id,base_frame_id)]
        if len(set(names))!=3:raise ValueError('Distinct fixed map/odom/base frames required')
        if type(limits) is not dict or set(limits)!=set(FEEDBACK_LIMITS):
            raise ValueError('All prospective freshness and covariance limits must be explicitly declared')
        for key,value in limits.items():
            if (type(value) not in ((int,) if key.endswith('_ns') else (int,float))
                    or not math.isfinite(value) or value<=0):
                raise ValueError('Explicit positive finite feedback limits required')
        self._configuration=dict(world_from_map_xyyaw=frames.vector(world_from_map_xyyaw,3,'fixed world<-map').tolist(),
            map_frame_id=map_frame_id,odom_frame_id=odom_frame_id,base_frame_id=base_frame_id,limits=copy.deepcopy(limits))
        self._digest=self._configuration_digest();self.previous=None;self.failed=False

    def _configuration_digest(self):
        return hashlib.sha256(json.dumps(self._configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @property
    def configuration(self):return copy.deepcopy(self._configuration)

    def accept(self,localization,odometry_at_fix,current_odometry,*,now_ns):
        if self.failed:raise ValueError('Feedback stream terminal after invalid input; no automatic reset')
        try:return self._accept(localization,odometry_at_fix,current_odometry,now_ns)
        except Exception:
            self.failed=True
            raise

    def _accept(self,localization,odometry_at_fix,current_odometry,now_ns):
        if self._configuration_digest()!=self._digest:raise ValueError('Declared fixed feedback configuration changed')
        config=self._configuration;limits=config['limits'];stamp_parts(now_ns)
        loc=_checked_feedback_record(localization,'scan_odometry_localization_estimate')
        anchor=_checked_feedback_record(odometry_at_fix,'wheel_encoder_odometry_estimate')
        odom=_checked_feedback_record(current_odometry,'wheel_encoder_odometry_estimate')
        loc_stamp=stamp_value(loc['header']['stamp']);anchor_stamp=stamp_value(anchor['header']['stamp'])
        odom_stamp=stamp_value(odom['header']['stamp'])
        if (loc_stamp!=anchor_stamp or not loc_stamp<=odom_stamp<=now_ns
                or now_ns-odom_stamp>limits['maximum_odometry_age_ns']):
            raise ValueError('Matched localization-acquisition odometry and fresh nonfuture current odometry required')
        if loc['header']['frame_id']!=config['map_frame_id'] or any(
                r['header']['frame_id']!=config['odom_frame_id'] or r['child_frame_id']!=config['base_frame_id'] for r in (anchor,odom)):
            raise ValueError('Exact declared map<-odom<-base frame chain required, not swapped pose/twist frames')
        identity=lambda r:tuple(r['source_provenance'][k] for k in ('actor_root','capture_id','role'))
        if not identity(loc)==identity(anchor)==identity(odom) or anchor['source_provenance']!=odom['source_provenance']:
            raise ValueError('Same declared actor/capture and encoder source required')
        previous=self.previous;new_fix=previous is None or loc_stamp>previous['fix_stamp_ns']
        if previous is not None:
            if now_ns<previous['now_ns'] or odom_stamp<=previous['odometry_stamp_ns'] or loc_stamp<previous['fix_stamp_ns']:
                raise ValueError('Odometry/control clock must advance; no duplicate tick, old fix or restart')
            if (loc['source_provenance']!=previous['localization']['source_provenance']
                    or odom['source_provenance']!=previous['odometry']['source_provenance']):
                raise ValueError('Source/capture identity changed; no implicit new actor')
            if not new_fix and (loc!=previous['localization'] or anchor!=previous['odometry_at_fix']):
                raise ValueError('Repeated fix stamp must retain EXACT original fix and matched odometry')
        feedback=localization_feedback(loc,world_from_map_xyyaw=config['world_from_map_xyyaw'],
            map_frame_id=config['map_frame_id'],now_ns=now_ns,previous_stamp_ns=None,
            maximum_age_ns=limits['maximum_localization_age_ns'],maximum_position_std_m=limits['maximum_position_std_m'],
            maximum_heading_std_rad=limits['maximum_heading_std_rad'])
        for record in (anchor,odom):
            _odometry_reported_uncertainty(record,limits)
        map_from_odom=frames.transform_pose(_xyyaw(loc),frames.transform_pose(_xyyaw(anchor),[0.,0.,0.],inverse=True))
        map_pose=frames.transform_pose(map_from_odom,_xyyaw(odom));world=frames.transform_pose(config['world_from_map_xyyaw'],map_pose)
        correction_step=[0.,0.]
        if previous is not None:
            old_map_pose=frames.transform_pose(previous['map_from_odom_xyyaw'],_xyyaw(odom))
            correction_step=[math.hypot(map_pose[0]-old_map_pose[0],map_pose[1]-old_map_pose[1]),frames.wrap(map_pose[2]-old_map_pose[2])]
        angle=config['world_from_map_xyyaw'][2]+map_from_odom[2];c,s=math.cos(angle),math.sin(angle)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@frames.rotation_wxyz(odom['orientation_wxyz'])
        world_velocity=rotation@np.asarray(odom['linear_velocity_base_m_s']);world_angular=rotation@np.asarray(odom['angular_velocity_base_rad_s'])
        result=evidence.json_value(dict(version=FEEDBACK_VERSION,status='SOFTWARE_FEEDBACK_TRANSFORM_NOT_LIVE_READINESS',
            configuration=self.configuration,configuration_sha256=self._digest,localization_fix_stamp_ns=loc_stamp,
            localization_fix_age_ns=now_ns-loc_stamp,new_localization_fix=new_fix,odometry_stamp_ns=odom_stamp,
            odometry_age_ns=now_ns-odom_stamp,map_from_odom_xyyaw=map_from_odom,estimated_world_xyyaw=world,
            estimated_linear_velocity_world_m_s=world_velocity.tolist(),estimated_angular_velocity_world_rad_s=world_angular.tolist(),
            estimated_horizontal_speed_m_s=math.hypot(*world_velocity[:2]),
            estimated_signed_base_forward_speed_m_s=odom['linear_velocity_base_m_s'][0],
            map_correction_step_m_rad=correction_step,localization_fix_quality_at_its_own_stamp=feedback,
            raw_localization=loc,raw_odometry_at_fix=anchor,raw_current_odometry=odom,
            source_and_exact_timestamp_interpolation_claims_need_live_verification=True,
            covariance_channels_are_reported_NOT_independent_measured_error_or_fused_current_accuracy=True,
            map_correction_jump_used_as_velocity=False,first_estimate_used_to_reanchor_task=False,
            ground_truth_fallback=False,actual_motion_authorized=False,sensor_localization_Nav2_success=False,new_trial_count=0))
        # Commit only after every check and representation conversion passed.
        self.previous=copy.deepcopy(dict(localization=loc,odometry_at_fix=anchor,odometry=odom,fix_stamp_ns=loc_stamp,
            odometry_stamp_ns=odom_stamp,now_ns=now_ns,map_from_odom_xyyaw=map_from_odom))
        return result


class MatchedFeedbackInbox:
    """Bounded exact-stamp odometry history for delayed localization fixes.

    PURE callback plumbing: append genuine source-stamped odometry, then ask
    for feedback using a separately supplied localization message and current
    simulation clock. Missing future odometry is PENDING, not an empty/free
    observation or an invented new estimate. A historical missing/pruned
    anchor is HOLD, never latest/nearest odometry or pose interpolation.

    Exact scan/encoder acquisition alignment is an explicit FUTURE publisher
    requirement, not proof supplied by this class. A previously accepted fix
    keeps its ORIGINAL anchor even after bounded-history eviction. Freshness,
    covariance, source and frame checks remain those of FixedMapFeedbackStream.
    Invalid input is terminal, without commanding Stop/Pause or any recovery.
    No node, DDS, current SDK, controller, ground-truth fallback or live gate.
    """
    def __init__(self,*,feedback_configuration,maximum_records,maximum_history_ns):
        if (type(maximum_records) is not int or not 2<=maximum_records<=10000
                or type(maximum_history_ns) is not int or not 0<maximum_history_ns<=MAX_STAMP_NS):
            raise ValueError('Explicit bounded odometry-history count and simulation duration required')
        self._stream=FixedMapFeedbackStream(**feedback_configuration)
        self._config=dict(feedback_configuration=self._stream.configuration,
            maximum_records=maximum_records,maximum_history_ns=maximum_history_ns)
        self._digest=self._configuration_digest();self._records=[];self._evicted=0
        self._now=None;self._localization=None;self.failed=False

    def _configuration_digest(self):
        return hashlib.sha256(json.dumps(self._config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @property
    def configuration(self):return copy.deepcopy(self._config)

    @property
    def history(self):return copy.deepcopy(self._records)

    def _check_clock(self,now_ns):
        if self.failed:raise ValueError('Feedback inbox terminal; no automatic reset or stale feedback fallback')
        if self._digest!=self._configuration_digest() or self._stream.configuration!=self._config['feedback_configuration']:
            raise ValueError('Fixed feedback/history configuration changed')
        stamp_parts(now_ns)
        if self._now is not None and now_ns<self._now:raise ValueError('Simulation arrival/control clock regressed')

    def append_odometry(self,record,*,now_ns):
        try:
            self._check_clock(now_ns);record=_checked_feedback_record(record,'wheel_encoder_odometry_estimate')
            config=self._config['feedback_configuration'];stamp=stamp_value(record['header']['stamp'])
            if (record['header']['frame_id']!=config['odom_frame_id'] or record['child_frame_id']!=config['base_frame_id']
                    or stamp>now_ns or now_ns-stamp>self._config['maximum_history_ns']):
                raise ValueError('Exact frames and nonfuture in-history odometry acquisition stamp required')
            _odometry_reported_uncertainty(record,config['limits'])
            if self._records:
                previous=self._records[-1]
                if stamp<=stamp_value(previous['header']['stamp']) or record['source_provenance']!=previous['source_provenance']:
                    raise ValueError('Advancing odometry stamps and unchanged source/capture required')
            # A pending fix already declares its capture, even before the first
            # odometry arrives. Do not switch actors while waiting for a pair.
            if self._localization is not None:
                keys=('actor_root','capture_id','role')
                if any(record['source_provenance'][k]!=self._localization['source_provenance'][k] for k in keys):
                    raise ValueError('Odometry/localization capture identity differs')
            updated=[r for r in self._records if stamp-stamp_value(r['header']['stamp'])<=self._config['maximum_history_ns']]
            updated.append(record);updated=updated[-self._config['maximum_records']:]
            evicted=len(self._records)+1-len(updated)
            self._records=updated;self._evicted+=evicted;self._now=now_ns
            return dict(stamp_ns=stamp,retained_records=len(updated),evicted_records_this_append=evicted,
                total_evicted_records=self._evicted,source_is_NOT_live_authenticated=True,actual_motion_authorized=False)
        except Exception:
            self.failed=True
            raise

    def _result(self,status,*,feedback=None):
        return evidence.json_value(dict(version='HOSPITAL_EXACT_STAMP_FEEDBACK_INBOX_SOFTWARE_v1',status=status,
            configuration_sha256=self._digest,retained_odometry_records=len(self._records),
            total_evicted_records=self._evicted,feedback=feedback,
            usable_feedback_sample=feedback is not None,
            latest_or_nearest_anchor_substituted=False,interpolated_pose_or_covariance_invented=False,
            acquisition_alignment_and_transport_source_NOT_live_verified=True,
            pending_or_HOLD_does_NOT_authorize_reusing_old_commands=True,
            ground_truth_fallback=False,actual_motion_authorized=False,sensor_localization_Nav2_success=False,new_trial_count=0))

    def feedback(self,localization,*,now_ns):
        try:return self._feedback(localization,now_ns)
        except Exception:
            self.failed=True
            raise

    def _feedback(self,localization,now_ns):
        self._check_clock(now_ns);config=self._config['feedback_configuration'];limits=config['limits']
        loc=_checked_feedback_record(localization,'scan_odometry_localization_estimate')
        loc_stamp=stamp_value(loc['header']['stamp'])
        localization_feedback(loc,world_from_map_xyyaw=config['world_from_map_xyyaw'],map_frame_id=config['map_frame_id'],
            now_ns=now_ns,previous_stamp_ns=None,maximum_age_ns=limits['maximum_localization_age_ns'],
            maximum_position_std_m=limits['maximum_position_std_m'],maximum_heading_std_rad=limits['maximum_heading_std_rad'])
        if self._localization is not None:
            old_stamp=stamp_value(self._localization['header']['stamp'])
            if (loc['source_provenance']!=self._localization['source_provenance'] or loc_stamp<old_stamp
                    or loc_stamp==old_stamp and loc!=self._localization):
                raise ValueError('Advancing-or-identical original localization and unchanged source required')
        if self._records:
            latest=self._records[-1];stamp=stamp_value(latest['header']['stamp']);keys=('actor_root','capture_id','role')
            if any(loc['source_provenance'][k]!=latest['source_provenance'][k] for k in keys):
                raise ValueError('Localization/odometry capture identity differs')
            if now_ns-stamp>limits['maximum_odometry_age_ns']:
                raise ValueError('Latest retained odometry is stale; no old feedback fallback')
        self._localization=loc;self._now=now_ns
        if not self._records:return self._result('PENDING_ODOMETRY_NOT_A_FEEDBACK_SAMPLE')
        previous=self._stream.previous
        if previous is not None and previous['fix_stamp_ns']==loc_stamp:
            anchor=previous['odometry_at_fix']
        else:
            anchor=next((r for r in self._records if stamp_value(r['header']['stamp'])==loc_stamp),None)
        if anchor is None:
            if loc_stamp>stamp:return self._result('PENDING_EXACT_ANCHOR_NOT_YET_CAPTURED')
            self.failed=True
            return self._result('HOLD_EXACT_ANCHOR_MISSING_OR_PRUNED_NO_INTERPOLATION')
        if previous is not None and stamp==previous['odometry_stamp_ns']:
            return self._result('PENDING_NEW_ODOMETRY_NOT_A_DUPLICATE_FEEDBACK_SAMPLE')
        result=self._stream.accept(loc,anchor,latest,now_ns=now_ns)
        return self._result('EXACT_ANCHOR_FEEDBACK_SOFTWARE_SAMPLE_NOT_LIVE_READINESS',feedback=result)


RAY_MAP_VERSION='HOSPITAL_RAY_EVIDENCE_LOCALIZATION_MAP_REVIEW_v1'


def _ray_map_cells(start,end,*,origin,resolution,shape,tolerance,remaining):
    """Ray-line INTERIOR traversals and closed endpoint cells, not area proof.

    A ray exactly on a grid line does not clear cells on either side. A hit
    on a boundary marks every adjacent closed cell. No rounding/clipping of
    an OUTSIDE hit into a border obstacle. Traversal work is bounded BEFORE
    allocating crossing arrays; budget failure aborts the complete packet.
    """
    start=np.asarray(start,float);end=np.asarray(end,float);delta=end-start
    height,width=shape;lo=np.asarray(origin,float);hi=lo+resolution*np.array([width,height])
    first,last=0.,1.
    for axis in (0,1):
        if delta[axis]==0.:
            if not lo[axis]<start[axis]<hi[axis]:first,last=1.,0.;break
        else:
            a,b=sorted(((lo[axis]-start[axis])/delta[axis],(hi[axis]-start[axis])/delta[axis]))
            first=max(first,a);last=min(last,b)
    crossings=[];ranges=[]
    if first<last:
        for axis,size in ((0,width),(1,height)):
            if delta[axis]==0.:continue
            a,b=sorted(((start[axis]+first*delta[axis]-lo[axis])/resolution,
                (start[axis]+last*delta[axis]-lo[axis])/resolution))
            ranges.append((axis,range(max(0,math.ceil(a)),min(size,math.floor(b))+1)))
        estimate=sum(len(indices) for _,indices in ranges)+1
        if estimate>remaining:raise ValueError('Whole map packet exceeds declared cell-traversal budget; no partial integration')
        crossings=[first,last]
        for axis,indices in ranges:
            crossings.extend((lo[axis]+index*resolution-start[axis])/delta[axis] for index in indices)
        crossings=sorted(set(value for value in crossings if first<=value<=last))
    free=set();work=max(0,len(crossings)-1)
    for a,b in zip(crossings,crossings[1:]):
        point=start+((a+b)/2.)*delta;scaled=(point-lo)/resolution
        if any(abs(v-round(v))*resolution<=tolerance for v in scaled):continue
        col,row=map(math.floor,scaled)
        if 0<=row<height and 0<=col<width:free.add(row*width+col)
    occupied=set()
    if np.all(end>=lo-tolerance) and np.all(end<=hi+tolerance):
        choices=[]
        for value,size in zip((end-lo)/resolution,(width,height)):
            nearest=round(float(value))
            candidates=(nearest-1,nearest) if abs(value-nearest)*resolution<=tolerance else (math.floor(value),)
            choices.append([i for i in candidates if 0<=i<size])
        occupied={row*width+col for col in choices[0] for row in choices[1]}
    return free,occupied,work


class RayEvidenceMap:
    """PURE scan-derived -1/0/100 map candidate, NEVER the planning mask.

    Uses the same origin and FIRST external return of each validated ALL-hit
    packet. A free value means ray-line traversal under this raster model,
    NOT that the whole cell/robot footprint is safe. Unseen space stays -1.
    Registered simulator/virtual query poses are DISCLOSED, not sensor-only
    SLAM. STATIC_VIRTUAL_QUERY permits equal REAL stamps without forging time;
    TIME_ADVANCING_SENSOR requires strictly increasing acquisition stamps.

    No actual query, map publication, ROS node, scene/control edit or live
    provenance proof. Conditions, map epoch and sensor plane are mandatory.
    """
    def __init__(self,*,map_frame_id,world_from_map_xyyaw,origin_xy_m,shape_rows_columns,resolution_m,
            boundary_tolerance_m,environment_sha256,capture_mode,reference_sensor_world_z_m,
            maximum_sensor_height_error_m,maximum_ray_tilt_rad,maximum_cell_visits):
        frame_id(map_frame_id);world=frames.vector(world_from_map_xyyaw,3,'declared fixed world<-map').tolist()
        origin=frames.vector(origin_xy_m,2,'map bottom-left origin').tolist();shape=list(shape_rows_columns)
        if (len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape) or math.prod(shape)>4_000_000
                or type(resolution_m) not in (int,float) or not math.isfinite(resolution_m) or resolution_m<=0.
                or not 0.<float(np.float32(resolution_m))<math.inf):raise ValueError('Bounded explicit metric raster required')
        limits=(boundary_tolerance_m,maximum_sensor_height_error_m,maximum_ray_tilt_rad)
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0. for v in limits)
                or boundary_tolerance_m>=resolution_m/4. or maximum_ray_tilt_rad>=math.pi/2.
                or type(reference_sensor_world_z_m) not in (int,float) or not math.isfinite(reference_sensor_world_z_m)
                or type(maximum_cell_visits) is not int or not 0<maximum_cell_visits<=2_000_000):
            raise ValueError('Explicit bounded numerical/sensor-plane/traversal conditions required')
        if (not isinstance(environment_sha256,str) or len(environment_sha256)!=64
                or any(v not in '0123456789abcdef' for v in environment_sha256)
                or capture_mode not in ('STATIC_VIRTUAL_QUERY','TIME_ADVANCING_SENSOR')):
            raise ValueError('Exact declared environment epoch and capture mode required')
        self._config=dict(map_frame_id=map_frame_id,world_from_map_xyyaw=world,origin_xy_m=origin,
            shape_rows_columns=shape,resolution_m=float(resolution_m),boundary_tolerance_m=float(boundary_tolerance_m),
            environment_sha256=environment_sha256,capture_mode=capture_mode,reference_sensor_world_z_m=float(reference_sensor_world_z_m),
            maximum_sensor_height_error_m=float(maximum_sensor_height_error_m),maximum_ray_tilt_rad=float(maximum_ray_tilt_rad),
            maximum_cell_visits=maximum_cell_visits)
        self._digest=self._configuration_digest();self._cells=np.full(shape,-1,dtype=np.int8)
        self._identity=self._source_config=self._last=self._loaded=None;self._receipts=[];self.failed=False

    def _configuration_digest(self):
        return hashlib.sha256(json.dumps(self._config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @property
    def configuration(self):return copy.deepcopy(self._config)

    @property
    def cells(self):return self._cells.copy()

    def accept(self,packet):
        if self.failed:raise ValueError('Ray map terminal after invalid packet; no automatic recovery')
        try:return self._accept(packet)
        except Exception:
            self.failed=True
            raise

    def _accept(self,packet):
        validate_packet(packet);config=self._config
        if self._digest!=self._configuration_digest():raise ValueError('Fixed map definition changed')
        if not packet['publishable_without_silent_near_obstacle_loss']:raise ValueError('Near external hit blocks WHOLE map packet, not free space')
        acquisition=packet['acquisition'];source=packet['configuration'];stamp=acquisition['stamp_ns'];sequence=packet['sequence_number']
        if acquisition.get('environment_sha256')!=config['environment_sha256']:raise ValueError('Exact declared map environment epoch required')
        identity=tuple(acquisition[k] for k in ('actor_root','capture_id','source_role'))
        if config['map_frame_id'] in (source['base_frame_id'],source['sensor_frame_id']):raise ValueError('Fixed map frame must not be a moving sensor/base frame')
        if self._last is not None:
            if identity!=self._identity or source!=self._source_config or sequence<=self._last[0]:
                raise ValueError('Same map source/capture/configuration and advancing packet sequence required')
            valid=stamp==self._last[1] if config['capture_mode']=='STATIC_VIRTUAL_QUERY' else stamp>self._last[1]
            if not valid:raise ValueError('Declared static snapshot/sensor acquisition clock contract changed; no invented timestamp')
        geometry=scan_geometry(acquisition,source)
        if abs(geometry['sensor_origin_world_m'][2]-config['reference_sensor_world_z_m'])>config['maximum_sensor_height_error_m']:
            raise ValueError('Sensor origin outside declared reference-plane height; no world-height override')
        free=set();occupied=set();work=0
        for ray,decoded in zip(geometry['rays'],packet['decoded_rays']):
            if abs(ray['direction_world'][2])>math.sin(config['maximum_ray_tilt_rad']):
                raise ValueError('Nonplanar ray exceeds declared map plane; no silent flattening')
            hit=decoded['nearest_external_hit'];distance=ray['length_m'] if hit is None else hit['distance_m']
            origin=np.asarray(ray['origin_world_m']);end=origin+distance*np.asarray(ray['direction_world'])
            start_map=frames.transform_pose(config['world_from_map_xyyaw'],[*origin[:2],0.],inverse=True)[:2]
            end_map=frames.transform_pose(config['world_from_map_xyyaw'],[*end[:2],0.],inverse=True)[:2]
            f,o,visited=_ray_map_cells(start_map,end_map,origin=config['origin_xy_m'],resolution=config['resolution_m'],
                shape=config['shape_rows_columns'],tolerance=config['boundary_tolerance_m'],remaining=config['maximum_cell_visits']-work)
            work+=visited
            if work>config['maximum_cell_visits']:raise ValueError('Whole map packet exceeds traversal budget')
            free.update(f)
            if hit is not None:occupied.update(o)
        updated=self._cells.copy();flat=updated.reshape(-1);free_indices=list(free);occupied_indices=list(occupied)
        if free_indices:
            indices=np.array(free_indices);flat[indices[flat[indices]!=100]]=0
        if occupied_indices:flat[occupied_indices]=100
        # Validate/encode exact packet before committing the new map epoch.
        digest=hashlib.sha256(encode_packet(packet)).hexdigest()
        receipt=dict(sequence_number=sequence,stamp_ns=stamp,packet_sha256=digest,rays=len(geometry['rays']),
            ray_cell_visits=work,ray_traversed_cell_count=len(free),first_external_hit_cell_count=len(occupied))
        self._cells=updated;self._identity=identity;self._source_config=copy.deepcopy(source)
        self._last=(sequence,stamp);self._loaded=stamp if self._loaded is None else self._loaded;self._receipts.append(receipt)
        return copy.deepcopy(receipt)

    def record(self):
        if self.failed or self._last is None:raise ValueError('No healthy ray-map capture available for message encoding')
        if self._digest!=self._configuration_digest():raise ValueError('Fixed map definition changed')
        return evidence.json_value(dict(version=RAY_MAP_VERSION,role='LOCALIZATION_RAY_MAP_CANDIDATE_NOT_PLANNING_OR_LIVE_ADMISSION',
            configuration=self.configuration,configuration_sha256=self._digest,header=dict(frame_id=self._config['map_frame_id'],stamp=stamp_parts(self._last[1])),
            map_load_stamp=stamp_parts(self._loaded),source_identity=dict(zip(('actor_root','capture_id','source_role'),self._identity)),
            source_configuration=copy.deepcopy(self._source_config),
            data=self._cells.ravel(order='C').tolist(),data_sha256=hashlib.sha256(self._cells.tobytes()).hexdigest(),
            cell_counts={str(v):int(np.count_nonzero(self._cells==v)) for v in (-1,0,100)},receipts=copy.deepcopy(self._receipts),
            free_means_ray_line_traversal_NOT_full_cell_or_footprint_safety=True,
            map_registration_uses_supplied_simulator_or_virtual_pose_NOT_sensor_only_SLAM=True,
            source_and_environment_provenance_NOT_live_verified=True,planning_support_or_safety_mask_NOT_substituted=True,
            actual_map_adopted=False,actual_sensor_localization_Nav2_success=False,actual_motion_authorized=False,new_PhysX_trials=0))


def occupancy_grid_message(record):
    """Standard -1/0/100 row-major message ONLY; never a node or admission."""
    if (record.get('version')!=RAY_MAP_VERSION or record.get('role')!='LOCALIZATION_RAY_MAP_CANDIDATE_NOT_PLANNING_OR_LIVE_ADMISSION'
            or any(record.get(k) is not False for k in ('actual_map_adopted','actual_sensor_localization_Nav2_success','actual_motion_authorized'))):
        raise ValueError('Own unadopted scan-ray map required, not a planning-mask or success relabel')
    config=record['configuration'];checked=RayEvidenceMap(**config)
    if (checked.configuration!=config or checked._digest!=record['configuration_sha256']
            or record['header']['frame_id']!=config['map_frame_id']):raise ValueError('Map frame/configuration changed')
    rows,columns=config['shape_rows_columns'];data=record['data']
    if (type(data) is not list or len(data)!=rows*columns or any(type(v) is not int or v not in (-1,0,100) for v in data)
            or hashlib.sha256(np.asarray(data,np.int8).tobytes()).hexdigest()!=record['data_sha256']):
        raise ValueError('Exact trinary row-major ray map required; no unknown-to-free/image flip repair')
    stamp=stamp_value(record['header']['stamp']);loaded=stamp_value(record['map_load_stamp'])
    receipts=record['receipts']
    if not isinstance(receipts,list) or not receipts:raise ValueError('Explicit nonempty original ray capture receipts required')
    last_sequence=-1;last_stamp=None
    for item in receipts:
        sequence=item['sequence_number'];observed=item['stamp_ns'];stamp_parts(observed)
        digest=item['packet_sha256']
        if (type(sequence) is not int or sequence<=last_sequence or not isinstance(digest,str) or len(digest)!=64
                or any(ch not in '0123456789abcdef' for ch in digest)
                or type(item['rays']) is not int or item['rays']<=0
                or any(type(item[k]) is not int or item[k]<0 for k in
                    ('ray_cell_visits','ray_traversed_cell_count','first_external_hit_cell_count'))
                or item['ray_cell_visits']>config['maximum_cell_visits']):
            raise ValueError('Canonical bounded ray capture receipt required')
        if last_stamp is not None:
            valid=observed==last_stamp if config['capture_mode']=='STATIC_VIRTUAL_QUERY' else observed>last_stamp
            if not valid:raise ValueError('Ray capture clock contract changed')
        last_sequence,last_stamp=sequence,observed
    if loaded!=receipts[0]['stamp_ns'] or stamp!=last_stamp:raise ValueError('Map timestamps must match original ray receipts')
    if record['cell_counts']!={str(v):data.count(v) for v in (-1,0,100)}:raise ValueError('Map cell counts changed')
    identity=record['source_identity'];source=record['source_configuration']
    # Validate the SAME explicit source declaration without inventing a capture.
    scan_geometry(dict(**identity,stamp_ns=stamp,base_xyz=[0.,0.,0.],base_wxyz=[1.,0.,0.,0.]),source)
    if config['map_frame_id'] in (source['base_frame_id'],source['sensor_frame_id']):raise ValueError('Map frame cannot move with sensor/base')
    if (any(record.get(k) is not True for k in ('free_means_ray_line_traversal_NOT_full_cell_or_footprint_safety',
            'map_registration_uses_supplied_simulator_or_virtual_pose_NOT_sensor_only_SLAM',
            'source_and_environment_provenance_NOT_live_verified','planning_support_or_safety_mask_NOT_substituted'))
            or type(record.get('new_PhysX_trials')) is not int or record['new_PhysX_trials']!=0):
        raise ValueError('Unvalidated ray-map scope must remain explicit')
    from nav_msgs.msg import OccupancyGrid
    message=OccupancyGrid();message.header.frame_id=frame_id(config['map_frame_id'])
    message.header.stamp.sec,message.header.stamp.nanosec=divmod(stamp,10**9)
    message.info.map_load_time.sec,message.info.map_load_time.nanosec=divmod(loaded,10**9)
    message.info.width=columns;message.info.height=rows;message.info.resolution=float(np.float32(config['resolution_m']))
    message.info.origin.position.x=float(config['origin_xy_m'][0]);message.info.origin.position.y=float(config['origin_xy_m'][1])
    message.info.origin.orientation.w=1.;message.data=data
    return message

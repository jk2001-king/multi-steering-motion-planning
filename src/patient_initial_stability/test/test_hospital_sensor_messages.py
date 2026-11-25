"""Software-only scan/estimate and optional installed ROS CDR serialization."""

import copy
import json
import math
import sys

import numpy as np
import pytest

import hospital_sensor_messages as messages


def scan_inputs():
    acquisition=dict(stamp_ns=12345678901,actor_root='/World/SOFTWARE_FIXTURE_ACTOR',
        base_xyz=[1.,2.,-.003],base_wxyz=[1.,0.,0.,0.],source_role='SOFTWARE_FIXTURE',capture_id='SOFTWARE_ONLY_NOT_AN_EXPERIMENT')
    config=dict(base_frame_id='fixture_base',sensor_frame_id='fixture_laser',sensor_xyz_in_base=[.1,0.,.35],
        sensor_wxyz_in_base=[1.,0.,0.,0.],ray_count=4,angle_min_rad=-math.pi,angle_increment_rad=math.pi/2,
        range_min_m=.1,range_max_m=12.,rate_hz=5.,noise_model='ideal_native_raycast_no_noise',
        own_collision_paths=['/World/SOFTWARE_FIXTURE_ACTOR/base/shape'])
    rays=messages.scan_geometry(acquisition,config)['rays']
    def hit(i,distance,path='/World/EXTERNAL_FIXTURE'):
        ray=rays[i];point=np.asarray(ray['origin_world_m'])+distance*np.asarray(ray['direction_world'])
        return dict(collision=path,distance_m=distance,position_m=point.tolist(),normal=[0.,0.,1.])
    records=[dict(index=i,hits=[]) for i in range(4)]
    records[1]['hits']=[hit(1,.15,config['own_collision_paths'][0]),hit(1,2.)]
    records[2]['hits']=[hit(2,.3,'/World/SOFTWARE_FIXTURE_ACTOR/UNREVIEWED_SHAPE'),hit(2,1.)]
    return acquisition,config,records,hit


def packet():
    acquisition,config,records,_=scan_inputs()
    return messages.scan_packet(acquisition,config,records,0)


def test_first_return_preserves_occluded_bad_barycentric_position_without_using_it():
    a,c,records,hit=scan_inputs();c['return_model']=messages.FIRST_RETURN_MODEL
    hidden=hit(1,11.,'/World/OCCLUDED_FIXTURE');hidden['position_m'][2]+=.00035
    records[1]['hits'].insert(0,hidden);original=copy.deepcopy(records)
    result=messages.scan_packet(a,c,records,0)
    assert records==original
    assert result['scan']['ranges'][1]==dict(kind='RANGE',metres=2.)
    assert result['raw_records'][1]['hits'][0]==hidden
    audit=result['decoded_rays'][1]['occluded_raw_return_audit'][0]
    assert audit['NOT_used_as_observed_first_return'] and audit['raw_position_NOT_repaired']
    assert messages.decode_packet(messages.encode_packet(result))==result
    c.pop('return_model')
    with pytest.raises(ValueError,match='off-ray'):messages.scan_packet(a,c,records,0)


@pytest.mark.parametrize('kind',['nearest','tie','own','nonfinite','normal','path','below_min'])
def test_first_return_never_waives_visible_or_malformed_return(kind):
    a,c,records,hit=scan_inputs();c['return_model']=messages.FIRST_RETURN_MODEL
    bad=hit(1,11.,'/World/OCCLUDED_FIXTURE')
    if kind=='nearest':bad=hit(1,1.)
    if kind=='tie':bad=hit(1,2.)
    if kind=='own':bad=hit(1,11.,c['own_collision_paths'][0])
    if kind in ('nearest','tie','own'):bad['position_m'][2]+=.001
    if kind=='nonfinite':bad['distance_m']=float('nan')
    if kind=='normal':bad['normal']=[0.,0.,0.]
    if kind=='path':bad['collision']=''
    if kind=='below_min':bad=hit(1,.05)
    records[1]['hits'].append(bad)
    if kind=='below_min':
        result=messages.scan_packet(a,c,records,0)
        assert not result['publishable_without_silent_near_obstacle_loss']
        assert result['scan']['ranges'][1]['kind']=='BLOCKED_BELOW_MIN'
    else:
        with pytest.raises(ValueError):messages.scan_packet(a,c,records,0)


def estimate():
    return dict(header=dict(stamp=dict(sec=12,nanosec=345678901),frame_id='fixture_map'),
        source='scan_odometry_localization_estimate',position_xyz_m=[1.,2.,0.],
        orientation_wxyz=[1.,0.,0.,0.],covariance_6x6=np.diag([.0001,.0001,0.,0.,0.,.0001]).tolist())


def feedback(record=None,**kwargs):
    arguments=dict(world_from_map_xyyaw=[10.,20.,math.pi/2],map_frame_id='fixture_map',now_ns=12445678901,
        previous_stamp_ns=None,maximum_age_ns=200000000,maximum_position_std_m=.02,maximum_heading_std_rad=.02)
    arguments.update(kwargs)
    return messages.localization_feedback(estimate() if record is None else record,**arguments)


def test_raw_complete_scan_and_exact_self_hits_survive_JSON_roundtrip_without_nonfinite_values():
    p=packet();before=copy.deepcopy(p);encoded=messages.encode_packet(p)
    assert messages.decode_packet(encoded)==p and p==before
    assert b'Infinity' not in encoded and b'NaN' not in encoded
    assert p['scan']['ranges']==[dict(kind='NO_RETURN'),dict(kind='RANGE',metres=2.),
        dict(kind='RANGE',metres=.3),dict(kind='NO_RETURN')]
    assert len(p['decoded_rays'][1]['raw_hits'])==2 and len(p['decoded_rays'][1]['exact_self_hits'])==1
    assert p['decoded_rays'][2]['nearest_external_hit']['collision'].endswith('UNREVIEWED_SHAPE')
    assert p['scan']['time_increment']==0. and p['scan']['scan_time']==.2
    assert p['scan']['angle_max']==pytest.approx(math.pi/2)
    assert p['scan']['intensities']==[] and p['scan']['header']['stamp']==dict(sec=12,nanosec=345678901)
    assert not p['actual_sensor_success'] and not p['localization_or_Nav2_success'] and not p['actual_motion_authorized']


def test_supported_native_numpy_fields_are_normalized_before_writing_not_after_a_trial():
    a,c,records,_=scan_inputs()
    a['base_xyz']=np.asarray(a['base_xyz'],dtype=np.float32)
    a['base_wxyz']=np.asarray(a['base_wxyz'],dtype=np.float32)
    c['sensor_xyz_in_base']=np.asarray(c['sensor_xyz_in_base'],dtype=np.float32)
    c['sensor_wxyz_in_base']=np.asarray(c['sensor_wxyz_in_base'],dtype=np.float32)
    for key in ('angle_min_rad','angle_increment_rad','range_min_m','range_max_m','rate_hz'):
        c[key]=np.float64(c[key])
    geometry=messages.scan_geometry(a,c)
    for ray,record in zip(geometry['rays'],records):
        for hit in record['hits']:
            hit['distance_m']=np.float64(hit['distance_m'])
            hit['position_m']=np.asarray(ray['origin_world_m'])+hit['distance_m']*np.asarray(ray['direction_world'])
            hit['normal']=np.asarray(hit['normal'],dtype=np.float32)
            hit['additional_native_metadata']='SOFTWARE_METADATA_RETAINED'
    p=messages.scan_packet(a,c,records,0)
    assert messages.decode_packet(messages.encode_packet(p))==p
    assert p['acquisition']['base_xyz']==a['base_xyz'].astype(float).tolist()
    assert p['raw_records'][1]['hits'][0]['additional_native_metadata']=='SOFTWARE_METADATA_RETAINED'
    assert not p['actual_sensor_success']


def test_native_vector_wrappers_do_not_need_pickle_or_deepcopy_to_preserve_their_components():
    class UncopyableNativeVector:
        def __init__(self,values):self.values=values
        def __array__(self,dtype=None):return np.asarray(self.values,dtype=dtype)
        def __deepcopy__(self,memo):raise RuntimeError('SOFTWARE_NATIVE_VECTOR_CANNOT_BE_PICKLED')
    a,c,records,_=scan_inputs()
    a['base_xyz']=UncopyableNativeVector(a['base_xyz'])
    a['base_wxyz']=UncopyableNativeVector(a['base_wxyz'])
    c['sensor_xyz_in_base']=UncopyableNativeVector(c['sensor_xyz_in_base'])
    c['sensor_wxyz_in_base']=UncopyableNativeVector(c['sensor_wxyz_in_base'])
    for record in records:
        for hit in record['hits']:
            hit['position_m']=UncopyableNativeVector(hit['position_m'])
            hit['normal']=UncopyableNativeVector(hit['normal'])
    p=messages.scan_packet(a,c,records,0)
    assert messages.decode_packet(messages.encode_packet(p))==p
    assert p['acquisition']['base_xyz']==[1.,2.,-.003]


def test_near_external_obstacle_is_durably_encodable_but_never_publishable_as_free_space():
    acquisition,config,records,hit=scan_inputs();records[0]['hits']=[hit(0,.05)]
    p=messages.scan_packet(acquisition,config,records,0)
    assert messages.decode_packet(messages.encode_packet(p))==p
    assert p['blocked_ray_indices']==[0] and p['scan']['ranges'][0]==dict(kind='BLOCKED_BELOW_MIN',metres=.05)
    before=set(sys.modules)
    with pytest.raises(ValueError,match='Blocked'):messages.ros_message_objects(p)
    assert not any(name in sys.modules and name not in before for name in ('rclpy','sensor_msgs','omni.physx'))
    with pytest.raises(ValueError,match='Nonpublishable'):messages.PacketSequence().accept(p)


@pytest.mark.parametrize('fault',['missing','duplicate','order','index_bool','nan','offray','nonunit','self_prefix','frame','same_frame','role','noise','time'])
def test_bad_acquisition_or_hits_never_become_a_valid_empty_scan(fault):
    a,c,records,hit=scan_inputs()
    if fault=='missing':records.pop()
    elif fault=='duplicate':records[1]['index']=0
    elif fault=='order':records.reverse()
    elif fault=='index_bool':records[0]['index']=False
    elif fault=='nan':records[0]['hits']=[hit(0,1.)];records[0]['hits'][0]['distance_m']=float('nan')
    elif fault=='offray':records[0]['hits']=[hit(0,1.)];records[0]['hits'][0]['position_m'][2]+=.01
    elif fault=='nonunit':a['base_wxyz']=[2.,0.,0.,0.]
    elif fault=='self_prefix':c['own_collision_paths']=['/World/OTHER_ACTOR/shape']
    elif fault=='frame':c['sensor_frame_id']='/laser'
    elif fault=='same_frame':c['sensor_frame_id']=c['base_frame_id']
    elif fault=='role':a['source_role']='VALIDATED_SENSOR_SUCCESS'
    elif fault=='noise':c['noise_model']='UNTESTED_GAUSSIAN'
    else:a['stamp_ns']=float(a['stamp_ns'])
    with pytest.raises((ValueError,TypeError)):messages.scan_packet(a,c,records,0)


@pytest.mark.parametrize('fault',['range','timing','frame','TF','success','version','raw','boolean_type'])
def test_packet_metadata_cannot_disagree_with_raw_observations_or_relabel_success(fault):
    p=packet()
    if fault=='range':p['scan']['ranges'][2]=dict(kind='NO_RETURN')
    elif fault=='timing':p['scan']['time_increment']=.001
    elif fault=='frame':p['scan']['header']['frame_id']='base_link'
    elif fault=='TF':p['configuration']['sensor_xyz_in_base'][0]+=.1
    elif fault=='success':p['actual_sensor_success']=True
    elif fault=='version':p['version']='UNREVIEWED'
    elif fault=='raw':p['raw_records'][1]['hits']=[]
    else:p['actual_motion_authorized']=0
    with pytest.raises(ValueError):messages.encode_packet(p)


@pytest.mark.parametrize('payload',[b'{"kind":"A","kind":"B"}',b'{"data":NaN}',b'{"data":Infinity}',b'',b'{}',b'\xff'])
def test_malformed_duplicate_nonfinite_or_empty_wire_records_fail_closed(payload):
    with pytest.raises((ValueError,UnicodeDecodeError)):messages.decode_packet(payload)


def test_packet_size_is_bounded_before_evidence_or_message_publication(monkeypatch):
    monkeypatch.setattr(messages,'MAX_PACKET_BYTES',32)
    with pytest.raises(ValueError,match='size exceeded'):messages.encode_packet(packet())


@pytest.mark.parametrize('field,value',[('range_max_m',1e40),('range_min_m',1e-60),('angle_increment_rad',1e-60)])
def test_float32_wire_overflow_or_underflow_never_changes_a_finite_observation_into_valid_free_space(field,value):
    a,c,records,_=scan_inputs();c[field]=value
    # Empty explicit rays isolate the wire-format check, not a fabricated
    # native measurement or off-ray failure from the old synthetic geometry.
    records=[dict(index=i,hits=[]) for i in range(4)]
    with pytest.raises(ValueError,match='float32'):messages.scan_packet(a,c,records,0)


def test_sequence_clock_and_actor_cannot_implicitly_restart_or_duplicate():
    sequence=messages.PacketSequence();p=packet();sequence.accept(p);before=sequence.last
    with pytest.raises(ValueError):sequence.accept(p)
    assert sequence.last==before
    for field in ('time','actor','capture'):
        a,c,records,_=scan_inputs();a['stamp_ns']+=200000000
        if field=='time':a['stamp_ns']=before[3]
        elif field=='actor':
            a['actor_root']='/World/NEW_FIXTURE';c['own_collision_paths']=['/World/NEW_FIXTURE/base/shape']
        else:a['capture_id']='NEW_UNREVIEWED_CAPTURE'
        changed=messages.scan_packet(a,c,records,1)
        with pytest.raises(ValueError):sequence.accept(changed)
        assert sequence.last==before
    a,c,records,_=scan_inputs();a['stamp_ns']+=200000000
    sequence.accept(messages.scan_packet(a,c,records,1));assert sequence.last[2:]==(1,a['stamp_ns'])
    a['stamp_ns']+=200000000;c['rate_hz']=10.
    with pytest.raises(ValueError):sequence.accept(messages.scan_packet(a,c,records,2))


def test_estimate_maps_to_fixed_world_without_moving_route_and_uses_full_XY_covariance():
    record=estimate();before=copy.deepcopy(record);result=feedback(record)
    assert record==before and result['estimated_world_xyyaw']==pytest.approx([8.,21.,math.pi/2])
    assert result['age_ns']==100000000 and result['largest_XY_std_m']==.01
    assert not result['estimate_used_to_reanchor_reference_or_move_task'] and not result['ground_truth_pose_substituted']
    assert not result['localization_or_navigation_success'] and not result['actual_motion_authorized']
    record['covariance_6x6'][0][1]=record['covariance_6x6'][1][0]=.00009
    assert feedback(record)['largest_XY_std_m']==pytest.approx(math.sqrt(.00019))
    with pytest.raises(ValueError,match='uncertainty'):feedback(record,maximum_position_std_m=.012)


@pytest.mark.parametrize('fault',['frame','GT','future','old','duplicate','clock_regression','covariance_shape','asymmetry','negative','nonfinite','uncertainty','quaternion','limit'])
def test_bad_or_stale_localization_cannot_fall_back_to_GT_or_reanchor_the_task(fault):
    r=estimate();kwargs={}
    if fault=='frame':r['header']['frame_id']='odom'
    elif fault=='GT':r['source']='simulator_ground_truth'
    elif fault=='future':kwargs['now_ns']=12345678900
    elif fault=='old':kwargs['now_ns']=13000000000
    elif fault=='duplicate':kwargs['previous_stamp_ns']=12345678901
    elif fault=='clock_regression':kwargs['previous_stamp_ns']=12445678901
    elif fault=='covariance_shape':r['covariance_6x6']=[0.]*36
    elif fault=='asymmetry':r['covariance_6x6'][0][1]=.00001
    elif fault=='negative':r['covariance_6x6'][5][5]=-.0001
    elif fault=='nonfinite':r['position_xyz_m'][0]=float('nan')
    elif fault=='uncertainty':kwargs['maximum_heading_std_rad']=.001
    elif fault=='quaternion':r['orientation_wxyz']=[0.,0.,0.,0.]
    else:kwargs['maximum_age_ns']=True
    with pytest.raises((ValueError,TypeError)):feedback(r,**kwargs)


def test_nanosecond_stamps_roundtrip_without_float_time_or_unit_guessing():
    for value in (0,1,999999999,1000000000,12345678901,messages.MAX_STAMP_NS):
        assert messages.stamp_value(messages.stamp_parts(value))==value
    for value in (-1,True,1.,messages.MAX_STAMP_NS+1):
        with pytest.raises(ValueError):messages.stamp_parts(value)
    for value in (dict(sec=0,nanosec=1000000000),dict(sec=-1,nanosec=1),dict(sec=0,nanosec=True)):
        with pytest.raises(ValueError):messages.stamp_value(value)


@pytest.mark.parametrize('sensor_yaw',[0.,math.pi/3])
def test_installed_ROS_message_CDR_roundtrip_without_initializing_a_node_or_DDS(sensor_yaw):
    serialization=pytest.importorskip('rclpy.serialization',reason='Matching ROS Python ABI is not on this offline interpreter path')
    import rclpy
    assert not rclpy.ok()
    a,c,records,_=scan_inputs()
    c['sensor_wxyz_in_base']=[math.cos(sensor_yaw/2),0.,0.,math.sin(sensor_yaw/2)]
    geometry=messages.scan_geometry(a,c)
    # Rotate the synthetic ALL-hit points along with the sensor, not the
    # actual environment, robot or a physical observation.
    for ray,record in zip(geometry['rays'],records):
        for hit in record['hits']:
            hit['position_m']=(np.asarray(ray['origin_world_m'])+hit['distance_m']*np.asarray(ray['direction_world'])).tolist()
    p=messages.scan_packet(a,c,records,0);before=copy.deepcopy(p);objects=messages.ros_message_objects(p)
    for name,original in objects.items():
        assert original is not None
        restored=serialization.deserialize_message(serialization.serialize_message(original),type(original))
        assert restored==original
        if name=='scan':
            assert restored.header.stamp.sec==12 and restored.header.stamp.nanosec==345678901
            assert restored.header.frame_id=='fixture_laser' and restored.time_increment==0.
            assert len(restored.ranges)==4 and math.isinf(restored.ranges[0]) and restored.ranges[2]==pytest.approx(.3)
            assert restored.angle_max==pytest.approx(restored.angle_min+3*restored.angle_increment,abs=2e-7)
            assert len(restored.intensities)==0
        if name=='base_to_sensor_tf':
            tf=restored.transforms[0]
            assert tf.header.frame_id=='fixture_base' and tf.child_frame_id=='fixture_laser'
            assert tf.transform.translation.x==.1 and tf.transform.translation.z==.35
            assert tf.transform.rotation.w==math.cos(sensor_yaw/2) and tf.transform.rotation.x==0.
            assert tf.transform.rotation.z==math.sin(sensor_yaw/2)
    assert not rclpy.ok(), 'No context/node/publisher may be initialized by this codec test'
    assert p==before, 'ROS float32 quantization must not rewrite the original raw JSON observations'


def feedback_stream_inputs():
    """Invented SOFTWARE data, never native observations or a driving trial."""
    provenance=lambda kind,topic,node:dict(kind=kind,role='SOFTWARE_FIXTURE',topic=topic,publisher_node=node,
        capture_id='SOFTWARE_ONLY_NOT_A_TRIAL',actor_root='/World/SOFTWARE_FIXTURE_ACTOR')
    loc=estimate();loc['header']['stamp']=messages.stamp_parts(1000000000)
    loc['source_provenance']=provenance(loc['source'],'/fixture/amcl_pose','/fixture/amcl')
    loc['orientation_wxyz']=[math.cos(.3),0.,0.,math.sin(.3)]
    anchor=copy.deepcopy(loc);anchor.update(source='wheel_encoder_odometry_estimate',child_frame_id='fixture_base',
        position_xyz_m=[4.,-1.,.01],orientation_wxyz=[math.cos(-.1),0.,0.,math.sin(-.1)],
        linear_velocity_base_m_s=[-.05,.01,-.002],angular_velocity_base_rad_s=[.001,.002,-.03],
        twist_covariance_6x6=np.diag([1e-6]*6).tolist())
    anchor['header']['frame_id']='fixture_odom'
    anchor['source_provenance']=provenance(anchor['source'],'/fixture/encoder_odom','/fixture/encoder')
    current=copy.deepcopy(anchor);current['header']['stamp']=messages.stamp_parts(1100000000)
    current['position_xyz_m'][0]+=.01
    limits=dict(maximum_localization_age_ns=2000000000,maximum_odometry_age_ns=200000000,
        maximum_position_std_m=.02,maximum_heading_std_rad=.02,maximum_odometry_position_std_m=.1,
        maximum_odometry_heading_std_rad=.1,maximum_linear_velocity_std_m_s=.1,maximum_angular_velocity_std_rad_s=.1)
    config=dict(world_from_map_xyyaw=[10.,20.,math.pi/2],map_frame_id='fixture_map',
        odom_frame_id='fixture_odom',base_frame_id='fixture_base',limits=limits)
    return loc,anchor,current,config


def _software_planar_matrix(pose):
    x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,x],[s,c,y],[0.,0.,1.]])


@pytest.mark.parametrize('world_yaw',[-2.9,0.,.7,math.pi/2,math.pi])
def test_async_feedback_matches_independent_homogeneous_composition_not_first_estimate_reanchor(world_yaw):
    loc,anchor,odom,config=feedback_stream_inputs();config['world_from_map_xyyaw'][2]=world_yaw
    before=copy.deepcopy((loc,anchor,odom,config));stream=messages.FixedMapFeedbackStream(**config)
    result=stream.accept(loc,anchor,odom,now_ns=1150000000)
    expected=_software_planar_matrix(config['world_from_map_xyyaw'])@_software_planar_matrix([1.,2.,.6])@np.linalg.inv(
        _software_planar_matrix([4.,-1.,-.2]))@_software_planar_matrix([4.01,-1.,-.2])
    assert result['estimated_world_xyyaw']==pytest.approx([expected[0,2],expected[1,2],math.atan2(expected[1,0],expected[0,0])])
    yaw=world_yaw+.6;c,s=math.cos(yaw),math.sin(yaw)
    expected_velocity=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@np.array([-.05,.01,-.002])
    assert result['estimated_linear_velocity_world_m_s']==pytest.approx(expected_velocity)
    assert result['estimated_signed_base_forward_speed_m_s']==-.05
    assert result['estimated_horizontal_speed_m_s']==pytest.approx(math.hypot(.05,.01))
    assert result['odometry_age_ns']==50000000 and result['localization_fix_age_ns']==150000000
    assert result['new_localization_fix'] and not result['first_estimate_used_to_reanchor_task']
    assert not result['actual_motion_authorized'] and not result['sensor_localization_Nav2_success']
    assert not result['ground_truth_fallback'] and result['new_trial_count']==0
    assert result['source_and_exact_timestamp_interpolation_claims_need_live_verification']
    assert result['covariance_channels_are_reported_NOT_independent_measured_error_or_fused_current_accuracy']
    assert (loc,anchor,odom,config)==before and result['raw_current_odometry']==odom
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_same_fix_can_drive_fresh_odom_ticks_without_forged_new_localization_or_time_reset():
    loc,anchor,odom,config=feedback_stream_inputs();stream=messages.FixedMapFeedbackStream(**config)
    first=stream.accept(loc,anchor,odom,now_ns=1100000000)
    odom['header']['stamp']=messages.stamp_parts(1200000000);odom['position_xyz_m'][0]+=.02
    second=stream.accept(loc,anchor,odom,now_ns=1200000000)
    assert first['new_localization_fix'] and not second['new_localization_fix']
    assert second['localization_fix_stamp_ns']==first['localization_fix_stamp_ns']==1000000000
    assert second['localization_fix_age_ns']==200000000 and second['odometry_stamp_ns']==1200000000
    assert math.dist(second['estimated_world_xyyaw'][:2],first['estimated_world_xyyaw'][:2])==pytest.approx(.02)
    assert second['map_correction_step_m_rad']==pytest.approx([0.,0.])
    assert second['estimated_horizontal_speed_m_s']==first['estimated_horizontal_speed_m_s']


def test_localization_correction_jump_is_not_reported_as_physical_velocity_or_task_reset():
    loc,anchor,odom,config=feedback_stream_inputs();stream=messages.FixedMapFeedbackStream(**config)
    first=stream.accept(loc,anchor,odom,now_ns=1100000000)
    # Identical odometric pose/velocity but a new map localization fix.
    odom['header']['stamp']=messages.stamp_parts(1200000000);anchor=copy.deepcopy(odom)
    loc['header']['stamp']=messages.stamp_parts(1200000000)
    loc['position_xyz_m']=[2.,2.,0.]
    second=stream.accept(loc,anchor,odom,now_ns=1200000000)
    assert second['new_localization_fix'] and second['map_correction_step_m_rad'][0]>.9
    assert second['estimated_horizontal_speed_m_s']==pytest.approx(first['estimated_horizontal_speed_m_s'])
    assert not second['map_correction_jump_used_as_velocity'] and not second['first_estimate_used_to_reanchor_task']
    assert second['raw_localization']==loc and second['localization_fix_age_ns']==0


def feedback_inbox(**overrides):
    *_,config=feedback_stream_inputs()
    arguments=dict(feedback_configuration=config,maximum_records=4,maximum_history_ns=1000000000)
    arguments.update(overrides)
    return messages.MatchedFeedbackInbox(**arguments)


def test_feedback_inbox_pairs_delayed_fix_with_original_acquisition_not_latest_odometry():
    loc,anchor,current,config=feedback_stream_inputs();inbox=feedback_inbox()
    before=copy.deepcopy((loc,anchor,current,config))
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    result=inbox.feedback(loc,now_ns=1150000000)
    expected=messages.FixedMapFeedbackStream(**config).accept(loc,anchor,current,now_ns=1150000000)
    assert result['feedback']==expected and result['usable_feedback_sample']
    assert result['feedback']['raw_odometry_at_fix']==anchor
    assert result['feedback']['raw_current_odometry']==current
    assert result['feedback']['estimated_world_xyyaw']!=messages._xyyaw(loc)
    assert not result['latest_or_nearest_anchor_substituted'] and not result['interpolated_pose_or_covariance_invented']
    assert not result['actual_motion_authorized'] and not result['sensor_localization_Nav2_success']
    assert (loc,anchor,current,config)==before


def test_feedback_inbox_expected_callback_wait_is_not_terminal_or_duplicate_feedback():
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    assert inbox.feedback(loc,now_ns=1000000000)['status']=='PENDING_ODOMETRY_NOT_A_FEEDBACK_SAMPLE'
    assert not inbox.failed
    inbox.append_odometry(anchor,now_ns=1000000000)
    assert inbox.feedback(loc,now_ns=1000000000)['usable_feedback_sample']
    waiting=inbox.feedback(loc,now_ns=1050000000)
    assert waiting['status']=='PENDING_NEW_ODOMETRY_NOT_A_DUPLICATE_FEEDBACK_SAMPLE'
    assert waiting['feedback'] is None and not waiting['usable_feedback_sample'] and not inbox.failed
    inbox.append_odometry(current,now_ns=1100000000)
    result=inbox.feedback(loc,now_ns=1150000000)
    assert result['usable_feedback_sample'] and not result['feedback']['new_localization_fix']
    assert result['feedback']['localization_fix_stamp_ns']==1000000000


def test_feedback_inbox_waits_for_future_anchor_without_retiming_any_measurement():
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.append_odometry(anchor,now_ns=1000000000)
    loc['header']['stamp']=messages.stamp_parts(1100000000)
    waiting=inbox.feedback(loc,now_ns=1100000000)
    assert waiting['status']=='PENDING_EXACT_ANCHOR_NOT_YET_CAPTURED' and not inbox.failed
    inbox.append_odometry(current,now_ns=1100000000)
    result=inbox.feedback(loc,now_ns=1100000000)
    assert result['feedback']['raw_odometry_at_fix']==current
    assert result['feedback']['localization_fix_stamp_ns']==1100000000


@pytest.mark.parametrize('reason',['between_samples','pruned_count','pruned_time'])
def test_feedback_inbox_missing_historical_anchor_is_HOLD_not_nearest_or_interpolated_pose(reason):
    loc,anchor,current,_=feedback_stream_inputs()
    inbox=feedback_inbox(maximum_records=2,maximum_history_ns=50000000 if reason=='pruned_time' else 1000000000)
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    if reason=='between_samples':loc['header']['stamp']=messages.stamp_parts(1050000000)
    if reason=='pruned_count':
        next_record=copy.deepcopy(current);next_record['header']['stamp']=messages.stamp_parts(1200000000)
        inbox.append_odometry(next_record,now_ns=1200000000)
    result=inbox.feedback(loc,now_ns=1200000000)
    assert result['status']=='HOLD_EXACT_ANCHOR_MISSING_OR_PRUNED_NO_INTERPOLATION'
    assert result['feedback'] is None and not result['usable_feedback_sample'] and inbox.failed
    with pytest.raises(ValueError,match='terminal'):inbox.feedback(loc,now_ns=1200000000)


def test_feedback_inbox_accepted_fix_retains_original_anchor_after_bounded_history_eviction():
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox(maximum_records=2)
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    first=inbox.feedback(loc,now_ns=1100000000)
    for index in range(2,6):
        stamp=1000000000+index*100000000;record=copy.deepcopy(current)
        record['header']['stamp']=messages.stamp_parts(stamp);record['position_xyz_m'][0]+=.01*index
        receipt=inbox.append_odometry(record,now_ns=stamp)
    assert len(inbox.history)==2 and receipt['total_evicted_records']==4
    result=inbox.feedback(loc,now_ns=1500000000)
    assert result['feedback']['raw_odometry_at_fix']==first['feedback']['raw_odometry_at_fix']==anchor
    assert not result['feedback']['new_localization_fix'] and result['feedback']['localization_fix_age_ns']==500000000
    assert result['feedback']['raw_current_odometry']==record


@pytest.mark.parametrize('fault',['duplicate','regression','future','arrival_clock','source','frame','child','covariance','config','capture_after_pending'])
def test_feedback_inbox_invalid_odometry_is_terminal_before_partial_history_commit(fault):
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.append_odometry(anchor,now_ns=1000000000);before=inbox.history
    now=1100000000
    if fault=='duplicate':current=copy.deepcopy(anchor)
    elif fault=='regression':current['header']['stamp']=messages.stamp_parts(900000000)
    elif fault=='future':now=1050000000
    elif fault=='arrival_clock':now=999999999
    elif fault=='source':current['source_provenance']['publisher_node']='/fixture/different'
    elif fault=='frame':current['header']['frame_id']='fixture_map'
    elif fault=='child':current['child_frame_id']='fixture_odom'
    elif fault=='covariance':current['covariance_6x6'][0][0]=1.
    elif fault=='config':inbox._config['maximum_records']=100
    else:
        inbox.feedback(loc,now_ns=1000000000)
        current['source_provenance']['capture_id']='DIFFERENT_CAPTURE'
    with pytest.raises(ValueError):inbox.append_odometry(current,now_ns=now)
    assert inbox.failed and inbox.history==before
    with pytest.raises(ValueError,match='terminal'):inbox.append_odometry(anchor,now_ns=now)


@pytest.mark.parametrize('fault',['stale_odom','stale_fix','changed_same_fix','old_fix','source','capture','GT','frame','future','clock'])
def test_feedback_inbox_invalid_localization_cannot_reuse_old_feedback_or_current_as_anchor(fault):
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    inbox.feedback(loc,now_ns=1100000000);before=inbox.history;now=1150000000
    if fault=='stale_odom':now=1400000000
    elif fault=='stale_fix':now=3100000000
    elif fault=='changed_same_fix':loc['position_xyz_m'][0]+=.01
    elif fault=='old_fix':loc['header']['stamp']=messages.stamp_parts(900000000)
    elif fault=='source':loc['source_provenance']['publisher_node']='/different/amcl'
    elif fault=='capture':loc['source_provenance']['capture_id']='DIFFERENT_CAPTURE'
    elif fault=='GT':loc['source']='simulator_ground_truth'
    elif fault=='frame':loc['header']['frame_id']='fixture_odom'
    elif fault=='future':loc['header']['stamp']=messages.stamp_parts(1200000000)
    else:now=1000000000
    with pytest.raises(ValueError):inbox.feedback(loc,now_ns=now)
    assert inbox.failed and inbox.history==before
    with pytest.raises(ValueError,match='terminal'):inbox.feedback(loc,now_ns=now)


@pytest.mark.parametrize('argument,value',[('maximum_records',True),('maximum_records',1),('maximum_records',10001),
    ('maximum_history_ns',True),('maximum_history_ns',0),('maximum_history_ns',1.)])
def test_feedback_inbox_explicit_bounded_history_settings_required(argument,value):
    with pytest.raises(ValueError):feedback_inbox(**{argument:value})


def test_feedback_inbox_copied_inputs_configuration_and_outputs_cannot_move_original_anchor():
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    anchor['position_xyz_m'][0]+=10.;current['position_xyz_m'][0]+=10.
    config=inbox.configuration;config['feedback_configuration']['world_from_map_xyyaw'][0]+=20.
    history=inbox.history;history[0]['position_xyz_m'][0]+=30.
    first=inbox.feedback(loc,now_ns=1100000000)
    first['feedback']['raw_odometry_at_fix']['position_xyz_m'][0]+=40.
    _,original,next_record,_=feedback_stream_inputs();next_record['header']['stamp']=messages.stamp_parts(1200000000)
    inbox.append_odometry(next_record,now_ns=1200000000)
    second=inbox.feedback(loc,now_ns=1200000000)
    assert second['feedback']['raw_odometry_at_fix']==original
    assert second['feedback']['configuration']['world_from_map_xyyaw']==[10.,20.,math.pi/2]
    assert inbox.configuration['feedback_configuration']['world_from_map_xyyaw']==[10.,20.,math.pi/2]


def test_feedback_inbox_new_fix_on_consumed_odometry_tick_waits_then_uses_exact_new_anchor():
    loc,anchor,current,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.append_odometry(anchor,now_ns=1000000000);inbox.append_odometry(current,now_ns=1100000000)
    first=inbox.feedback(loc,now_ns=1100000000)
    loc['header']['stamp']=messages.stamp_parts(1100000000);loc['position_xyz_m'][0]+=.5
    waiting=inbox.feedback(loc,now_ns=1100000000)
    assert waiting['feedback'] is None and not inbox.failed
    next_record=copy.deepcopy(current);next_record['header']['stamp']=messages.stamp_parts(1200000000)
    inbox.append_odometry(next_record,now_ns=1200000000)
    result=inbox.feedback(loc,now_ns=1200000000)['feedback']
    assert result['new_localization_fix'] and result['raw_odometry_at_fix']==current
    assert result['map_correction_step_m_rad'][0]>.4
    assert result['estimated_horizontal_speed_m_s']==first['feedback']['estimated_horizontal_speed_m_s']
    assert not result['map_correction_jump_used_as_velocity']


def test_feedback_inbox_pending_fix_identity_cannot_switch_before_first_odometry():
    loc,anchor,_,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.feedback(loc,now_ns=1000000000)
    anchor['source_provenance']['actor_root']='/World/DIFFERENT_FIXTURE'
    with pytest.raises(ValueError,match='capture identity'):inbox.append_odometry(anchor,now_ns=1000000000)
    assert inbox.failed and inbox.history==[]


def test_feedback_inbox_same_pending_fix_cannot_change_without_a_new_source_stamp():
    loc,_,_,_=feedback_stream_inputs();inbox=feedback_inbox()
    inbox.feedback(loc,now_ns=1000000000);loc['position_xyz_m'][0]+=.01
    with pytest.raises(ValueError,match='identical original'):inbox.feedback(loc,now_ns=1000000000)
    assert inbox.failed and inbox.history==[]


def test_world_twist_rotation_retains_full_odometry_roll_not_a_yaw_only_sensor_or_velocity_flattening():
    loc,anchor,odom,config=feedback_stream_inputs()
    # Pure roll at current stamp; yaw of anchor remains independently -0.2.
    roll=.4;odom['orientation_wxyz']=[math.cos(roll/2),math.sin(roll/2),0.,0.]
    result=messages.FixedMapFeedbackStream(**config).accept(loc,anchor,odom,now_ns=1100000000)
    angle=math.pi/2+.8;c,s=math.cos(angle),math.sin(angle)
    rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    rx=np.array([[1.,0.,0.],[0.,math.cos(roll),-math.sin(roll)],[0.,math.sin(roll),math.cos(roll)]])
    assert result['estimated_linear_velocity_world_m_s']==pytest.approx(rz@rx@odom['linear_velocity_base_m_s'])
    assert result['estimated_angular_velocity_world_rad_s']==pytest.approx(rz@rx@odom['angular_velocity_base_rad_s'])


@pytest.mark.parametrize('fault',['stale_fix','stale_odom','future_odom','wrong_fix_stamp','map_frame','odom_frame','child_frame',
    'ground_truth_source','ground_truth_topic','capture','actor','publisher','role','changed_same_fix','changed_same_anchor',
    'duplicate_tick','clock_regression','old_fix','shape','asymmetry','negative','nonfinite_twist','quaternion',
    'velocity_uncertainty','position_uncertainty','localization_uncertainty','unknown_channel'])
def test_async_feedback_bad_input_is_terminal_without_commit_reanchor_fallback_or_auto_retry(fault):
    loc,anchor,odom,config=feedback_stream_inputs();stream=messages.FixedMapFeedbackStream(**config)
    stream.accept(loc,anchor,odom,now_ns=1100000000);before=copy.deepcopy(stream.previous)
    odom['header']['stamp']=messages.stamp_parts(1200000000);now=1200000000
    if fault=='stale_fix':now=3100000000;odom['header']['stamp']=messages.stamp_parts(now)
    elif fault=='stale_odom':now=1500000000
    elif fault=='future_odom':odom['header']['stamp']=messages.stamp_parts(now+1)
    elif fault=='wrong_fix_stamp':anchor['header']['stamp']=messages.stamp_parts(1000000001)
    elif fault=='map_frame':loc['header']['frame_id']='fixture_odom'
    elif fault=='odom_frame':odom['header']['frame_id']='fixture_map'
    elif fault=='child_frame':odom['child_frame_id']='fixture_laser'
    elif fault=='ground_truth_source':odom['source']='simulator_ground_truth'
    elif fault=='ground_truth_topic':odom['source_provenance']['topic']='/sim/ground_truth/odom'
    elif fault=='capture':odom['source_provenance']['capture_id']='NEW_UNREVIEWED_CAPTURE'
    elif fault=='actor':odom['source_provenance']['actor_root']='/World/OTHER'
    elif fault=='publisher':loc['source_provenance']['publisher_node']='/different_localizer'
    elif fault=='role':odom['source_provenance']['role']='VERIFIED_SENSOR_SUCCESS'
    elif fault=='changed_same_fix':loc['position_xyz_m'][0]+=.01
    elif fault=='changed_same_anchor':anchor['position_xyz_m'][0]+=.01
    elif fault=='duplicate_tick':odom['header']['stamp']=messages.stamp_parts(1100000000)
    elif fault=='clock_regression':now=1099999999
    elif fault=='old_fix':loc['header']['stamp']=anchor['header']['stamp']=messages.stamp_parts(999999999)
    elif fault=='shape':odom['covariance_6x6']=[0.]*36
    elif fault=='asymmetry':odom['covariance_6x6'][0][1]=.00001
    elif fault=='negative':odom['twist_covariance_6x6'][0][0]=-.0001
    elif fault=='nonfinite_twist':odom['linear_velocity_base_m_s'][0]=math.nan
    elif fault=='quaternion':odom['orientation_wxyz']=[2.,0.,0.,0.]
    elif fault=='velocity_uncertainty':odom['twist_covariance_6x6'][0][0]=1.
    elif fault=='position_uncertainty':odom['covariance_6x6'][0][0]=1.
    elif fault=='localization_uncertainty':loc['covariance_6x6'][0][0]=1.
    else:odom['unreviewed_new_field']='not silently ignored'
    with pytest.raises((ValueError,TypeError)):stream.accept(loc,anchor,odom,now_ns=now)
    assert stream.previous==before and stream.failed
    loc,anchor,odom,_=feedback_stream_inputs();odom['header']['stamp']=messages.stamp_parts(1300000000)
    with pytest.raises(ValueError,match='terminal'):stream.accept(loc,anchor,odom,now_ns=1300000000)


@pytest.mark.parametrize('fault',['missing_limit','bool_limit','zero_limit','nan_limit','frames','transform'])
def test_feedback_stream_has_no_implicit_frames_mount_age_covariance_or_live_authority_defaults(fault):
    *_,config=feedback_stream_inputs()
    if fault=='missing_limit':config['limits'].pop('maximum_odometry_age_ns')
    elif fault=='bool_limit':config['limits']['maximum_localization_age_ns']=True
    elif fault=='zero_limit':config['limits']['maximum_heading_std_rad']=0.
    elif fault=='nan_limit':config['limits']['maximum_position_std_m']=math.nan
    elif fault=='frames':config['odom_frame_id']=config['map_frame_id']
    else:config['world_from_map_xyyaw'][0]=math.inf
    with pytest.raises(ValueError):messages.FixedMapFeedbackStream(**config)


def test_feedback_configuration_copy_cannot_move_fixed_map_and_internal_mutation_is_detected():
    loc,anchor,odom,config=feedback_stream_inputs();stream=messages.FixedMapFeedbackStream(**config)
    edited=stream.configuration;edited['world_from_map_xyyaw'][0]+=100.
    assert stream.configuration==config
    stream._configuration['world_from_map_xyyaw'][0]+=100.
    with pytest.raises(ValueError,match='configuration changed'):stream.accept(loc,anchor,odom,now_ns=1100000000)


def test_actual_installed_AMCL_and_odometry_CDR_decode_to_feedback_records_without_node_DDS_or_Kit():
    serialization=pytest.importorskip('rclpy.serialization',reason='Requires matching ROS Python3.10 ABI, never loaded into Kit')
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    assert not rclpy.ok()
    loc,anchor,odom,config=feedback_stream_inputs()
    def wire(record,message_type):
        message=message_type();h=record['header'];message.header.frame_id=h['frame_id']
        message.header.stamp.sec=h['stamp']['sec'];message.header.stamp.nanosec=h['stamp']['nanosec']
        p,q=message.pose.pose.position,message.pose.pose.orientation
        p.x,p.y,p.z=record['position_xyz_m'];q.w,q.x,q.y,q.z=record['orientation_wxyz']
        message.pose.covariance=np.asarray(record['covariance_6x6']).ravel().tolist()
        if message_type is Odometry:
            message.child_frame_id=record['child_frame_id'];linear,angular=message.twist.twist.linear,message.twist.twist.angular
            linear.x,linear.y,linear.z=record['linear_velocity_base_m_s'];angular.x,angular.y,angular.z=record['angular_velocity_base_rad_s']
            message.twist.covariance=np.asarray(record['twist_covariance_6x6']).ravel().tolist()
        return serialization.deserialize_message(serialization.serialize_message(message),message_type)
    raw_loc=messages.localization_record_from_ros(wire(loc,PoseWithCovarianceStamped),source_provenance=loc['source_provenance'])
    raw_anchor=messages.odometry_record_from_ros(wire(anchor,Odometry),source_provenance=anchor['source_provenance'])
    raw_odom=messages.odometry_record_from_ros(wire(odom,Odometry),source_provenance=odom['source_provenance'])
    assert raw_loc==loc and raw_anchor==anchor and raw_odom==odom
    result=messages.FixedMapFeedbackStream(**config).accept(raw_loc,raw_anchor,raw_odom,now_ns=1150000000)
    assert not result['sensor_localization_Nav2_success'] and not result['actual_motion_authorized'] and not rclpy.ok()
    inbox=messages.MatchedFeedbackInbox(feedback_configuration=config,maximum_records=4,maximum_history_ns=1000000000)
    inbox.append_odometry(raw_anchor,now_ns=1000000000);inbox.append_odometry(raw_odom,now_ns=1100000000)
    assert inbox.feedback(raw_loc,now_ns=1150000000)['feedback']==result and not rclpy.ok()


def test_installed_encoder_odometry_and_TF_message_factory_retains_full_pose_twist_clock_covariance():
    serialization=pytest.importorskip('rclpy.serialization',reason='Matching ROS Python ABI unavailable; never load system ROS into Kit')
    import rclpy
    assert not rclpy.ok()
    _,_,record,_=feedback_stream_inputs();record['orientation_wxyz']=[math.cos(.2),math.sin(.2),0.,0.]
    before=copy.deepcopy(record);objects=messages.odometry_ros_message_objects(record)
    decoded={key:serialization.deserialize_message(serialization.serialize_message(value),type(value)) for key,value in objects.items()}
    back=messages.odometry_record_from_ros(decoded['odometry'],source_provenance=record['source_provenance'])
    assert back==record and record==before and not rclpy.ok()
    transform=decoded['odom_to_base_tf'].transforms[0];odom=decoded['odometry']
    assert transform.header==odom.header and transform.child_frame_id==odom.child_frame_id
    assert [getattr(transform.transform.translation,axis) for axis in ('x','y','z')]==record['position_xyz_m']
    assert [getattr(transform.transform.rotation,axis) for axis in ('w','x','y','z')]==record['orientation_wxyz']
    assert [getattr(odom.twist.twist.linear,axis) for axis in ('x','y','z')]==record['linear_velocity_base_m_s']
    assert [getattr(odom.twist.twist.angular,axis) for axis in ('x','y','z')]==record['angular_velocity_base_rad_s']


@pytest.mark.parametrize('fault',['frame','kind','covariance','timestamp','quaternion'])
def test_encoder_message_factory_rejects_invalid_input_before_optional_ROS_import(fault):
    _,_,record,_=feedback_stream_inputs();before=set(sys.modules)
    if fault=='frame':record['child_frame_id']=record['header']['frame_id']
    elif fault=='kind':record['source']='simulator_ground_truth'
    elif fault=='covariance':record['twist_covariance_6x6'][0][0]=-1.
    elif fault=='timestamp':record['header']['stamp']['nanosec']=1000000000
    else:record['orientation_wxyz']=[2.,0.,0.,0.]
    with pytest.raises(ValueError):messages.odometry_ros_message_objects(record)
    assert not any(n in sys.modules and n not in before for n in ('rclpy','nav_msgs','tf2_msgs'))


def ray_map_inputs(**overrides):
    """Small asymmetric SOFTWARE fixture, not hospital observations/a map."""
    arguments=dict(map_frame_id='fixture_map',world_from_map_xyyaw=[0.,0.,0.],origin_xy_m=[0.,0.],
        shape_rows_columns=[3,5],resolution_m=1.,boundary_tolerance_m=1e-9,environment_sha256='a'*64,
        capture_mode='STATIC_VIRTUAL_QUERY',reference_sensor_world_z_m=.5,maximum_sensor_height_error_m=1e-6,
        maximum_ray_tilt_rad=1e-6,maximum_cell_visits=100)
    arguments.update(overrides)
    return arguments


def ray_map_packet(*,sequence=0,stamp=1000000000,position=(.5,1.5,0.),angle=0.,distance=2.,extra_hits=(),
        environment='a'*64,config_edits=None,acquisition_edits=None):
    acquisition,config,_,_=scan_inputs()
    acquisition.update(stamp_ns=stamp,base_xyz=list(position),environment_sha256=environment)
    if acquisition_edits:acquisition.update(acquisition_edits)
    config.update(sensor_xyz_in_base=[0.,0.,.5],ray_count=2,angle_min_rad=angle,angle_increment_rad=.001,range_max_m=4.)
    if config_edits:config.update(config_edits)
    records=[]
    for ray in messages.scan_geometry(acquisition,config)['rays']:
        def hit(d,path='/World/EXTERNAL_FIXTURE'):
            return dict(collision=path,distance_m=d,
                position_m=(np.asarray(ray['origin_world_m'])+d*np.asarray(ray['direction_world'])).tolist(),normal=[0.,0.,1.])
        hits=[] if distance is None else [hit(distance)]
        hits.extend(hit(d,path) for d,path in extra_hits)
        records.append(dict(index=len(records),hits=hits))
    return messages.scan_packet(acquisition,config,records,sequence)


def test_ray_map_only_clears_observed_interiors_and_marks_first_hit_not_beyond_or_unknown():
    grid=messages.RayEvidenceMap(**ray_map_inputs());p=ray_map_packet(extra_hits=[(3.,'/World/BEHIND_FIRST')])
    before=copy.deepcopy(p);receipt=grid.accept(p)
    assert grid.cells.tolist()==[[-1]*5,[0,0,100,-1,-1],[-1]*5]
    assert p==before and receipt['first_external_hit_cell_count']==1
    record=grid.record();assert record['data']==grid.cells.ravel().tolist()
    assert record['cell_counts']=={'-1':12,'0':2,'100':1}
    assert not record['actual_map_adopted'] and not record['actual_sensor_localization_Nav2_success']
    assert record['source_identity']['source_role']=='SOFTWARE_FIXTURE'


@pytest.mark.parametrize('endpoint,expected',[
    ([2.,1.5],{6,7}),([2.,1.],{1,2,6,7}),([5.,1.5],{9}),([5.1,1.5],set())])
def test_ray_endpoint_boundary_supercover_and_no_outside_hit_border_clamping(endpoint,expected):
    _,occupied,_=messages._ray_map_cells([.5,1.5],endpoint,origin=[0.,0.],resolution=1.,
        shape=[3,5],tolerance=1e-9,remaining=100)
    assert occupied==expected


def test_ray_on_gridline_does_not_clear_either_side_and_outside_origin_clips_to_roi():
    free,_,_=messages._ray_map_cells([.5,1.],[4.5,1.],origin=[0.,0.],resolution=1.,shape=[3,5],tolerance=1e-9,remaining=100)
    assert free==set()
    free,_,_=messages._ray_map_cells([-2.,1.5],[2.5,1.5],origin=[0.,0.],resolution=1.,shape=[3,5],tolerance=1e-9,remaining=100)
    assert free=={5,6,7}


def test_ray_no_return_never_adds_border_obstacle_and_observed_occupied_is_sticky():
    grid=messages.RayEvidenceMap(**ray_map_inputs());grid.accept(ray_map_packet())
    grid.accept(ray_map_packet(sequence=1,distance=None))
    assert grid.cells.tolist()==[[-1]*5,[0,0,100,0,0],[-1]*5]
    snapshot=grid.cells;snapshot[:]=0
    assert grid.cells[0,0]==-1
    changed=grid.configuration;changed['origin_xy_m'][0]=999
    assert grid.configuration['origin_xy_m']==[0.,0.]


def test_ray_exact_self_hit_ignored_but_unknown_actor_prefix_still_external():
    own='/World/SOFTWARE_FIXTURE_ACTOR/base/shape';unreviewed='/World/SOFTWARE_FIXTURE_ACTOR/UNKNOWN'
    grid=messages.RayEvidenceMap(**ray_map_inputs())
    grid.accept(ray_map_packet(extra_hits=[(.2,own),(.3,unreviewed)]))
    assert grid.cells[1].tolist()==[100,-1,-1,-1,-1]


def test_ray_declared_rotated_fixed_frame_and_nonzero_bottom_left_origin_are_not_reanchored():
    config=ray_map_inputs(world_from_map_xyyaw=[10.,20.,math.pi/2],origin_xy_m=[-1.,-2.])
    grid=messages.RayEvidenceMap(**config)
    # Map start (-.5,-.5), heading +x -> world start (10.5,19.5), heading +y.
    grid.accept(ray_map_packet(position=(10.5,19.5,0.),angle=math.pi/2))
    assert grid.cells.tolist()==[[-1]*5,[0,0,100,-1,-1],[-1]*5]


@pytest.mark.parametrize('mode,second_stamp,valid',[
    ('STATIC_VIRTUAL_QUERY',1000000000,True),('STATIC_VIRTUAL_QUERY',1000000001,False),
    ('TIME_ADVANCING_SENSOR',1000000001,True),('TIME_ADVANCING_SENSOR',1000000000,False),
    ('TIME_ADVANCING_SENSOR',999999999,False)])
def test_ray_static_real_stamp_not_forged_and_sensor_clock_must_advance(mode,second_stamp,valid):
    grid=messages.RayEvidenceMap(**ray_map_inputs(capture_mode=mode));grid.accept(ray_map_packet())
    p=ray_map_packet(sequence=1,stamp=second_stamp)
    if valid:assert grid.accept(p)['stamp_ns']==second_stamp
    else:
        with pytest.raises(ValueError,match='clock'):grid.accept(p)
        assert grid.failed


@pytest.mark.parametrize('fault',['environment','capture','actor','role','config','sequence','height','tilt','near','moving_map','internal_config','work_budget'])
def test_ray_packet_failure_is_atomic_and_terminal_not_partial_free_space(fault):
    config=ray_map_inputs()
    if fault=='work_budget':config['maximum_cell_visits']=6
    grid=messages.RayEvidenceMap(**config);grid.accept(ray_map_packet())
    before=grid.cells;receipts=copy.deepcopy(grid._receipts);kwargs=dict(sequence=1)
    if fault=='environment':kwargs['environment']='b'*64
    elif fault=='capture':kwargs['acquisition_edits']={'capture_id':'OTHER_CAPTURE'}
    elif fault=='actor':kwargs['acquisition_edits']={'actor_root':'/World/OTHER'};kwargs['config_edits']={'own_collision_paths':['/World/OTHER/shape']}
    elif fault=='role':kwargs['acquisition_edits']={'source_role':'UNVALIDATED_NATIVE_QUERY_CAPTURE'}
    elif fault=='config':kwargs['config_edits']={'rate_hz':6.}
    elif fault=='sequence':kwargs['sequence']=0
    elif fault=='height':kwargs['position']=(.5,1.5,.01)
    elif fault=='tilt':kwargs['config_edits']={'sensor_wxyz_in_base':[math.cos(.01),0.,math.sin(.01),0.]}
    elif fault=='near':kwargs['distance']=.05
    elif fault=='moving_map':kwargs['config_edits']={'base_frame_id':'fixture_map'}
    elif fault=='internal_config':grid._config['origin_xy_m'][0]=99.
    else:kwargs['distance']=None
    with pytest.raises(ValueError):grid.accept(ray_map_packet(**kwargs))
    assert grid.failed and np.array_equal(grid.cells,before) and grid._receipts==receipts
    with pytest.raises(ValueError,match='terminal'):grid.accept(ray_map_packet(sequence=2))
    with pytest.raises(ValueError):grid.record()


@pytest.mark.parametrize('change',[
    {'shape_rows_columns':[True,5]},{'shape_rows_columns':[2001,2001]},{'resolution_m':math.nan},
    {'resolution_m':1e-50},{'boundary_tolerance_m':.25},{'maximum_ray_tilt_rad':math.pi/2},
    {'maximum_cell_visits':0},{'capture_mode':'INVENT_TIME'},{'environment_sha256':'MISSING'},
    {'world_from_map_xyyaw':[math.nan,0.,0.]}])
def test_ray_map_has_no_implicit_plane_grid_epoch_budget_or_unknown_free_policy(change):
    with pytest.raises(ValueError):messages.RayEvidenceMap(**ray_map_inputs(**change))
    grid=messages.RayEvidenceMap(**ray_map_inputs())
    with pytest.raises(ValueError):grid.record()


@pytest.mark.parametrize('fault',['planning_role','success','data','count','frame','stamp','receipt','empty_receipts','scope','shape'])
def test_invalid_map_rejected_before_ROS_import_not_relabelled_planning_mask_or_live_success(fault):
    grid=messages.RayEvidenceMap(**ray_map_inputs());grid.accept(ray_map_packet());record=grid.record()
    if fault=='planning_role':record['role']='PLANNING_SAFETY_GRID'
    elif fault=='success':record['actual_sensor_localization_Nav2_success']=True
    elif fault=='data':record['data'][0]=0
    elif fault=='count':record['cell_counts']['-1']=0
    elif fault=='frame':record['header']['frame_id']='fixture_base'
    elif fault=='stamp':record['header']['stamp']['nanosec']=1
    elif fault=='receipt':record['receipts'][0]['packet_sha256']='invalid'
    elif fault=='empty_receipts':record['receipts']=[]
    elif fault=='scope':record['source_and_environment_provenance_NOT_live_verified']=False
    else:record['configuration']['shape_rows_columns']=[True,5]
    before=set(sys.modules)
    with pytest.raises((ValueError,TypeError)):messages.occupancy_grid_message(record)
    assert not any(name in sys.modules and name not in before for name in ('rclpy','nav_msgs','omni.physx'))


def test_installed_ROS_occupancy_grid_CDR_signed_unknown_bottom_left_row_major_without_context():
    serialization=pytest.importorskip('rclpy.serialization',reason='Matching ROS Python3.10 ABI only; no Kit injection')
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    grid=messages.RayEvidenceMap(**ray_map_inputs());grid.accept(ray_map_packet());record=grid.record()
    message=messages.occupancy_grid_message(record)
    wire=serialization.deserialize_message(serialization.serialize_message(message),OccupancyGrid)
    assert list(wire.data)==[-1]*5+[0,0,100,-1,-1]+[-1]*5
    assert (wire.info.width,wire.info.height)==(5,3) and wire.info.resolution==1.
    assert wire.info.origin.orientation.w==1. and wire.info.origin.position.x==0. and wire.info.origin.position.y==0.
    assert wire.header.stamp.sec==wire.info.map_load_time.sec==1
    assert wire.header.stamp.nanosec==wire.info.map_load_time.nanosec==0
    assert not rclpy.ok() and not record['actual_map_adopted']

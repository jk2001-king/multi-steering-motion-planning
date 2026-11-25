"""Coverage diagnostics never join tasks, invent support or admit driving."""

import copy

import numpy as np
import pytest

import hospital_mission_geometry as mission


def rectangle(x0,x1,y0,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def _slab_gap_mesh(path,y0,y1,z0=0.,z1=1.):
    return dict(path=path,world_points_m=[[0.,y0,z0],[0.,y1,z0],[0.,y1,z1],[0.,y0,z1]],
        face_vertex_counts=[3,3],face_vertex_indices=[0,1,2,0,2,3])


def test_structural_aperture_uses_every_X_and_whole_height_slab_without_source_mutation():
    meshes=[_slab_gap_mesh('left',-2.,-.6),_slab_gap_mesh('right',.6,2.),
        _slab_gap_mesh('thin_low_protrusion',-.6,-.5,.01,.03)]
    meshes[-1]['world_points_m']=[[x+.2,y,z] for x,y,z in meshes[-1]['world_points_m']]
    before=copy.deepcopy(meshes)
    result=mission.authored_slab_transverse_gap(meshes,center_y_m=0.)
    assert result['clear_Y_interval_m']==[-.5,.6] and result['width_m']==pytest.approx(1.1)
    assert result['all_X_and_whole_slab_projection'] and meshes==before
    assert result['other_objects_or_body_cooked_traversal_NOT_certified']


@pytest.mark.parametrize('fault',['empty','duplicate','nan','nontriangle','bad_index','filled_gap','infinite_center','reversed_slab'])
def test_structural_aperture_refuses_missing_invalid_or_obstructed_geometry(fault):
    meshes=[_slab_gap_mesh('left',-2.,-.6),_slab_gap_mesh('right',.6,2.)];kwargs=dict(center_y_m=0.)
    if fault=='empty':meshes=[]
    elif fault=='duplicate':meshes[1]['path']='left'
    elif fault=='nan':meshes[0]['world_points_m'][0][2]=float('nan')
    elif fault=='nontriangle':meshes[0]['face_vertex_counts']=[4]
    elif fault=='bad_index':meshes[0]['face_vertex_indices'][0]=100
    elif fault=='filled_gap':meshes.append(_slab_gap_mesh('middle',-1.,1.))
    elif fault=='infinite_center':kwargs['center_y_m']=float('inf')
    else:kwargs.update(lower_z_m=.5,upper_z_m=.1)
    with pytest.raises(ValueError):mission.authored_slab_transverse_gap(meshes,**kwargs)


def test_south_vertex_widening_keeps_upper_hinge_side_X_Z_and_outside_anchors():
    source=np.array([[.1,y,.2] for y in (-2.,-1.,-.6,0.,.6,1.,2.)]);before=source.copy()
    result,report=mission.south_widening_vertices(source,original_gap_y_m=[-.6,.6],
        lower_anchor_y_m=-2.,geometric_width_m=1.5)
    assert np.array_equal(source,before) and not np.shares_memory(result,source)
    assert np.array_equal(result[:,[0,2]],source[:,[0,2]])
    assert result[0,1]==-2. and np.array_equal(result[4:,1],source[4:,1])
    assert result[2,1]==pytest.approx(-.9) and np.diff(result[:,1]).min()>0.
    assert report['new_gap_Y_m']==pytest.approx([-.9,.6])
    assert report['NEW_architectural_vertex_geometry_NOT_rigid_source_preservation']


@pytest.mark.parametrize('width',[0.,1.,1.2,3.,float('nan'),float('inf')])
def test_south_vertex_widening_rejects_narrower_folded_or_nonfinite_definitions(width):
    with pytest.raises(ValueError):mission.south_widening_vertices([[0.,0.,0.]],original_gap_y_m=[-.6,.6],
        lower_anchor_y_m=-2.,geometric_width_m=width)


def test_common_condition_replaces_only_its_source_masks_and_preserves_overlapping_blockers():
    parts={k:np.zeros((12,12),bool) for k in ('retained','door','threshold')}
    parts['door'][3,3]=True;parts['retained'][3,3]=True
    parts['door'][4,4]=True;parts['threshold'][5,5]=True;parts['threshold'][6,6]=True
    base=np.zeros((12,12),bool);base[5,5]=True  # Unsupported/medical overlap must survive.
    original=base|parts['retained']|parts['door']|parts['threshold']
    historical=np.zeros_like(base);historical[8,8]=True
    before=[a.copy() for a in (base,original,historical,*parts.values())]
    candidate=mission.compose_recorded_common_condition(base,parts,historical,original)
    assert candidate[3,3] and candidate[5,5] and candidate[8,8]
    assert not candidate[4,4] and not candidate[6,6]
    for a,b in zip(before,(base,original,historical,*parts.values())):assert np.array_equal(a,b)


@pytest.mark.parametrize('fault',['missing_partition','changed_baseline','empty_door','empty_threshold',
    'empty_historical','shape','dtype','empty'])
def test_common_condition_refuses_incomplete_or_invented_source_contributions(fault):
    parts={k:np.zeros((12,12),bool) for k in ('retained','door','threshold')}
    parts['door'][2,2]=True;parts['threshold'][4,4]=True
    base=np.zeros((12,12),bool);original=base|parts['door']|parts['threshold']
    historical=np.zeros_like(base);historical[8,8]=True
    if fault=='missing_partition':parts.pop('retained')
    elif fault=='changed_baseline':original[1,1]=True
    elif fault=='empty_door':parts['door'][:]=False;original=base|parts['threshold']
    elif fault=='empty_threshold':parts['threshold'][:]=False;original=base|parts['door']
    elif fault=='empty_historical':historical[:]=False
    elif fault=='shape':historical=np.ones((13,12),bool)
    elif fault=='dtype':historical=historical.astype(int)
    else:base=np.zeros((0,0),bool);original=base;historical=base;parts={k:base for k in parts}
    with pytest.raises(ValueError):mission.compose_recorded_common_condition(base,parts,historical,original)


def test_fixed_unknown_and_support_masks_are_not_inferred_from_a_union_difference():
    grid=dict(raster=dict(origin_xy_m=[0.,0.],resolution_m=.1,shape_rows_columns=[10,10]),
        analysis_roi_xy_m=[.01,.99,.01,.99])
    supported=np.ones((10,10),bool);supported[4,4]=False
    fixed=mission.fixed_cell_blockers(grid,supported)
    assert fixed[4,4] and fixed[0,:].all() and fixed[-1,:].all()
    assert fixed[:,0].all() and fixed[:,-1].all() and not fixed[5,5]
    original_obstacle=np.zeros_like(fixed);original_obstacle[4,4]=True
    legacy=(fixed|original_obstacle)&~original_obstacle
    assert not legacy[4,4] and fixed[4,4]  # Regression: independently blocked overlap erased.


@pytest.mark.parametrize('fault',['dtype','shape','nonfinite_origin','nonfinite_ROI','inverted_ROI',
    'nan_resolution','infinite_resolution','zero_resolution','bool_resolution'])
def test_fixed_blocker_mask_rejects_malformed_or_unbounded_source_grid(fault):
    grid=dict(raster=dict(origin_xy_m=[0.,0.],resolution_m=.1,shape_rows_columns=[10,10]),
        analysis_roi_xy_m=[0.,1.,0.,1.]);supported=np.ones((10,10),bool)
    if fault=='dtype':supported=supported.astype(int)
    elif fault=='shape':grid['raster']['shape_rows_columns']=[100000,100000]
    elif fault=='nonfinite_origin':grid['raster']['origin_xy_m'][0]=float('nan')
    elif fault=='nonfinite_ROI':grid['analysis_roi_xy_m'][0]=float('inf')
    elif fault=='inverted_ROI':grid['analysis_roi_xy_m']=[1.,0.,0.,1.]
    else:grid['raster']['resolution_m']={'nan_resolution':float('nan'),'infinite_resolution':float('inf'),
        'zero_resolution':0.,'bool_resolution':True}[fault]
    with pytest.raises(ValueError):mission.fixed_cell_blockers(grid,supported)


def transfer_assets():
    return [dict(path='/World/Hospital/bed'+str(i),name_based_family='hospital_bed',
        recorded_full_collision_union_bound=dict(minimum_m=[float(x),1.,0.],maximum_m=[float(x)+2.,2.,1.]))
        for i,x in enumerate([1.,8.,40.])]


def test_box_ports_have_fixed_reserve_all_sides_and_no_map_feedback():
    bound=dict(minimum_m=[3.,3.,0.],maximum_m=[5.,4.,1.])
    before=copy.deepcopy(bound);ports=mission.bed_box_port_candidates(bound)
    assert bound==before and len(ports)==6 and len({p['id'] for p in ports})==6
    assert len({tuple(p['world_xyyaw'][:2]) for p in ports})==4
    for p in ports:
        polygon=mission.footprint_polygon(p['world_xyyaw']);lo,hi=polygon.min(axis=0),polygon.max(axis=0)
        if 'north' in p['id']:gap=lo[1]-bound['maximum_m'][1]
        elif 'south' in p['id']:gap=bound['minimum_m'][1]-hi[1]
        elif 'west' in p['id']:gap=bound['minimum_m'][0]-hi[0]
        else:gap=lo[0]-bound['maximum_m'][0]
        assert gap==pytest.approx(.150,abs=1e-12)
        assert 'NOT_CLINICAL' in p['role']


def test_port_outside_ROI_is_unknown_not_a_false_free_or_physical_failure():
    empty=np.zeros((100,100),bool);ports=[dict(id='outside',world_xyyaw=[9.8,5.,0.])]
    result=mission.bed_port_geometry_review(ports,empty,empty,[0.,0.],.1,[0.,10.,0.,10.],
        [rectangle(0.,10.,0.,10.)])[0]
    assert not result['full_footprint_inside_analysis_ROI']
    assert result['outside_ROI_is_unknown_NOT_proven_physical_collision']
    assert not result['known_obstacle_cell_overlap'] and not result['full_rectangle_clear_in_recorded_map']
    assert result['unsupported_full_footprint_area_m2']>0.


def test_port_known_wall_overlap_is_not_erased_by_a_partial_ROI():
    blocked=np.zeros((100,100),bool);blocked[:,93:96]=True
    result=mission.bed_port_geometry_review([dict(id='outside',world_xyyaw=[9.8,5.,0.])],
        blocked,blocked,[0.,0.],.1,[0.,10.,0.,10.],[rectangle(0.,12.,0.,10.)])[0]
    assert result['known_obstacle_cell_overlap'] and not result['full_rectangle_clear_in_recorded_map']
    assert result['outside_ROI_is_unknown_NOT_proven_physical_collision']
    assert result['unsupported_full_footprint_area_m2']==0.


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_port_geometry_rejects_invalid_full_bed_bounds(bad):
    bound=dict(minimum_m=[0.,0.,0.],maximum_m=[bad,1.,1.])
    with pytest.raises(ValueError):mission.bed_box_port_candidates(bound)


def test_selected_full_bed_bounds_extend_ROI_without_omitting_other_beds_or_controls():
    assets=transfer_assets();anchors=[dict(id='old_A',world_xyyaw=[0.,0.,0.]),dict(id='old_B',world_xyyaw=[3.,5.,1.])]
    before=copy.deepcopy((assets,anchors));selected=[a['path'] for a in assets[:2]]
    report=mission.transfer_bed_coverage_bounds(assets,anchors,[rectangle(-2.,15.,-2.,10.)],selected)
    assert (assets,anchors)==before and len(report['all_named_bed_roots'])==3
    assert report['ordered_user_selected_bed_roots']==selected
    assert report['analysis_roi_xy_m'][1]>10. and report['analysis_roi_xy_m'][1]<40.
    assert report['all_original_controls_retained']==2
    assert [r['full_asset_XY_box_inside_recorded_support_union'] for r in report['all_named_bed_roots']]==[True,True,False]
    assert not report['new_support_or_scene_edit'] and not report['actual_motion_authorized']


def test_bed_support_uses_union_not_summed_areas_or_assumed_robot_goal():
    assets=transfer_assets();selected=[a['path'] for a in assets[:2]]
    report=mission.transfer_bed_coverage_bounds(assets,[dict(id='old',world_xyyaw=[0.,0.,0.])],
        [rectangle(1.,2.,1.,2.),rectangle(1.,2.,1.,2.)],selected)
    assert report['all_named_bed_roots'][0]['full_asset_XY_box_unsupported_authored_area_m2']==1.
    assert report['selected_bed_support_NOT_H6_start_goal_or_route_adoption']


@pytest.mark.parametrize('fault',['missing','duplicate_selection','duplicate_asset','inverted','nonfinite','too_large'])
def test_transfer_coverage_rejects_missing_bounds_and_never_coarsens_or_omits_selected_bed(fault):
    assets=transfer_assets();selected=[a['path'] for a in assets[:2]]
    if fault=='missing':selected[1]='/World/Hospital/MISSING'
    elif fault=='duplicate_selection':selected[1]=selected[0]
    elif fault=='duplicate_asset':assets.append(copy.deepcopy(assets[0]))
    elif fault=='inverted':assets[0]['recorded_full_collision_union_bound']['maximum_m'][0]=0.
    elif fault=='nonfinite':assets[0]['recorded_full_collision_union_bound']['minimum_m'][0]=float('nan')
    elif fault=='too_large':
        assets[1]['recorded_full_collision_union_bound'].update(minimum_m=[1000.,1000.,0.],maximum_m=[1002.,1001.,1.])
    with pytest.raises(ValueError):mission.transfer_bed_coverage_bounds(assets,
        [dict(id='old',world_xyyaw=[0.,0.,0.])],[rectangle(-2.,15.,-2.,10.)],selected)


def test_cardinal_rectangle_masks_have_no_false_free_poses_against_original_SAT():
    rng=np.random.default_rng(5072);blocked=np.zeros((80,100),bool)
    blocked[35:40,40:45]=True;blocked[10:15,60:80]=True;resolution=.05
    original=blocked.copy();masks=mission.cardinal_body_sweep_masks(blocked,resolution)
    assert np.array_equal(blocked,original)
    from hospital_geometry_tools import pose_collision
    for name,yaw in [('free_x',0.),('free_y',np.pi/2)]:
        candidates=np.argwhere(masks[name])
        indices=rng.choice(len(candidates),min(200,len(candidates)),replace=False)
        for y,x in candidates[indices]:
            pose=[(x+.5)*resolution,(y+.5)*resolution,yaw]
            assert not pose_collision(blocked,pose,np.zeros(2),resolution)
            for dy,dx in [(0,1),(1,0)]:
                if masks[name][y+dy,x+dx]:
                    target=[pose[0]+dx*resolution,pose[1]+dy*resolution,yaw]
                    assert mission.translation_sweep_free(blocked,pose,target,np.zeros(2),resolution)


def test_disk_rotation_certificate_retains_full_rectangle_at_all_sampled_angles():
    from hospital_geometry_tools import pose_collision
    blocked=np.zeros((80,90),bool);blocked[20:30,40:45]=True;resolution=.05
    masks=mission.cardinal_body_sweep_masks(blocked,resolution)
    points=np.argwhere(masks['rotatable'])
    assert len(points)>0
    for y,x in points[::max(1,len(points)//15)]:
        for yaw in np.linspace(0.,2*np.pi,37):
            assert not pose_collision(blocked,[(x+.5)*resolution,(y+.5)*resolution,yaw],np.zeros(2),resolution)


def test_center_point_connection_is_not_promoted_through_a_body_too_narrow_gap():
    blocked=np.zeros((70,90),bool);blocked[:,43:47]=True;blocked[31:39,43:47]=False
    anchors=[dict(id='original_left',world_xyyaw=[2.,3.5,0.]),dict(id='original_right',world_xyyaw=[7.,3.5,0.])]
    report,arrays=mission.cardinal_body_connection_review(anchors,blocked,[0.,0.],.1)
    assert all(a['source_full_footprint_clear'] for a in report['anchors'])
    assert not report['all_unordered_anchor_pairs'][0]['sufficient_GEOMETRIC_body_sweep_connection']
    assert report['all_unordered_anchor_pairs'][0]['negative_result_NOT_full_orientation_continuous_geometric_infeasibility']
    assert report['restricted_axis_aligned_holonomic_geometric_certificate_NOT_a_robot_route']
    assert not report['actual_motion_authorized'] and report['new_PhysX_trial_count']==0


def test_all_original_controls_and_pairs_remain_in_open_body_sweep_review():
    blocked=np.zeros((80,100),bool)
    anchors=[dict(id='A',world_xyyaw=[2.,2.,0.]),dict(id='B',world_xyyaw=[5.,2.,np.pi/2]),
             dict(id='C',world_xyyaw=[7.,6.,np.pi])]
    for a in anchors:a['source_pose_NOT_changed']=True  # Existing reviewed controls carry metadata already.
    before=copy.deepcopy(anchors)
    report,arrays=mission.cardinal_body_connection_review(anchors,blocked,np.zeros(2),.1)
    assert anchors==before and len(report['all_unordered_anchor_pairs'])==3
    assert all(a['sufficient_GEOMETRIC_body_sweep_connection'] for a in report['all_unordered_anchor_pairs'])
    assert all(a['source_pose_NOT_changed'] for a in report['anchors'])
    for a,b in zip(anchors,report['anchors']):assert a['world_xyyaw']==b['world_xyyaw']
    assert report['cleared_disk_rotation_cell_count']>0
    assert np.all(arrays['geometry_components_x'][~arrays['free_x']]==0)
    assert np.all(arrays['geometry_components_y'][~arrays['free_y']]==0)


def test_optimistic_disk_outer_approximation_keeps_every_sampled_free_full_body_pose():
    from hospital_geometry_tools import pose_collision
    blocked=np.zeros((70,90),bool);blocked[20:30,45:50]=True;resolution=.1
    anchors=[dict(id='CONTROL',world_xyyaw=[2.,3.,0.])]
    review,arrays=mission.optimistic_inscribed_disk_review(anchors,blocked,[0.,0.],resolution)
    rng=np.random.default_rng(6508);examined=0
    for _ in range(400):
        pose=[float(rng.uniform(1.,8.)),float(rng.uniform(1.,6.)),float(rng.uniform(-np.pi,np.pi))]
        if pose_collision(blocked,pose,np.zeros(2),resolution):continue
        x,y=np.floor(np.array(pose[:2])/resolution).astype(int)
        assert arrays['optimistic_disk_cells'][y,x]
        examined+=1
    assert examined>100 and review['eight_neighbor_cell_connections']
    assert review['point_obstacle_centers_and_optimistic_cells_are_a_relaxation_NOT_an_inflated_blocked_map']


def test_optimistic_disk_can_certify_separation_only_within_recorded_cell_policy():
    from scipy.ndimage import label
    blocked=np.zeros((70,90),bool);blocked[:,43:47]=True;blocked[31:39,43:47]=False
    anchors=[dict(id='LEFT',world_xyyaw=[2.,3.5,0.]),dict(id='RIGHT',world_xyyaw=[7.,3.5,np.pi/2])]
    points,_=label(~blocked,structure=np.ones((3,3)))
    assert points[35,20]==points[35,70]!=0  # A point robot can cross the gap.
    necessary,arrays=mission.optimistic_inscribed_disk_review(anchors,blocked,[0.,0.],.1)
    pair=necessary['all_unordered_anchor_pairs'][0]
    assert not pair['necessary_optimistic_disk_connection']
    assert pair['disconnected_implies_no_continuous_full_rectangle_center_path_IN_THIS_RASTER_POLICY']
    assert pair['NOT_global_physical_or_multisteering_planner_infeasibility']


def test_positive_full_body_sweep_certificate_never_contradicts_optimistic_necessary_connection():
    blocked=np.zeros((70,90),bool);blocked[30:40,43:47]=True
    anchors=[dict(id='A',world_xyyaw=[2.,2.,0.]),dict(id='B',world_xyyaw=[7.,5.,np.pi/2])]
    sufficient,_=mission.cardinal_body_connection_review(anchors,blocked,[0.,0.],.1)
    necessary,_=mission.optimistic_inscribed_disk_review(anchors,blocked,[0.,0.],.1)
    assert sufficient['all_unordered_anchor_pairs'][0]['sufficient_GEOMETRIC_body_sweep_connection']
    assert necessary['all_unordered_anchor_pairs'][0]['necessary_optimistic_disk_connection']


def test_diagnostic_partition_projects_every_collider_once_and_preserves_overlaps(monkeypatch):
    identities=['/World/Hospital/DOOR/shape','/World/Hospital/THRESHOLD/shape',
                '/World/Hospital/DOOR_OTHER/shape']
    masks={p:np.zeros((5,6),bool) for p in identities}
    masks[identities[0]][2,3]=True;masks[identities[1]][1,1]=True;masks[identities[2]][2,3]=True
    coverage=dict(coverage_grid=dict(included_obstacles=[dict(path=p) for p in identities]),original_field='KEEP')
    bundle=dict(primitives=[dict(path=p) for p in identities]);before=copy.deepcopy((coverage,bundle))
    calls=[]
    def project(c,b,data):
        chosen=[r['path'] for r in c['coverage_grid']['included_obstacles']]
        assert chosen==[r['path'] for r in b['primitives']]
        assert c['original_field']=='KEEP';calls.extend(chosen)
        mask=np.zeros((5,6),bool)
        for p in chosen:mask|=masks[p]
        return dict(SOFTWARE_ONLY=True),None,None,mask
    monkeypatch.setattr(mission,'refine_recorded_mesh_grid',project)
    review,parts=mission.recorded_obstacle_partition_masks(coverage,bundle,None,
        dict(door='/World/Hospital/DOOR',threshold='/World/Hospital/THRESHOLD'))
    assert sorted(calls)==sorted(identities) and len(calls)==len(set(calls))
    assert review['partitions']['retained']['original_collider_paths']==[identities[2]]
    assert parts['door'][2,3] and parts['retained'][2,3]
    assert (coverage,bundle)==before


@pytest.mark.parametrize('fault',['missing','duplicate','overlap','broad_root','same_roots','reserved_key'])
def test_unknown_or_ambiguous_diagnostic_partition_is_not_an_obstacle_exemption(fault):
    ids=['/World/Hospital/D/shape','/World/Hospital/T/shape']
    c=dict(coverage_grid=dict(included_obstacles=[dict(path=p) for p in ids]))
    b=dict(primitives=[dict(path=p) for p in ids]);roots=dict(door='/World/Hospital/D',threshold='/World/Hospital/T')
    if fault=='missing':roots['door']='/World/Hospital/MISSING'
    elif fault=='duplicate':b['primitives'].append(copy.deepcopy(b['primitives'][0]))
    elif fault=='overlap':roots['threshold']='/World/Hospital/D/shape'
    elif fault=='broad_root':roots['door']='/World/Hospital'
    elif fault=='same_roots':roots['threshold']=roots['door']
    elif fault=='reserved_key':roots['retained']=roots.pop('door')
    with pytest.raises(ValueError):mission.recorded_obstacle_partition_masks(c,b,None,roots)


def test_factorial_is_copy_only_and_retains_other_obstacles_and_medical_keepouts():
    shape=(70,90);parts={k:np.zeros(shape,bool) for k in ['retained','door','threshold']}
    parts['door'][30,40]=True;parts['retained'][30,40]=True  # Shared obstacle cell must survive.
    parts['threshold'][32,42]=True
    base=np.zeros(shape,bool);base[32,42]=True  # Medical/unsupported base must survive.
    baseline=base|parts['retained']|parts['door']|parts['threshold']
    snapshots=[a.copy() for a in [base,baseline,*parts.values()]]
    anchors=[dict(id='A',world_xyyaw=[2.,2.,0.]),dict(id='B',world_xyyaw=[7.,5.,0.])]
    report,grids=mission.diagnostic_two_root_factorial(anchors,parts,base,baseline,[0.,0.],.1)
    assert len(report['cases'])==4 and np.array_equal(grids['door_1_threshold_1'],baseline)
    assert grids['door_0_threshold_0'][30,40] and grids['door_0_threshold_0'][32,42]
    for original,after in zip(snapshots,[base,baseline,*parts.values()]):assert np.array_equal(original,after)
    for case in report['cases']:
        assert case['all_unchanged_anchor_poses']==anchors
        assert case['removed_door_COPY_is_NOT_a_physically_open_door_or_adopted_setup']
    assert report['new_PhysX_trial_count']==0 and not report['actual_motion_authorized']


def test_factorial_refuses_a_changed_baseline_instead_of_silently_adopting_it():
    parts={k:np.zeros((20,20),bool) for k in ['retained','door','threshold']}
    base=np.zeros((20,20),bool);baseline=base.copy();baseline[5,5]=True
    with pytest.raises(ValueError):
        mission.diagnostic_two_root_factorial([dict(id='A',world_xyyaw=[1.,1.,0.])],parts,base,baseline,[0.,0.],.1)


@pytest.fixture
def historical_door_fixture():
    import json
    from pathlib import Path
    coverage=json.loads((mission.GATE/'hospital_h6_geometry_review/coverage/f6d4d271c1a5ea5e/coverage_review.json').read_bytes())
    path,h3=mission.pinned_H3_geometry(coverage['input_sha256'],coverage['contract']['anchors'])
    assert path.parent.name=='setup_r01' and path.parent.parent.name=='hospital_h3_turn_setup'
    audit=Path(h3['source_audit'])
    assert mission.hashlib.sha256(audit.read_bytes()).hexdigest()==h3['source_audit_sha256']
    old=json.loads(audit.read_bytes());leaf=copy.deepcopy(next(a for a in old['local_assets'] if a['path']=='/World/Hospital/SM_Door_02b9'))
    folder=mission.GATE/'hospital_h6_geometry_review/mesh_refinement/d39bce7a869cfad6'
    bundle=json.loads((folder/'mesh_geometry_bundle.json').read_bytes())
    record=copy.deepcopy(next(p for p in bundle['primitives'] if p['path']==leaf['collision']['primitives'][0]['path']))
    prefix=record['array_prefix']
    with np.load(folder/'recorded_world_mesh_arrays.npz') as a:
        arrays={prefix+suffix:a[prefix+suffix].copy() for suffix in ['_points','_counts','_indices']}
    return record,arrays,leaf


def test_exact_original_H3_recorded_door_geometry_not_deleted_or_AABB_fitted(historical_door_fixture):
    record,arrays,leaf=historical_door_fixture
    snapshot=copy.deepcopy((record,leaf));array_snapshot={k:v.copy() for k,v in arrays.items()}
    review,mesh=mission.historical_H3_open_door_mesh(record,arrays,leaf)
    assert review['vertex_count']==284 and review['triangle_count']==200
    assert review['maximum_vertex_recipe_residual_m']<1e-12
    assert review['original_hinge_specification']['additional_world_Z_rotation_deg']==pytest.approx(59.999634731112806)
    assert review['door_remains_an_obstacle_in_the_OFFLINE_candidate']
    assert mesh==leaf['collision']['primitives'][0]['authored_mesh']
    assert (record,leaf)==snapshot
    assert all(np.array_equal(v,array_snapshot[k]) for k,v in arrays.items())


@pytest.mark.parametrize('fault',['root','identity','duplicate','topology','approximation','nonfinite','new_pose','bounds','float_indices'])
def test_historical_door_source_changes_block_instead_of_selecting_new_angle(historical_door_fixture,fault):
    record,arrays,leaf=historical_door_fixture;prefix=record['array_prefix']
    mesh=leaf['collision']['primitives'][0]['authored_mesh']
    if fault=='root':leaf['path']='/World/Hospital/ANOTHER_DOOR'
    elif fault=='identity':leaf['collision']['primitives'][0]['path']+='_CHANGED'
    elif fault=='duplicate':leaf['collision']['primitives'].append(copy.deepcopy(leaf['collision']['primitives'][0]))
    elif fault=='topology':mesh['face_vertex_indices'][0]=2
    elif fault=='approximation':record['geometry']['recorded_physics_approximation']='convexHull'
    elif fault=='nonfinite':arrays[prefix+'_points'][0,0]=float('nan')
    elif fault=='new_pose':arrays[prefix+'_points'][:,0]+=.02
    elif fault=='bounds':leaf['collision']['primitives'][0]['world_aligned_bound']['maximum_m'][0]-=.2
    elif fault=='float_indices':arrays[prefix+'_indices']=arrays[prefix+'_indices'].astype(float)
    with pytest.raises(ValueError):mission.historical_H3_open_door_mesh(record,arrays,leaf)


def test_recorded_original_door_angle_is_not_a_new_planning_parameter(historical_door_fixture):
    record,arrays,leaf=historical_door_fixture
    review,_=mission.historical_H3_open_door_mesh(record,arrays,leaf)
    spec=copy.deepcopy(review['original_hinge_specification']);spec['additional_world_Z_rotation_deg']=90.
    with pytest.raises(ValueError):
        mission.historical_H3_open_door_mesh(record,arrays,leaf,recorded_specification=spec)


def test_direct_translation_connector_checks_entire_sweep_not_just_endpoints():
    from hospital_geometry_tools import pose_collision
    blocked=np.zeros((60,100),bool);blocked[29:32,49:51]=True
    first=[2.,3.,0.];last=[8.,3.,0.]
    assert not pose_collision(blocked,first,np.zeros(2),.1)
    assert not pose_collision(blocked,last,np.zeros(2),.1)
    assert not mission.translation_sweep_free(blocked,first,last,np.zeros(2),.1)
    with pytest.raises(ValueError):mission.translation_sweep_free(blocked,first,[8.,3.,.1],np.zeros(2),.1)


@pytest.mark.parametrize('fault',['nonbool','nonfinite','zero_step','large_step','empty','noncardinal','duplicate','outside'])
def test_body_sweep_unknown_or_changed_source_is_not_silently_repaired(fault):
    blocked=np.zeros((50,70),bool);anchors=[dict(id='A',world_xyyaw=[2.,2.,0.])];resolution=.1
    if fault=='nonbool':blocked=blocked.astype(float)
    elif fault=='nonfinite':resolution=float('nan')
    elif fault=='zero_step':resolution=0.
    elif fault=='large_step':resolution=1.1
    elif fault=='empty':blocked=np.zeros((0,0),bool)
    elif fault=='noncardinal':anchors[0]['world_xyyaw'][2]=.01
    elif fault=='duplicate':anchors.append(copy.deepcopy(anchors[0]))
    elif fault=='outside':
        anchors[0]['world_xyyaw']=[-1.,-1.,0.]
        result,_=mission.cardinal_body_connection_review(anchors,blocked,[0.,0.],resolution)
        assert not result['anchors'][0]['source_full_footprint_clear']
        assert result['anchors'][0]['sufficient_body_sweep_components']==[]
        return
    with pytest.raises(ValueError):mission.cardinal_body_connection_review(anchors,blocked,[0.,0.],resolution)


def row(path='/World/Hospital/OBSTACLE',classification='outside_ROI',*,minimum=(.3,.3,.01),maximum=(.4,.4,.2)):
    result=dict(path=path,is_active=True,collision_enabled_value=classification!='disabled',classification=classification)
    if classification!='disabled':result['world_aligned_bound']=dict(minimum_m=list(minimum),maximum_m=list(maximum),
        size_m=[b-a for a,b in zip(minimum,maximum)])
    return result


def inventory(rows):
    from collections import Counter
    return dict(collision_prims=rows,collision_prim_count=len(rows),classification_counts=dict(Counter(r['classification'] for r in rows)),
        authored_inventory_coverage_complete=True,errors=[],active_unloaded_payload_paths=[])


def build(rows=None,supports=None,roi=(0.,1.,0.,1.)):
    return mission.build_authored_grid(inventory([row()] if rows is None else rows),'/World/ACTOR',roi,
        [rectangle(0.,1.,0.,1.)] if supports is None else supports)


def test_previously_outside_ROI_obstacle_is_reclassified_NOT_ignored_in_integrated_map():
    source=[row()];before=copy.deepcopy(source)
    report,blocked,supported,obstacles=build(source)
    assert source==before and report['old_outside_ROI_rows_now_included']==1
    assert obstacles[35,35] and supported[35,35] and blocked[35,35]
    assert not report['H6_start_goal_path_control_or_motion_admitted']


def test_original_disabled_has_no_invented_bound_and_only_exact_actor_self_is_excluded():
    source=[row('/World/ACTOR/base/shape','robot_self_excluded'),row('/World/Hospital/DISABLED','disabled'),
        row('/World/ACTOR_UNKNOWN/shape')]
    report,blocked,*_=build(source)
    assert [r['path'] for r in report['included_obstacles']]==['/World/ACTOR_UNKNOWN/shape']
    assert blocked[35,35] and len(report['excluded_colliders'])==2


def test_unknown_support_holes_stay_blocked_and_overlapping_area_is_not_double_counted():
    source=[row(minimum=(2.,2.,.01),maximum=(3.,3.,.2))]
    supports=[rectangle(0.,.6,0.,1.),rectangle(.4,.8,0.,1.)]
    report,blocked,supported,_=build(source,supports)
    assert supported[:,75].all() and not supported[:,90].any()
    assert blocked[:,90].all() and not blocked[:,75].any()
    assert report['support_union_partition_count']==2


def test_subcell_support_gap_is_not_filled_by_corners_or_sum_of_polygon_areas():
    source=[row(minimum=(2.,2.,.01),maximum=(3.,3.,.2))]
    _,blocked,supported,_=build(source,[rectangle(0.,.501,0.,1.),rectangle(.502,1.,0.,1.)])
    assert not supported[:,50].any() and blocked[:,50].all()
    assert supported[:,49].all() and supported[:,51].all()


def test_original_height_slab_excludes_floor_and_roof_but_preserves_low_wall():
    source=[row('/World/FLOOR',minimum=(0.,0.,.001),maximum=(1.,1.,.001)),
        row('/World/ROOF',minimum=(0.,0.,3.),maximum=(1.,1.,3.1)),row('/World/LOW_WALL')]
    report,*_=build(source)
    assert [r['path'] for r in report['included_obstacles']]==['/World/LOW_WALL']
    assert report['original_height_slab_m']==[.006,.450]


@pytest.mark.parametrize('fault',['duplicate','count','missing_bound','bad_bound','unknown_flag','classification','unloaded','error','incomplete','self_mismatch'])
def test_incomplete_or_inconsistent_inventory_fails_before_free_space_is_created(fault):
    source=inventory([row()])
    if fault=='duplicate':source=inventory([row(),row()])
    elif fault=='count':source['collision_prim_count']=2
    elif fault=='missing_bound':source['collision_prims'][0].pop('world_aligned_bound')
    elif fault=='bad_bound':source['collision_prims'][0]['world_aligned_bound']['minimum_m'][0]=float('nan')
    elif fault=='unknown_flag':source['collision_prims'][0]['collision_enabled_value']=None
    elif fault=='classification':source['classification_counts']={}
    elif fault=='unloaded':source['active_unloaded_payload_paths']=['/World/UNREAD_PAYLOAD']
    elif fault=='error':source['errors']=['SOFTWARE_ONLY_MISSING_BOUND']
    elif fault=='incomplete':source['authored_inventory_coverage_complete']=False
    else:source=inventory([row('/World/NOT_OWN/shape','robot_self_excluded')])
    with pytest.raises((ValueError,KeyError)):mission.build_authored_grid(source,'/World/ACTOR',[0.,1.,0.,1.],[rectangle(0.,1.,0.,1.)])


@pytest.mark.parametrize('supports',[[rectangle(0.,0.,0.,1.)],[[[0.,0.],[1.,1.],[1.,0.],[0.,1.]]],[],[[[0.,0.],[1.,0.],[.5,1.]]]])
def test_nonrectangular_crossed_degenerate_or_missing_support_is_not_repaired(supports):
    with pytest.raises(ValueError):build(supports=supports)


@pytest.mark.parametrize('roi',[(1.,0.,0.,1.),(0.,1.,0.,float('inf')),(0.,1000.,0.,1000.)])
def test_invalid_or_unbounded_ROI_is_not_silently_cropped_or_coarsened(roi):
    with pytest.raises(ValueError):build(roi=roi)


def test_point_connectivity_is_not_a_body_route_and_all_pose_pairs_are_kept():
    supports=[rectangle(0.,4.,0.,4.)]
    source=[row(minimum=(5.,5.,.01),maximum=(6.,6.,.2))]
    grid,blocked,*_=build(source,supports,roi=(0.,4.,0.,4.))
    anchors=[dict(id='first',world_xyyaw=[1.,1.,0.]),dict(id='second',world_xyyaw=[3.,1.,np.pi/2]),
        dict(id='edge',world_xyyaw=[.1,.1,0.])]
    result=mission.task_anchor_review(anchors,grid,blocked,supports)
    assert len(result['all_unordered_anchor_pairs'])==3
    assert all(r['same_POINT_center_component'] for r in result['all_unordered_anchor_pairs'])
    assert result['anchors'][2]['point_center_free_in_source_raster']
    assert not result['anchors'][2]['full_original_10mm_footprint_clear_in_source_raster']
    assert result['anchors'][2]['unsupported_full_footprint_area_m2']>0.
    assert result['controls_NOT_a_frozen_H6_start_goal_or_path'] and not result['actual_motion_authorized']


def test_separate_center_components_do_not_get_a_straight_line_or_teleport_bridge():
    supports=[rectangle(0.,4.,0.,4.)]
    source=[row(minimum=(1.99,0.,.01),maximum=(2.01,4.,.2))]
    grid,blocked,*_=build(source,supports,roi=(0.,4.,0.,4.))
    result=mission.task_anchor_review([dict(id='a',world_xyyaw=[1.,1.,0.]),dict(id='b',world_xyyaw=[3.,1.,0.])],grid,blocked,supports)
    assert not result['all_unordered_anchor_pairs'][0]['same_POINT_center_component']
    assert result['point_disconnection_NOT_global_continuous_geometric_infeasibility']


def test_new_generic_builder_reproduces_the_entire_saved_H5_grid_with_its_original_ROI():
    import json
    from pathlib import Path
    path=mission.GATE/'hospital_h5_geometry_candidates/common_map/e16a2898d38c414b/common_map_candidate.json'
    common=json.loads(path.read_bytes());raw=json.loads(Path(common['current_live_map_receipt']).read_bytes())
    actor=json.loads(Path(raw['retained_result']).read_bytes())['physics_preparation_snapshot']['new_robot_root']
    report,blocked,supported,_=mission.build_authored_grid(raw['inventory'],actor,raw['review_roi_xy_m'],common['exact_same_authored_support_rectangles'])
    grid=next(Path(name) for name in common['output_sha256'] if name.endswith('.npz'))
    with np.load(grid) as old:
        assert np.array_equal(blocked,old['occupied'])
        assert np.array_equal(supported,old['supported_full_cells'])
        assert np.array_equal(report['raster']['origin_xy_m'],old['origin_xy_m'])
        assert report['raster']['resolution_m']==float(old['resolution_m'])
    assert not report['H6_start_goal_path_control_or_motion_admitted']


def test_prior_flush_open_environment_is_not_silently_assumed_after_source_reopen():
    h3=dict(suppressed_threshold_root='/World/Hospital/THRESHOLD',
        environment_condition='H3_flush_fully_open_continuous_floor',door_condition='fully_open_baseline_session_only')
    audit=dict(local_assets=[dict(path='/World/Hospital/DOOR',
        collision=dict(bound=dict(minimum_m=[0.,0.,0.],maximum_m=[.1,1.,2.])))])
    source=inventory([row('/World/Hospital/THRESHOLD/mesh',minimum=(0.,0.,.001),maximum=(.2,1.,.025)),
        row('/World/Hospital/DOOR/mesh',minimum=(0.,0.,0.),maximum=(.7,.8,2.))])
    before=copy.deepcopy(source);report=mission.doorway_condition_review(h3,audit,source,door_root='/World/Hospital/DOOR')
    assert source==before and report['currently_recorded_enabled_threshold_paths']==['/World/Hospital/THRESHOLD/mesh']
    assert not report['same_H3_threshold_suppression_recorded'] and not report['door_AABB_matches_recorded_H3_within_1um']
    assert report['physical_clearance_or_causal_obstacle_failure_NOT_inferred_from_AABB']
    assert report['no_door_threshold_flag_or_scene_change_performed']


def test_original_H3_geometry_comes_from_pinned_lineage_not_a_newer_folder_with_same_pose():
    import json
    protocol=mission.GATE/'hospital_h5_experiments/cohort_01_P_development_r01/runtime_protocol.json'
    anchors,pins=mission.existing_task_controls(protocol,mission.ROOT/'studies/IJCAS/reports/technical_evidence/korean_review_manifest.json')
    path,record=mission.pinned_H3_geometry(pins,anchors)
    assert path==mission.GATE/'hospital_h3_turn_setup/setup_r01/hospital_h3_turn_geometry.json'
    assert record['environment_condition']=='H3_flush_fully_open_continuous_floor'
    newer=mission.GATE/'hospital_h3_turn_setup/setup_r03/hospital_h3_turn_geometry.json'
    assert newer.is_file() and str(newer) not in pins
    with pytest.raises(ValueError,match='Exactly one'):mission.pinned_H3_geometry({},anchors)


def typed(values,name):
    if isinstance(values,list) and values and isinstance(values[0],list):
        child='Vec3f' if len(values[0])==3 else 'Vec4d'
        return dict(type=name,values=[dict(type=child,values=v) for v in values])
    return dict(type=name,values=values)


def archived_triangle():
    points=[[0.,0.,.01],[.2,0.,.01],[0.,.2,.2]]
    matrix=np.eye(4);matrix[3,:3]=[1.,2.,0.]
    attrs={name:dict(value=typed(value,kind),time_samples=[]) for name,value,kind in (
        ('points',points,'Vec3fArray'),('faceVertexCounts',[3],'IntArray'),('faceVertexIndices',[0,1,2],'IntArray'),
        ('holeIndices',[],'IntArray'))}
    attrs['physics:approximation']=dict(value='none',time_samples=[])
    return dict(type='Mesh',attributes=attrs,world_matrix=typed(matrix.tolist(),'Matrix4d'))


def _selective_archive_fixture(tmp_path):
    import hashlib,json
    import hospital_compact_usd_fingerprint as compact
    selected='/World/Hospital/selected_mesh';other='/World/Hospital/other_mesh'
    row=archived_triangle();row.update(active=True,relationships={},bound_physics_material=None)
    payload=dict(protected_environment_before_Stop={selected:row,other:copy.deepcopy(row)},
        environment_after_actual_static={})
    archive=tmp_path/'original.json';archive.write_text(json.dumps(payload,indent=2))
    stat=archive.stat()
    manifest=dict(codec=compact.FORMAT,prim_row_count=2,original_archive=str(archive),
        original_archive_bytes=stat.st_size,original_archive_mtime_ns=stat.st_mtime_ns,
        original_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),approved_disabled_paths=[],
        compact_environment_rows={p:compact.archived_row(p,r,[]) for p,r in payload['protected_environment_before_Stop'].items()})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),selected,other,payload


def test_selective_reader_matches_unchanged_original_decoder_and_does_not_claim_full_hash(tmp_path):
    import hospital_compact_usd_fingerprint as compact
    path,digest,selected,other,payload=_selective_archive_fixture(tmp_path)
    rows,receipt=mission.read_authenticated_mesh_rows(path,[selected],expected_manifest_sha256=digest)
    legacy={p:r for p,r,_ in compact.iter_archived_rows(tmp_path/'original.json')}
    assert rows=={selected:legacy[selected]} and other not in rows
    assert receipt['complete_row_name_roster_verified']==2
    assert receipt['selected_semantically_authenticated_meshes']==1
    assert receipt['new_full_archive_rehash_performed'] is False
    assert receipt['actual_motion_authorized'] is False


@pytest.mark.parametrize('fault',['bad_hash','missing_path','duplicate_selection','outside_hospital','empty',
    'changed_stat','wrong_fingerprint','missing_roster','duplicate_row','bad_selected_json'])
def test_selective_reader_fails_closed_for_source_or_selection_faults(tmp_path,fault):
    import hashlib,json
    path,digest,selected,other,payload=_selective_archive_fixture(tmp_path)
    selection=[selected];manifest=json.loads(path.read_text());archive=tmp_path/'original.json'
    if fault=='bad_hash':digest='0'*64
    elif fault=='missing_path':selection=['/World/Hospital/missing']
    elif fault=='duplicate_selection':selection=[selected,selected]
    elif fault=='outside_hospital':selection=['/World/not_hospital']
    elif fault=='empty':selection=[]
    else:
        if fault=='changed_stat':manifest['original_archive_bytes']+=1
        elif fault=='wrong_fingerprint':manifest['compact_environment_rows'][selected]['row_sha256']='0'*64
        else:
            text=archive.read_text()
            if fault=='missing_roster':text=text.replace('"'+other+'"','"/World/Hospital/replaced"')
            elif fault=='duplicate_row':text=text.replace('"'+other+'"','"'+selected+'"')
            else:text=text.replace('"Mesh"','INVALID_JSON',1)
            archive.write_text(text);stat=archive.stat()
            manifest.update(original_archive_bytes=stat.st_size,original_archive_mtime_ns=stat.st_mtime_ns)
        path.write_text(json.dumps(manifest));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError,KeyError)):
        mission.read_authenticated_mesh_rows(path,selection,expected_manifest_sha256=digest)


def test_selective_reader_never_parses_unselected_payloads(tmp_path):
    import hashlib,json
    path,digest,selected,other,payload=_selective_archive_fixture(tmp_path)
    archive=tmp_path/'original.json';text=archive.read_text()
    offset=text.index('"'+other+'"');text=text[:offset]+text[offset:].replace('"Mesh"','BAD_UNSELECTED_JSON',1)
    archive.write_text(text);manifest=json.loads(path.read_text());stat=archive.stat()
    manifest.update(original_archive_bytes=stat.st_size,original_archive_mtime_ns=stat.st_mtime_ns)
    path.write_text(json.dumps(manifest));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rows,receipt=mission.read_authenticated_mesh_rows(path,[selected],expected_manifest_sha256=digest)
    assert rows[selected]==payload['protected_environment_before_Stop'][selected]
    assert receipt['new_full_archive_rehash_performed'] is False


def test_selective_archive_reader_refuses_Kit_before_any_file_access(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        mission.read_authenticated_mesh_rows('/MUST_NOT_READ',[],expected_manifest_sha256='0'*64)


def _metadata_source_fixture(*,with_scale=False):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    UsdPhysics=pytest.importorskip('pxr.UsdPhysics');UsdShade=pytest.importorskip('pxr.UsdShade')
    Sdf=pytest.importorskip('pxr.Sdf');Vt=pytest.importorskip('pxr.Vt')
    import hospital_h4_root_difference_readback as fields
    asset=Usd.Stage.CreateInMemory();root=UsdGeom.Xform.Define(asset,'/Asset');asset.SetDefaultPrim(root.GetPrim())
    mesh=UsdGeom.Mesh.Define(asset,'/Asset/Mesh')
    mesh.CreatePointsAttr(Vt.Vec3fArray([(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]));mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0,1,2]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([(0.,0.,1.)]*3));mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr('none');UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    if with_scale:
        Gf=pytest.importorskip('pxr.Gf')
        UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3f(.01))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdGeom.PrimvarsAPI(root).CreatePrimvar('inherited_control',Sdf.ValueTypeNames.Float,'constant').Set(2.)
    pv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying')
    pv.Set(Vt.Vec2fArray([(0.,0.),(1.,0.),(0.,1.)]));pv.SetIndices(Vt.IntArray([0,1,2]))
    stage=Usd.Stage.CreateInMemory();instance=UsdGeom.Xform.Define(stage,'/World/Hospital/instance').GetPrim()
    instance.GetReferences().AddReference(asset.GetRootLayer().identifier);instance.SetInstanceable(True)
    path='/World/Hospital/instance/Mesh';prim=stage.GetPrimAtPath(path)
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    row=dict(active=bool(prim.IsActive()),type=prim.GetTypeName(),attributes={
        a.GetName():dict(value=fields.semantic_value(a.Get()),
            time_samples=[[t,fields.semantic_value(a.Get(t))] for t in a.GetTimeSamples()]) for a in prim.GetAttributes()},
        relationships={r.GetName():list(map(str,r.GetTargets())) for r in prim.GetRelationships()},
        world_matrix=fields.semantic_value(UsdGeom.XformCache().GetLocalToWorldTransform(prim)),
        bound_physics_material=str(material.GetPath()) if material else None)
    return stage,asset,mesh,{path:row}


def test_source_metadata_GET_records_actual_proxy_interpolation_and_inherited_indexed_primvars_without_edit():
    import json
    stage,asset,mesh,expected=_metadata_source_fixture()
    before=[s.GetRootLayer().ExportToString() for s in (stage,asset)]
    result=mission.source_mesh_composition_metadata(stage,expected);item=result[0]
    assert [s.GetRootLayer().ExportToString() for s in (stage,asset)]==before
    assert item['normals_interpolation']=='vertex' and item['normal_count']==3
    assert item['exact_original_values_world_topology_normals_extent_and_material_retained']
    assert item['mesh_and_ancestor_composition'][0]['instance_proxy']
    assert any(r['instanceable'] and r['instance'] for r in item['mesh_and_ancestor_composition'])
    primvars={p['name']:p for p in item['inherited_primvars']}
    assert primvars['inherited_control']['interpolation']=='constant'
    assert primvars['st']['indexed'] and primvars['st']['interpolation']=='faceVarying'
    assert primvars['st']['indices_sha256']
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_source_metadata_GET_private_embedded_loader_uses_same_existing_stage_without_import_cache_mutation():
    import runpy,sys
    stage,asset,mesh,expected=_metadata_source_fixture();before=dict(sys.modules)
    namespace=runpy.run_path(str(mission.ROOT/'studies/IJCAS/scripts/hospital_private_loader.py'))['load'](mission.__file__)
    result=namespace['source_mesh_composition_metadata'](stage,expected)
    assert result==mission.source_mesh_composition_metadata(stage,expected)
    assert all(sys.modules[k] is v for k,v in before.items())


def _anonymous_session_fixture(*, frame_change=False, child=False):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    Gf=pytest.importorskip('pxr.Gf')
    stage,asset,mesh,rows=_metadata_source_fixture(with_scale=True)
    if child:UsdGeom.Xform.Define(asset,'/Asset/Mesh/UnreviewedChild')
    control=UsdGeom.Xform.Define(stage,'/World/Hospital/unselected_instance').GetPrim()
    control.GetReferences().AddReference(asset.GetRootLayer().identifier);control.SetInstanceable(True)
    metadata=dict(status='SOURCE_MESH_COMPOSITION_AND_RETAINED_PAUSED_STATE_VERIFIED_GET_ONLY',error=None,
        checks=dict(original=True),meshes=mission.source_mesh_composition_metadata(stage,rows))
    path=next(iter(rows));row=rows[path];matrix=mission.semantic_array(row['world_matrix'],'world')
    if frame_change:
        transform=Gf.Matrix4d(1.);transform.SetRotate(Gf.Rotation(Gf.Vec3d(0.,0.,1.),15.))
        matrix=np.asarray(transform)@matrix;matrix[3,:3]=[.2,.1,0.]
    local=mission.semantic_array(row['attributes']['points']['value'],'points');local[1,0]=1.2
    world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
    _,arrays=mission.mesh_local_geometry_recipe(row,world,target_world_matrix=matrix)
    by_path={path:arrays};plan=mission.mesh_session_override_plan(metadata,rows,by_path)
    return stage,asset,metadata,rows,by_path,plan


@pytest.mark.parametrize('frame_change',[False,True])
def test_anonymous_session_override_only_selected_instance_and_native_arrays(frame_change):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    stage,asset,metadata,rows,arrays,plan=_anonymous_session_fixture(frame_change=frame_change)
    original=[layer.ExportToString() for layer in (stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]
    report,text,scratch=mission.preview_anonymous_mesh_session_override(stage,plan,arrays)
    assert original==[layer.ExportToString()for layer in (stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]
    assert scratch is not stage and report['geometry_array_opinions']==3
    assert report['additional_leaf_matrix_ops']==int(frame_change)
    assert scratch.GetPrimAtPath('/World/Hospital/unselected_instance').IsInstance()
    assert not scratch.GetPrimAtPath('/World/Hospital/instance').IsInstance()
    assert stage.GetPrimAtPath('/World/Hospital/instance').IsInstance()
    assert stage.GetPrimAtPath('/World/Hospital/unselected_instance').GetPrototype()==stage.GetPrimAtPath('/World/Hospital/instance').GetPrototype()
    path=next(iter(rows));mesh=UsdGeom.Mesh(scratch.GetPrimAtPath(path))
    original_pv=UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).GetPrimvar('st')
    new_pv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    assert np.array_equal(original_pv.Get(),new_pv.Get()) and np.array_equal(original_pv.GetIndices(),new_pv.GetIndices())
    assert new_pv.GetInterpolation()=='faceVarying'
    assert mesh.GetPrim().GetAttribute('xformOp:scale').Get()==stage.GetPrimAtPath(path).GetAttribute('xformOp:scale').Get()
    original_order=list(stage.GetPrimAtPath(path).GetAttribute('xformOpOrder').Get())
    assert list(mesh.GetPrim().GetAttribute('xformOpOrder').Get())==original_order+(
        ['xformOp:transform:h6_layout'] if frame_change else [])
    assert 'instanceable = false' in text and 'physics:collisionEnabled' not in text
    assert not report['actual_motion_authorized'] and report['whole_original_Hospital_or_shader_network_NOT_reconstructed']


@pytest.mark.parametrize('fault',['status','error','failed_check','missing','duplicate','fingerprint',
    'normal_interpolation','normal_count','subdivision','orientation','root_parent','multiple_roots',
    'nonproxy','extra_geometry','changed_points','changed_extent','missing_indices'])
def test_session_plan_rejects_unverified_metadata_or_geometry_before_any_authoring(fault):
    stage,asset,meta,rows,arrays,plan=_anonymous_session_fixture()
    before=[layer.ExportToString()for layer in(stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]
    path=next(iter(rows));record=meta['meshes'][0]
    if fault=='status':meta['status']='HOLD'
    elif fault=='error':meta['error']='failure'
    elif fault=='failed_check':meta['checks']['original']=False
    elif fault=='missing':meta['meshes']=[]
    elif fault=='duplicate':meta['meshes']*=2
    elif fault=='fingerprint':record['original_complete_value_world_material_row_sha256']='0'*64
    elif fault=='normal_interpolation':record['normals_interpolation']='faceVarying'
    elif fault=='normal_count':record['normal_count']=4
    elif fault=='subdivision':record['subdivision_scheme']='catmullClark'
    elif fault=='orientation':record['orientation']='leftHanded'
    elif fault=='root_parent':
        next(a for a in record['mesh_and_ancestor_composition']if a['instance'])['path']='/World/Hospital'
    elif fault=='multiple_roots':record['mesh_and_ancestor_composition'].append(dict(instanceable=True,instance=True,path='/World/Hospital/other'))
    elif fault=='nonproxy':record['mesh_and_ancestor_composition'][0]['instance_proxy']=False
    elif fault=='extra_geometry':arrays['/World/Hospital/unreviewed']=copy.deepcopy(arrays[path])
    elif fault=='changed_points':arrays[path]['local_points'][0,0]+=.2
    elif fault=='changed_extent':arrays[path]['local_extent'][0,0]+=.2
    else:del arrays[path]['indices']
    with pytest.raises((ValueError,KeyError)):mission.mesh_session_override_plan(meta,rows,arrays)
    assert before==[layer.ExportToString()for layer in(stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]


@pytest.mark.parametrize('fault',['changed_array','extra_permission','parent_root','unreviewed_child'])
def test_scratch_override_rejects_extra_scope_and_never_mutates_input_fixture(fault):
    stage,asset,meta,rows,arrays,plan=_anonymous_session_fixture(frame_change=True,child=fault=='unreviewed_child')
    before=[layer.ExportToString()for layer in(stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]
    if fault=='changed_array':arrays[next(iter(arrays))]['local_points'][0,0]+=.2
    elif fault=='extra_permission':plan['meshes'][0]['allowed_property_opinions'].append('physics:collisionEnabled')
    elif fault=='parent_root':plan['roots_to_deinstance']=['/World/Hospital'];plan['meshes'][0]['instance_root']='/World/Hospital'
    with pytest.raises(ValueError):mission.preview_anonymous_mesh_session_override(stage,plan,arrays)
    assert before==[layer.ExportToString()for layer in(stage.GetRootLayer(),stage.GetSessionLayer(),asset.GetRootLayer())]


def test_scratch_review_refuses_Kit_before_accessing_stage(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        mission.preview_anonymous_mesh_session_override(object(),{}, {})


def test_scratch_review_refuses_file_backed_USD_without_edit(tmp_path):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    path=tmp_path/'not_a_permitted_source.usda';stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage,'/World');stage.GetRootLayer().Save();before=path.read_bytes()
    with pytest.raises(ValueError,match='anonymous'):
        mission.preview_anonymous_mesh_session_override(stage,{}, {})
    assert path.read_bytes()==before


def test_two_selected_mesh_sections_deinstance_one_root_only():
    Sdf=pytest.importorskip('pxr.Sdf')
    stage,asset,meta,rows,arrays,plan=_anonymous_session_fixture()
    assert Sdf.CopySpec(asset.GetRootLayer(),'/Asset/Mesh',asset.GetRootLayer(),'/Asset/Second')
    first=next(iter(rows));second='/World/Hospital/instance/Second'
    rows[second]=copy.deepcopy(rows[first]);arrays[second]=copy.deepcopy(arrays[first])
    meta['meshes']=mission.source_mesh_composition_metadata(stage,rows)
    plan=mission.mesh_session_override_plan(meta,rows,arrays)
    report,text,scratch=mission.preview_anonymous_mesh_session_override(stage,plan,arrays)
    assert report['selected_instance_roots']==1 and report['selected_meshes']==2
    assert report['geometry_array_opinions']==6 and text.count('instanceable = false')==1
    assert scratch.GetPrimAtPath('/World/Hospital/unselected_instance').IsInstance()


@pytest.mark.parametrize('entry',['reconstruct','export'])
def test_offline_source_reconstruction_and_export_refuse_Kit_before_input_access(monkeypatch,entry):
    import sys
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        if entry=='reconstruct':mission.reconstructed_anonymous_mesh_instances({}, {})
        else:mission.export_mesh_session_override_review('/MUST_NOT_READ','/MUST_NOT_READ')


def _common_environment_fixture():
    UsdGeom=pytest.importorskip('pxr.UsdGeom');Gf=pytest.importorskip('pxr.Gf')
    import hospital_h4_root_difference_readback as fields
    stage,asset,meta,rows,arrays,plan=_anonymous_session_fixture(frame_change=True)
    for name in ('first_door','first_threshold'):
        prim=UsdGeom.Xform.Define(stage,'/World/Hospital/'+name).GetPrim()
        prim.GetReferences().AddReference(asset.GetRootLayer().identifier);prim.SetInstanceable(True)
    first='/World/Hospital/first_door/Mesh';matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(first)))
    target=matrix.copy();target[3,0]+=.2
    parked=[]
    for name in ('cart','chair'):
        root='/World/Hospital/'+name;xform=UsdGeom.Xform.Define(stage,root)
        xform.AddTranslateOp().Set(Gf.Vec3d(2.,3.,0.))
        xform.AddOrientOp().Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
        # Real selected roots use double quaternion/scale ops.
        xform.ClearXformOpOrder();xform.GetPrim().RemoveProperty('xformOp:orient')
        xform.GetPrim().RemoveProperty('xformOp:translate')
        xform.AddTranslateOp().Set(Gf.Vec3d(2.,3.,0.))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.,Gf.Vec3d(0.)))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.))
        UsdGeom.Xform.Define(stage,root+'/geometry').AddTranslateOp().Set(Gf.Vec3d(.1,0.,0.))
        before=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(xform.GetPrim()));after=before.copy();after[3,0]+=1.
        parked.append(dict(root=root,source_root_ordered_ops=[dict(name=o.GetOpName(),inverse=False,
            value=fields.semantic_value(o.Get()),time_samples=[]) for o in xform.GetOrderedXformOps()],
            source_root_world_matrix=before.tolist(),proposed_root_world_matrix=after.tolist(),
            exact_prospective_Session_default_overrides={'xformOp:translate':dict(type='Vec3d',values=[3.,3.,0.])}))
    condition=dict(version='H6_COMMON_ENVIRONMENT_ANONYMOUS_COMPOSITION_v1',actual_motion_authorized=False,
        first_door=dict(root='/World/Hospital/first_door',path=first,
            source_world_matrix=matrix.tolist(),target_world_matrix=target.tolist()),
        disabled_thresholds=[next(iter(rows)),'/World/Hospital/first_threshold/Mesh'],parked_assets=parked)
    return stage,asset,arrays,plan,condition


def test_combined_common_condition_keeps_source_shared_instances_UV_and_original_ops():
    stage,asset,arrays,plan,condition=_common_environment_fixture()
    before=[layer.ExportToString() for layer in stage.GetUsedLayers()]
    report,text,scratch=mission.preview_anonymous_common_environment(stage,plan,arrays,condition)
    assert before==[layer.ExportToString() for layer in stage.GetUsedLayers()]
    assert report['no_second_destination_root_rotation_applied']
    assert report['total_selected_instance_roots']==3
    assert report['destination_geometry_phase_counts']['selected_instance_roots']==1
    assert report['threshold_collision_flags_disabled_exactly']==condition['disabled_thresholds']
    for path in condition['disabled_thresholds']:
        assert scratch.GetPrimAtPath(path).GetAttribute('physics:collisionEnabled').Get() is False
        assert stage.GetPrimAtPath(path).GetAttribute('physics:collisionEnabled').Get() is True
    assert scratch.GetPrimAtPath('/World/Hospital/unselected_instance').IsInstance()
    assert 'xformOp:transform:h6_common' in text and not report['actual_motion_authorized']
    assert scratch.GetRootLayer().ExportToString()==stage.GetRootLayer().ExportToString()


@pytest.mark.parametrize('fault',['authority','duplicate_threshold','extra_parked','same_first',
    'first_scale','first_z','first_source','root_order','root_source','new_op','wrong_target','zero_quaternion'])
def test_combined_condition_rejects_extra_or_inconsistent_scope_without_source_mutation(fault):
    stage,asset,arrays,plan,c=_common_environment_fixture()
    before=[layer.ExportToString() for layer in stage.GetUsedLayers()]
    if fault=='authority':c['actual_motion_authorized']=True
    elif fault=='duplicate_threshold':c['disabled_thresholds']*=2
    elif fault=='extra_parked':c['parked_assets']*=2
    elif fault=='same_first':c['first_door']['path']=plan['meshes'][0]['path']
    elif fault=='first_scale':c['first_door']['target_world_matrix'][0][0]*=2.
    elif fault=='first_z':c['first_door']['target_world_matrix'][3][2]+=.1
    elif fault=='first_source':c['first_door']['source_world_matrix'][3][0]+=.1
    elif fault=='root_order':c['parked_assets'][0]['source_root_ordered_ops'].reverse()
    elif fault=='root_source':c['parked_assets'][0]['source_root_world_matrix'][3][0]+=.1
    elif fault=='new_op':c['parked_assets'][0]['exact_prospective_Session_default_overrides']['xformOp:scale']={}
    elif fault=='wrong_target':c['parked_assets'][0]['proposed_root_world_matrix'][3][0]+=1.
    else:c['parked_assets'][0]['exact_prospective_Session_default_overrides']['xformOp:orient']=dict(
        type='Quatd',real=0.,imaginary=dict(type='Vec3d',values=[0.,0.,0.]))
    with pytest.raises((ValueError,KeyError)):mission.preview_anonymous_common_environment(stage,plan,arrays,c)
    assert before==[layer.ExportToString() for layer in stage.GetUsedLayers()]


def test_combined_environment_refuses_Kit_before_SDK_or_inputs(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        mission.preview_anonymous_common_environment(object(),{}, {}, {})
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        mission.export_common_environment_session_review('/MUST_NOT_READ',output='/MUST_NOT_WRITE')


def test_source_interpolation_absent_from_old_value_archive_is_observed_not_inferred_from_count():
    stage,asset,mesh,expected=_metadata_source_fixture()
    mesh.SetNormalsInterpolation('faceVarying')
    item=mission.source_mesh_composition_metadata(stage,expected)[0]
    assert item['normals_interpolation']=='faceVarying'
    assert item['normal_count']==item['point_count']==3
    assert item['exact_original_values_world_topology_normals_extent_and_material_retained']


@pytest.fixture(scope='module')
def recorded_common_preparation_data():
    pytest.importorskip('pxr.Usd')
    import json
    base=mission.GATE/'hospital_h6_task_geometry/0e5f25f64d2256c2/native_local_geometry_r01'
    folder=base/'common_environment_session_composition_r01'
    native=json.loads((base/'native_local_geometry_review.json').read_text())
    plan=json.loads((base/'session_instance_composition_r02/session_opinion_plan.json').read_text())
    fixture=copy.deepcopy(plan);rows=json.loads((base/'authenticated_selected_source_rows.json').read_text())['original_mesh_rows']
    for child in ('source_metadata_GET_r01','remaining_common_environment_source_GET_r01'):
        meta=json.loads((base/child/'live_source_metadata.json').read_text())
        if child.startswith('remaining'):
            rows.update(json.loads((base/child/'authenticated_selected_source_rows.json').read_text())['original_mesh_rows'])
            for record in meta['meshes']:
                root=next(a['path'] for a in record['mesh_and_ancestor_composition'] if a['instance'])
                fixture['roots_to_deinstance'].append(root)
                fixture['meshes'].append(dict(path=record['path'],instance_root=root,
                    source_world_matrix=mission.semantic_array(rows[record['path']]['world_matrix'],'world').tolist(),
                    normal_interpolation='vertex',observed_primvars=record['inherited_primvars']))
    return base,folder,fixture,rows


def _recorded_common_preparation_fixture(data):
    import json
    from pxr import UsdGeom,UsdShade,Gf
    import hospital_compact_usd_fingerprint as compact
    base,folder,fixture,rows=data;stage,assets,_=mission.reconstructed_anonymous_mesh_instances(fixture,rows)
    recipe=json.loads((folder/'common_condition_recipe.json').read_text());selected=list(rows)
    for asset in recipe['parked_assets']:
        root=UsdGeom.Xform.Define(stage,asset['root'])
        for record in asset['source_root_ordered_ops']:
            value=record['value']
            if record['name']=='xformOp:translate':root.AddTranslateOp().Set(Gf.Vec3d(*value['values']))
            elif record['name']=='xformOp:orient':root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(value['real'],Gf.Vec3d(*value['imaginary']['values'])))
            else:root.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*value['values']))
        mesh=UsdGeom.Mesh.Define(stage,asset['collider']);mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
        mesh.AddScaleOp().Set(Gf.Vec3f(.01))
        material=UsdShade.Material.Define(stage,asset['root']+'/Looks/PreservedShaderControl')
        selected.extend((asset['collider'],str(material.GetPath())))
    before={p:mission.common_environment_descriptor(stage,p) for p in selected};after=copy.deepcopy(before)
    text=(folder/'anonymous_common_Session_opinions.usda').read_text();layer,attrs,_=mission.checked_common_Session_patch(text)
    for path,target in after.items():
        for spec in attrs.values():
            if str(spec.path.GetPrimPath())==path:
                target['attributes'][spec.name]=dict(default_sha256=compact.native_digest(spec.default),samples=[])
        for name in ('xformOp:transform:h6_layout','xformOp:transform:h6_common'):
            if path+'.'+name in attrs:target['world_matrix']=np.asarray(attrs[path+'.'+name].default*Gf.Matrix4d(target['world_matrix'])).tolist()
        for asset in recipe['parked_assets']:
            if path.startswith(asset['root']+'/'):
                delta=np.linalg.inv(asset['source_root_world_matrix'])@np.asarray(asset['proposed_root_world_matrix'])
                target['world_matrix']=(np.asarray(target['world_matrix'])@delta).tolist()
    return stage,assets,text,before,after


def test_reviewed_property_writer_matches_combined_patch_without_layer_import_or_shared_asset_edits(recorded_common_preparation_data):
    stage,assets,text,before,after=_recorded_common_preparation_fixture(recorded_common_preparation_data)
    root=stage.GetRootLayer().ExportToString();sources=[a.GetRootLayer().ExportToString() for a in assets]
    report=mission.apply_common_Session_patch(stage,text,before,after,authorized=True)
    assert stage.GetRootLayer().ExportToString()==root
    assert sources==[a.GetRootLayer().ExportToString() for a in assets]
    assert report['property_opinions']==33 and report['selected_instance_roots']==9
    assert mission.verify_common_environment_descriptors(stage,after)
    for index in range(9):assert stage.GetPrimAtPath('/World/Hospital/UnselectedControl'+str(index).zfill(2)).IsInstance()
    with pytest.raises(ValueError):mission.apply_common_Session_patch(stage,text,before,after,authorized=True)


@pytest.mark.parametrize('permission',[False,None,0,1,'yes'])
def test_live_environment_writer_and_runner_require_literal_consent_before_SDK(permission):
    import asyncio
    with pytest.raises(ValueError):mission.apply_common_Session_patch(object(),None,{}, {},authorized=permission)
    with pytest.raises(ValueError):asyncio.run(mission.run_common_preparation_once('/MUST_NOT_READ',authorized=permission))
    with pytest.raises(ValueError):mission.schedule_common_preparation('/MUST_NOT_READ',authorized=permission)


@pytest.mark.parametrize('fault',['extra_property','outside_root','true_threshold','wrong_count','reference','time_samples'])
def test_common_patch_schema_rejects_unreviewed_scope_before_live_writes(recorded_common_preparation_data,fault):
    from pxr import Sdf
    _,folder,_,_=recorded_common_preparation_data
    layer=Sdf.Layer.CreateAnonymous('bad_patch.usda');layer.ImportFromString((folder/'anonymous_common_Session_opinions.usda').read_text())
    if fault=='outside_root':Sdf.CreatePrimInLayer(layer,'/World/Unreviewed')
    elif fault=='wrong_count':Sdf.CreatePrimInLayer(layer,'/World/Hospital/Extra')
    elif fault=='reference':Sdf.CreatePrimInLayer(layer,'/World/Hospital').referenceList.prependedItems=[Sdf.Reference('MUST_NOT_OPEN.usd')]
    else:
        path='/World/Hospital/Geo_M_DoorFloor20/Geo_M_DoorFloor/Geo_M_DoorFloor'
        prim=layer.GetPrimAtPath(path)
        if fault=='extra_property':Sdf.AttributeSpec(prim,'physics:contactOffset',Sdf.ValueTypeNames.Float).default=.1
        elif fault=='true_threshold':layer.GetAttributeAtPath(path+'.physics:collisionEnabled').default=True
        else:layer.SetTimeSample(Sdf.Path(path+'.physics:collisionEnabled'),0.,False)
    with pytest.raises(ValueError):mission.checked_common_Session_patch(layer.ExportToString())


def test_property_writer_detects_changed_material_values_before_any_live_opinion(recorded_common_preparation_data):
    stage,assets,text,before,after=_recorded_common_preparation_fixture(recorded_common_preparation_data)
    from pxr import Sdf
    material=next(p for p in before if p.endswith('/PreservedShaderControl'))
    stage.GetPrimAtPath(material).CreateAttribute('unreviewed',Sdf.ValueTypeNames.Float).Set(1.)
    snapshot=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):mission.apply_common_Session_patch(stage,text,before,after,authorized=True)
    assert stage.GetSessionLayer().ExportToString()==snapshot


def test_source_signature_retains_original10_interpolation_and_primvars(recorded_common_preparation_data):
    import json
    from pxr import Usd,UsdGeom
    stage,assets,text,before,after=_recorded_common_preparation_fixture(recorded_common_preparation_data)
    base=recorded_common_preparation_data[0];observed=[]
    for child in ('source_metadata_GET_r01','remaining_common_environment_source_GET_r01'):
        observed.extend(json.loads((base/child/'live_source_metadata.json').read_text())['meshes'])
    for record in observed:
        current=stage.GetPrimAtPath(record['path']);ancestors=[]
        while current and not current.IsPseudoRoot():
            ancestors.append(dict(path=str(current.GetPath()),type_name=current.GetTypeName(),active=bool(current.IsActive()),
                loaded=bool(current.IsLoaded()),instanceable=bool(current.IsInstanceable()),instance=bool(current.IsInstance()),instance_proxy=bool(current.IsInstanceProxy())))
            current=current.GetParent()
        record['mesh_and_ancestor_composition']=ancestors
    # This fixture reconstructs source values/frames but not every original
    # ancestor type. Test the reader's exact signature against its own typed
    # hierarchy; the protocol separately retains actual observed metadata.
    mission.source_composition_signature(stage,observed)
    path=observed[0]['path'];root=next(a['path'] for a in observed[0]['mesh_and_ancestor_composition'] if a['instance'])
    stage.GetPrimAtPath(root).SetInstanceable(False)
    with pytest.raises(ValueError):mission.source_composition_signature(stage,observed)


def test_prepare_only_protocol_export_is_anonymous_and_preserves_B2_FAIL(tmp_path,monkeypatch,recorded_common_preparation_data):
    import json
    base,folder,_,_=recorded_common_preparation_data
    prior=mission.GATE/'hospital_h5_experiments/cohort_04_B2_development_r01/route/hospital_h5_cohort_04_B2_development_r01_B2_multimode_transition_cost_ramp_r01.json'
    original=prior.read_bytes();monkeypatch.setattr(mission,'GATE',tmp_path)
    path=mission.export_common_preparation_protocol(folder/'common_environment_review.json',prior)
    p=json.loads(path.read_text());env=json.loads((path.parent/'environment_contract.json').read_text())
    assert prior.read_bytes()==original and p['prior_verdict_preserved']=='FAIL'
    assert (p['maximum_Stop_calls'],p['maximum_static_trials'],p['maximum_route_trials'])==(1,0,0)
    assert len(env['expected_after'])==14 and sum(v['type']=='Material' for v in env['expected_after'].values())==2
    assert p['new_robot_root'].startswith('/World/patient_transport_h6_') and p['start_xyyaw']==[14.192329139709472,16.568600039835417,0]
    assert len(json.loads((path.parent/'anonymous_source_reference.json').read_text())['fixed_configuration_rows'])==145
    context=mission.load_common_preparation_protocol(path)
    import hospital_h4_subfloor_compact_path_launch as shared
    from types import SimpleNamespace
    stage_stub=SimpleNamespace(GetRootLayer=lambda:None)
    sdk=SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage_stub)),
        timeline=SimpleNamespace(get_timeline_interface=lambda:object())))
    operation=mission.common_preparation_operations(context,sdk)
    assert operation.__code__ is shared.prepare_fresh_live_trial.__code__
    assert operation.__globals__['initialize_fresh_staged_display'].__name__=='display'
    assert not any((path.parent/n).exists() for n in ('dispatch_reserved.json','workflow_reserved.json','static','route','preparation.json'))
    with pytest.raises(ValueError):mission.export_common_preparation_protocol(folder/'common_environment_review.json',prior)


@pytest.mark.parametrize('fault',[None,'camera_during_Stop','Stop_timeout','foreign_Stop_edit','changed_source_after_Stop'])
def test_bound_original_preparation_with_combined_environment_never_Plays_or_Stops_twice(
        tmp_path,monkeypatch,recorded_common_preparation_data,fault):
    import asyncio,hashlib,json,sys
    from pathlib import Path
    from types import SimpleNamespace,ModuleType
    from pxr import Usd,UsdGeom,UsdShade,Gf
    import hospital_h4_subfloor_compact_path_launch as shared
    stage,assets,text,before,after=_recorded_common_preparation_fixture(recorded_common_preparation_data)
    old='/World/retained_SOFTWARE_B2';robot='/World/patient_transport_h6_SOFTWARE_prepare';marker='/World/H6_SOFTWARE_marker'
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage,old)
        cam=UsdGeom.Camera.Define(stage,'/OmniverseKit_Top');cam.CreateHorizontalApertureAttr(10.)
    UsdShade.Material.Define(stage,shared.preparation.old.constructor.MATERIAL)
    snapshot=dict(SOFTWARE_ONLY_complete_tensor=True);state=dict(playing=False,stopped=False,current_time_s=30.)
    events=[];pending=False
    def stop():
        nonlocal pending
        events.append('STOP');pending=True
        assert events.count('STOP')==1
    timeline=SimpleNamespace(is_playing=lambda:state['playing'],is_stopped=lambda:state['stopped'],
        get_current_time=lambda:state['current_time_s'],stop=stop)
    async def update():
        nonlocal pending
        if pending:
            if fault=='Stop_timeout':return
            state.update(playing=False,stopped=True,current_time_s=0.);pending=False
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                if fault=='camera_during_Stop':cam.GetHorizontalApertureAttr().Set(12.)
                if fault=='foreign_Stop_edit':UsdGeom.Xform.Define(stage,'/World/UNREVIEWED')
                if fault=='changed_source_after_Stop':stage.GetPrimAtPath(next(iter(before))).GetAttribute('points').Set([(0,0,0)])
    # Only interface stubs are installed; no Kit/PhysX engine or real timeline.
    modules={}
    for name in ('omni','omni.usd','omni.timeline','omni.kit','omni.kit.app','isaacsim','isaacsim.core','isaacsim.core.simulation_manager'):
        module=ModuleType(name);module.__path__=[];modules[name]=module;monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:setattr(modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],module)
    modules['omni.usd'].get_context=lambda:SimpleNamespace(get_stage=lambda:stage)
    modules['omni.timeline'].get_timeline_interface=lambda:timeline
    modules['omni.kit.app'].get_app=lambda:SimpleNamespace(next_update_async=update)
    modules['isaacsim.core.simulation_manager'].SimulationManager=SimpleNamespace(get_physics_sim_view=lambda:
        SimpleNamespace(create_articulation_view=lambda _:object()))
    sdk=SimpleNamespace(omni=modules['omni'])
    prior=tmp_path/'SOFTWARE_prior.json';prior.write_text(json.dumps(dict(input_sha256={
        str(Path(mission.__file__)):hashlib.sha256(Path(mission.__file__).read_bytes()).hexdigest()})))
    p=dict(cohort='SOFTWARE_ONLY',method='COMMON_ENVIRONMENT_PREPARATION_ONLY',output_directory=str(tmp_path),
        new_robot_root=robot,marker=marker,retired_robot_root=old,required_prior_result=str(prior),
        required_prior_tensor=snapshot,required_prior_timeline=copy.deepcopy(state),retiring_actor_fixed_configuration={'SOFTWARE_ONLY':True},
        expected_nominal_motor_baseline={'SOFTWARE_ONLY':True},start_xyyaw=[14.192329139709472,16.568600039835417,0.],
        Stop_wall_budget_s=15.,Stop_update_budget=3 if fault=='Stop_timeout' else 240,
        compact_readback_wall_budget_s=180.,initialize_new_actor_visuals_before_static=True,
        environment_contract=str(tmp_path/'environment_contract.json'),common_environment_patch=str(tmp_path/'patch.usda'))
    (tmp_path/'patch.usda').write_text(text)
    recipe=json.loads((recorded_common_preparation_data[1]/'common_condition_recipe.json').read_text())
    (tmp_path/'common_condition_recipe.json').write_text(json.dumps(recipe))
    observed=[];base=recorded_common_preparation_data[0]
    for child in ('source_metadata_GET_r01','remaining_common_environment_source_GET_r01'):
        observed.extend(json.loads((base/child/'live_source_metadata.json').read_text())['meshes'])
    for record in observed:
        current=stage.GetPrimAtPath(record['path']);record['mesh_and_ancestor_composition']=[]
        while current and not current.IsPseudoRoot():
            record['mesh_and_ancestor_composition'].append(dict(path=str(current.GetPath()),type_name=current.GetTypeName(),
                active=bool(current.IsActive()),loaded=bool(current.IsLoaded()),instanceable=bool(current.IsInstanceable()),
                instance=bool(current.IsInstance()),instance_proxy=bool(current.IsInstanceProxy())))
            current=current.GetParent()
    _,_,roots=mission.checked_common_Session_patch(text)
    (tmp_path/'environment_contract.json').write_text(json.dumps(dict(expected_before=before,expected_after=after,
        observed_source10_metadata=observed,exact_instance_roots=roots)))
    context=mission.common_preparation_context(p,tmp_path/'SOFTWARE_protocol.json')
    def compact_actual():
        result={}
        for path in before:
            descriptor=mission.common_environment_descriptor(stage,path);matrix=descriptor.pop('world_matrix')
            descriptor['world_sha256']=shared.preparation.compact.native_digest(Gf.Matrix4d(matrix))
            flag=stage.GetPrimAtPath(path).GetAttribute('physics:collisionEnabled')
            result[path]=dict(row_sha256=hashlib.sha256(shared.preparation.compact._json_bytes(descriptor)).hexdigest(),
                collision_enabled=flag.Get() if flag else None)
        for path in context.preparation.SELECTED:result[path]=dict(row_sha256='SOFTWARE_fixed_fourflag_row',collision_enabled=False)
        for i in range(4997-len(result)):result['SOFTWARE_UNCHANGED_ROW_'+str(i)]=dict(row_sha256='unchanged',collision_enabled=None)
        return result
    original=copy.deepcopy(compact_actual())
    for path in context.preparation.SELECTED:original[path]['collision_enabled']=True
    p['protected_environment_sha256']=context.preparation.compact.compare_rows(original,compact_actual(),context.preparation.SELECTED,True)
    (tmp_path/'SOFTWARE_protocol.json').write_text(json.dumps(p))
    context.preparation.load_manifest=lambda **kwargs:dict(compact_environment_rows=original)
    context.preparation.compact=SimpleNamespace(**vars(context.preparation.compact))
    context.preparation.compact.stage_rows=lambda *args:iter(compact_actual().items())
    monkeypatch.setattr(context.preparation.old,'tensor_snapshot',lambda _:copy.deepcopy(snapshot))
    monkeypatch.setattr(shared,'retiring_actor_configuration',lambda *args:p['retiring_actor_fixed_configuration'])
    async def display(own):
        assert state==dict(playing=False,stopped=True,current_time_s=0.)
        await fresh_environment_readback(own,stage,timeline,'before_renderer_initialization')
        shared.fresh_source_for_live_actor(own,stage);events.append('SOFTWARE_DISPLAY_PLACEHOLDER')
    monkeypatch.setattr(shared,'initialize_fresh_staged_display',display)
    context.stationary.critical_environment=lambda *args:dict(SOFTWARE_ONLY=True)
    before_root=stage.GetRootLayer().ExportToString()
    operation=mission.common_preparation_operations(context,sdk)
    if fault in (None,'camera_during_Stop'):
        result=asyncio.run(operation(context))
        assert result['timeline']==dict(playing=False,stopped=True,current_time_s=0.)
        assert result['live_new_robot_readback']['simulation_mass_kg']==265.
        assert len(result['live_new_robot_readback']['rigid_bodies'])==13
        assert len(result['live_new_robot_readback']['revolute_joint_names'])==12
        assert not result['Play_or_static_or_drive_executed']
        assert mission.verify_common_environment_descriptors(stage,after)
        assert events==['STOP','SOFTWARE_DISPLAY_PLACEHOLDER']
        assert stage.GetRootLayer().ExportToString()==before_root
        assert not stage.GetPrimAtPath(old).IsActive()
    else:
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):asyncio.run(operation(context))
        assert events==['STOP'] and not stage.GetPrimAtPath(robot)
        assert not context.preparation.LIVE.exists()
    assert not (tmp_path/'static').exists() and not (tmp_path/'route').exists()


@pytest.mark.parametrize('fault',['empty','missing_prim','outside_hospital','wrong_type','points_changed','wrong_matrix'])
def test_source_metadata_GET_rejects_missing_or_changed_original_values(fault):
    stage,asset,mesh,expected=_metadata_source_fixture();path=next(iter(expected))
    if fault=='empty':expected={}
    elif fault=='missing_prim':expected['/World/Hospital/missing']=expected.pop(path)
    elif fault=='outside_hospital':expected['/World/elsewhere']=expected.pop(path)
    elif fault=='wrong_type':expected[path]['type']='Xform'
    elif fault=='points_changed':mesh.GetPointsAttr().Set([(0.,0.,0.),(1.1,0.,0.),(0.,1.,0.)])
    else:expected[path]['world_matrix']['values'][3]['values'][0]=1.
    with pytest.raises(ValueError):mission.source_mesh_composition_metadata(stage,expected)


class _GETTimeline:
    def __init__(self):self.state=dict(playing=False,stopped=False,current_time_s=30.)
    def is_playing(self):return self.state['playing']
    def is_stopped(self):return self.state['stopped']
    def get_current_time(self):return self.state['current_time_s']
    def play(self):raise AssertionError('GET must never Play')
    def stop(self):raise AssertionError('GET must never Stop')
    def pause(self):raise AssertionError('GET must never Pause')


class _GETLayer:
    def __init__(self):self.text='immutable layer'
    def ExportToString(self):return self.text


class _GETStage:
    def __init__(self):self.root=_GETLayer();self.session=_GETLayer()
    def GetRootLayer(self):return self.root
    def GetSessionLayer(self):return self.session


@pytest.mark.parametrize('fault',['none','pre_timeline','pre_stage','pre_tensor','post_layer','post_timeline','post_tensor','post_stage'])
def test_source_GET_requires_exact_before_after_boundary_and_never_recovers(monkeypatch,fault):
    stage=_GETStage();timeline=_GETTimeline();expected_time=dict(timeline.state);expected_tensor=dict(body='saved')
    calls=[]
    def capture(*args):
        calls.append('metadata')
        if fault=='post_layer':stage.root.text='external change'
        elif fault=='post_timeline':timeline.state['current_time_s']+=1.
        return []
    monkeypatch.setattr(mission,'source_mesh_composition_metadata',capture)
    snapshots=iter([dict(body='wrong') if fault=='pre_tensor' else expected_tensor,
        dict(body='wrong') if fault=='post_tensor' else expected_tensor])
    stages=iter([object() if fault=='pre_stage' else stage,object() if fault=='post_stage' else stage])
    if fault=='pre_timeline':timeline.state['playing']=True
    invoke=lambda:mission.checked_source_mesh_metadata_GET(stage,timeline,lambda:next(stages),lambda:next(snapshots),
        expected_time,expected_tensor,{'/World/Hospital/selected':{}})
    if fault.startswith('pre_'):
        with pytest.raises(ValueError):invoke()
        assert calls==[]
    else:
        result=invoke();assert calls==['metadata']
        assert result['status'].startswith('SOURCE_')==(fault=='none')
        assert result['actual_motion_authorized'] is False
        assert result['source_metadata_NOT_environment_static_drive_or_sensor_admission']


def test_source_GET_entry_rejects_implicit_authority_before_file_or_SDK_access():
    with pytest.raises(RuntimeError,match='GET-only'):
        mission.run_source_mesh_metadata_GET_once('/MUST_NOT_ACCESS')


def _source_GET_entry_fixture(tmp_path,monkeypatch,*,fault='none'):
    import hashlib,json,runpy,sys
    from types import ModuleType,SimpleNamespace
    timeline=_GETTimeline();stage=_GETStage();counts=dict(view_gets=0,view_handles=0,snapshots=0)
    expected=dict(names=['joint'+str(i) for i in range(12)],link_names=['link'+str(i) for i in range(13)],body='saved')
    selected='/World/Hospital/selected';actor='/World/retained_B2'
    raw=dict(physics_preparation_snapshot=dict(new_robot_root=actor),actual_final_timeline=dict(timeline.state),
        actual_post_route_tensor_snapshot=expected)
    b2=tmp_path/'B2.json';b2.write_text(json.dumps(raw));source=tmp_path/'selected.json'
    source.write_text(json.dumps(dict(original_mesh_rows={selected:dict(type='Mesh')})))
    def snapshotter(tensor):
        counts['snapshots']+=1
        return dict(expected,body='changed') if fault=='tensor' else copy.deepcopy(expected)
    shared=dict(normalize_runtime_evidence=copy.deepcopy,
        preparation=SimpleNamespace(old=SimpleNamespace(tensor_snapshot=snapshotter)))
    monkeypatch.setattr(runpy,'run_path',lambda p:{'load':lambda p:shared})
    def get_view():
        counts['view_gets']+=1
        if fault=='no_view':return None
        def create_handle(path):
            assert path==actor+'/base_link';counts['view_handles']+=1;return object()
        return SimpleNamespace(create_articulation_view=create_handle)
    omni=ModuleType('omni');usd=ModuleType('omni.usd');tl=ModuleType('omni.timeline')
    usd.get_context=lambda:SimpleNamespace(get_stage=lambda:stage);tl.get_timeline_interface=lambda:timeline
    omni.usd=usd;omni.timeline=tl
    isaac=ModuleType('isaacsim');core=ModuleType('isaacsim.core');sm=ModuleType('isaacsim.core.simulation_manager')
    sm.SimulationManager=SimpleNamespace(get_physics_sim_view=get_view);isaac.core=core;core.simulation_manager=sm
    for name,value in [('omni',omni),('omni.usd',usd),('omni.timeline',tl),
            ('isaacsim',isaac),('isaacsim.core',core),('isaacsim.core.simulation_manager',sm)]:
        monkeypatch.setitem(sys.modules,name,value)
    def metadata(*args):
        if fault=='source':raise ValueError('Original source value changed')
        if fault=='post_layer':stage.session.text='external edit'
        return [dict(path=selected,normals_interpolation='vertex',normal_count=3,mesh_and_ancestor_composition=[])]
    monkeypatch.setattr(mission,'source_mesh_composition_metadata',metadata)
    target=tmp_path/'live_source_metadata.json';contract=tmp_path/'GET_contract.json'
    data=dict(version='H6_ORIGINAL_MESH_COMPOSITION_GET_ONLY_v1',GET_target=str(target),
        required_B2_result=str(b2),required_B2_actor=actor,required_actual_saved_timeline=dict(timeline.state),
        authenticated_selected_source_rows=str(source),selected_paths=[selected],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (b2,source)},
        actual_motion_authorized=False,new_PhysX_trial_count=0)
    contract.write_text(json.dumps(data))
    if fault=='timeline':timeline.state['playing']=True
    elif fault=='input':b2.write_text('changed input')
    return contract,target,counts


def test_source_GET_entry_existing_handle_only_and_duplicate_invocation_returns_saved_observation(tmp_path,monkeypatch):
    import json
    contract,target,counts=_source_GET_entry_fixture(tmp_path,monkeypatch)
    assert mission.run_source_mesh_metadata_GET_once(contract,authorized=True)==str(target)
    saved=target.read_bytes();result=json.loads(saved)
    assert result['status']=='SOURCE_MESH_COMPOSITION_AND_RETAINED_PAUSED_STATE_VERIFIED_GET_ONLY'
    assert all(result['checks'].values()) and counts==dict(view_gets=1,view_handles=1,snapshots=2)
    assert result['actual_motion_authorized'] is False and result['new_PhysX_trial_count']==0
    assert mission.run_source_mesh_metadata_GET_once(contract,authorized=True)==str(target)
    assert target.read_bytes()==saved and counts==dict(view_gets=1,view_handles=1,snapshots=2)


@pytest.mark.parametrize('fault',['input','timeline','no_view','tensor','source','post_layer'])
def test_source_GET_entry_preserves_HOLD_without_initialization_recovery_or_retry(tmp_path,monkeypatch,fault):
    import json
    contract,target,counts=_source_GET_entry_fixture(tmp_path,monkeypatch,fault=fault)
    if fault=='input':
        with pytest.raises(ValueError,match='input changed'):
            mission.run_source_mesh_metadata_GET_once(contract,authorized=True)
        assert counts==dict(view_gets=0,view_handles=0,snapshots=0) and not target.exists()
    else:
        mission.run_source_mesh_metadata_GET_once(contract,authorized=True);report=json.loads(target.read_bytes())
        assert report['status'].startswith('HOLD_') and not report['actual_motion_authorized']
        if fault in ('timeline','no_view'):assert counts['view_handles']==counts['snapshots']==0
        frozen=dict(counts);saved=target.read_bytes()
        mission.run_source_mesh_metadata_GET_once(contract,authorized=True)
        assert counts==frozen and target.read_bytes()==saved


def _local_recipe_source():
    row=archived_triangle();row['attributes'].update(
        orientation=dict(value='rightHanded',time_samples=[]),subdivisionScheme=dict(value='none',time_samples=[]))
    return row


def test_local_recipe_rounds_to_native_point3f_and_recomputes_normal_extent_without_mutation():
    row=_local_recipe_source();before=copy.deepcopy(row)
    matrix=mission.semantic_array(row['world_matrix'],'world_matrix')
    local=mission.semantic_array(row['attributes']['points']['value'],'points');local[1,0]+=.123456789
    target=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
    report,arrays=mission.mesh_local_geometry_recipe(row,target)
    assert row==before and arrays['local_points'].dtype==np.float32
    assert np.array_equal(arrays['local_extent'],[arrays['local_points'].min(0),arrays['local_points'].max(0)])
    assert np.allclose(np.linalg.norm(arrays['local_normals'],axis=1),1.)
    assert report['maximum_world_point_roundtrip_error_m']<1e-7
    assert report['source_interpolation_and_instance_metadata_NOT_inferred']
    assert not report['actual_motion_authorized']


def test_local_recipe_retains_explicit_rigid_leaf_frame_without_baking_a_wrong_matrix():
    row=_local_recipe_source();source=mission.semantic_array(row['attributes']['points']['value'],'points')
    matrix=np.eye(4);matrix[:2,:2]=[[0.,1.],[-1.,0.]];matrix[3,:3]=[20.,11.,1.]
    local=source.copy();local[:,0]*=1.22;world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
    report,arrays=mission.mesh_local_geometry_recipe(row,world,target_world_matrix=matrix)
    assert np.array_equal(arrays['world_matrix'],matrix)
    assert np.allclose(arrays['local_points'],local) and np.allclose(arrays['rounded_world_points_m'],world)


@pytest.mark.parametrize('fault',['nan','shape','singular','perspective','collapse','reverse','subdivision','orientation','time_samples','double_points'])
def test_local_recipe_rejects_invalid_native_geometry_or_unsupported_source(fault):
    row=_local_recipe_source();local=mission.semantic_array(row['attributes']['points']['value'],'points')
    matrix=mission.semantic_array(row['world_matrix'],'matrix');world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
    if fault=='nan':world[0,0]=float('nan')
    elif fault=='shape':world=world[:2]
    elif fault=='singular':matrix[:3,:3]=0.
    elif fault=='perspective':matrix[0,3]=1.
    elif fault=='collapse':world[1]=world[0]
    elif fault=='reverse':world=world[[0,2,1]]
    elif fault=='subdivision':row['attributes']['subdivisionScheme']['value']='catmullClark'
    elif fault=='orientation':row['attributes']['orientation']['value']='leftHanded'
    elif fault=='time_samples':row['attributes']['points']['time_samples']=[[1.,row['attributes']['points']['value']]]
    else:row['attributes']['points']['value']['type']='Vec3dArray'
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        mission.mesh_local_geometry_recipe(row,world,target_world_matrix=matrix)


def test_local_recipe_native_USD_export_reimport_retains_points_normals_extent_and_transform():
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    Gf=pytest.importorskip('pxr.Gf');Vt=pytest.importorskip('pxr.Vt');Sdf=pytest.importorskip('pxr.Sdf')
    row=_local_recipe_source();matrix=mission.semantic_array(row['world_matrix'],'matrix')
    local=mission.semantic_array(row['attributes']['points']['value'],'points');local[1,0]*=1.22
    world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
    _,arrays=mission.mesh_local_geometry_recipe(row,world)
    stage=Usd.Stage.CreateInMemory();mesh=UsdGeom.Mesh.Define(stage,'/Mesh')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(arrays['local_points']))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(arrays['local_normals']));mesh.SetNormalsInterpolation('vertex')
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(arrays['local_extent']))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(arrays['counts']))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(arrays['indices']))
    UsdGeom.Xformable(mesh.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(matrix.tolist()))
    layer=Sdf.Layer.CreateAnonymous('replay.usda');assert layer.ImportFromString(stage.GetRootLayer().ExportToString())
    replayed=Usd.Stage.Open(layer);mesh=UsdGeom.Mesh(replayed.GetPrimAtPath('/Mesh'))
    for attr,key in ((mesh.GetPointsAttr(),'local_points'),(mesh.GetNormalsAttr(),'local_normals'),
            (mesh.GetExtentAttr(),'local_extent'),(mesh.GetFaceVertexCountsAttr(),'counts'),
            (mesh.GetFaceVertexIndicesAttr(),'indices')):
        assert np.array_equal(np.asarray(attr.Get()),arrays[key])
    assert mesh.GetNormalsInterpolation()=='vertex'
    actual_matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    assert np.array_equal(actual_matrix,arrays['world_matrix'])
    assert np.array_equal((np.c_[np.asarray(mesh.GetPointsAttr().Get(),float),np.ones(len(local))]@actual_matrix)[:,:3],
        arrays['rounded_world_points_m'])


def test_typed_archive_geometry_uses_USD_ROW_vector_transform_not_an_invented_frame():
    row=archived_triangle();before=copy.deepcopy(row)
    meta,arrays=mission.archived_mesh_geometry(row,dict(minimum_m=[1.,2.,.01],maximum_m=[1.2,2.2,.2]))
    assert row==before and meta['projection']=='AUTHORED_STATIC_TRIANGLE_SLAB'
    assert np.array_equal(arrays['points'],[[1.,2.,.01],[1.2,2.,.01],[1.,2.2,.2]])
    assert np.array_equal(arrays['indices'],[0,1,2])


@pytest.mark.parametrize('kind',['unknown_approximation','convexHull','holes','time_samples','not_mesh'])
def test_unknown_or_approximated_geometry_is_not_silently_treated_as_a_triangle_aperture(kind):
    row=archived_triangle()
    if kind=='unknown_approximation':row['attributes'].pop('physics:approximation')
    elif kind=='convexHull':row['attributes']['physics:approximation']['value']='convexHull'
    elif kind=='holes':row['attributes']['holeIndices']['value']=typed([0],'IntArray')
    elif kind=='time_samples':row['attributes']['points']['time_samples']=[[1.,row['attributes']['points']['value']]]
    else:row['type']='Cube'
    meta,_=mission.archived_mesh_geometry(row,dict(minimum_m=[1.,2.,.01],maximum_m=[1.2,2.2,.2]))
    assert meta['projection']=='RETAIN_ORIGINAL_AABB'


@pytest.mark.parametrize('kind',['outside_bound','bad_index','nonfinite','perspective','face_count','matrix_shape'])
def test_bad_archived_geometry_fails_instead_of_becoming_an_empty_mesh(kind):
    row=archived_triangle();bound=dict(minimum_m=[1.,2.,.01],maximum_m=[1.2,2.2,.2])
    if kind=='outside_bound':bound['maximum_m'][0]=1.1
    elif kind=='bad_index':row['attributes']['faceVertexIndices']['value']=typed([0,1,3],'IntArray')
    elif kind=='nonfinite':row['attributes']['points']['value']['values'][0]['values'][0]=float('nan')
    elif kind=='perspective':row['world_matrix']['values'][0]['values'][3]=1.
    elif kind=='face_count':row['attributes']['faceVertexCounts']['value']=typed([4],'IntArray')
    else:row['world_matrix']=typed([1.,2.,3.],'Vec3d')
    with pytest.raises(ValueError):mission.archived_mesh_geometry(row,bound)


def projection_fixture(tmp_path):
    path=tmp_path/'authored_coverage_grid.npz';shape=(100,100);old=np.zeros(shape,bool);old[19:82,39:62]=True
    supported=np.ones(shape,bool);np.savez(path,occupied=old,supported_full_cells=supported,obstacle_AABBs=old)
    bound=dict(minimum_m=[.4,.2,.01],maximum_m=[.6,.8,.2])
    coverage=dict(coverage_grid=dict(raster=dict(origin_xy_m=[0.,0.],resolution_m=.01,shape_rows_columns=list(shape)),
        analysis_roi_xy_m=[0.,1.,0.,1.],
        included_obstacles=[dict(path='/World/FIXTURE_WALL',world_aligned_bound=bound)],original_height_slab_m=[.006,.450]),
        output_sha256={str(path):'SOFTWARE_ONLY_NOT_USED_BY_PURE_PROJECTOR'})
    arrays=dict(m_points=np.array([[.4,.2,.01],[.6,.2,.01],[.4,.25,.2]]),m_counts=np.array([3]),m_indices=np.array([0,1,2]))
    bundle=dict(primitives=[dict(path='/World/FIXTURE_WALL',array_prefix='m',source_world_aligned_bound=bound,
        geometry=dict(projection='AUTHORED_STATIC_TRIANGLE_SLAB',recorded_physics_approximation='none',reason='SOFTWARE_FIXTURE'))])
    return coverage,bundle,arrays


def test_triangle_projection_retains_actual_faces_and_unknown_AABB_never_turns_into_a_hole(tmp_path):
    coverage,bundle,arrays=projection_fixture(tmp_path)
    summary,blocked,supported,obstacles=mission.refine_recorded_mesh_grid(coverage,bundle,arrays)
    assert obstacles[21,45] and not obstacles[60,45] and summary['removed_AABB_only_cells']>0
    assert supported.all() and summary['triangle_projected_primitives']==1
    bundle['primitives'][0]['geometry']=dict(projection='RETAIN_ORIGINAL_AABB',reason='UNKNOWN_COOKING_POLICY')
    summary,blocked,_,_=mission.refine_recorded_mesh_grid(coverage,bundle,{})
    assert blocked[60,45] and summary['retained_AABB_primitives']==1
    assert summary['no_H6_task_route_cooked_support_or_drive_admission']


def test_triangle_refinement_never_erases_unsupported_cells_overlapping_an_old_AABB(tmp_path):
    coverage,bundle,arrays=projection_fixture(tmp_path)
    path=next(iter(coverage['output_sha256']))
    with np.load(path) as a:original={k:a[k].copy() for k in a.files}
    original['supported_full_cells'][60,45]=False  # AABB-only gap, not actual triangle.
    original['occupied']=original['obstacle_AABBs']|~original['supported_full_cells']
    np.savez(path,**original)
    summary,blocked,supported,obstacles=mission.refine_recorded_mesh_grid(coverage,bundle,arrays)
    assert not obstacles[60,45] and not supported[60,45] and blocked[60,45]
    assert summary['fixed_blockers_recovered_vs_legacy_union_subtraction']==1


def test_refinement_refuses_an_unexplained_original_blocked_cell(tmp_path):
    coverage,bundle,arrays=projection_fixture(tmp_path)
    path=next(iter(coverage['output_sha256']))
    with np.load(path) as a:original={k:a[k].copy() for k in a.files}
    original['occupied'][5,5]=True;np.savez(path,**original)
    with pytest.raises(ValueError,match='Original complete'):
        mission.refine_recorded_mesh_grid(coverage,bundle,arrays)


def test_corrected_projection_reuses_mesh_arrays_and_preserves_old_report_without_archive_extraction(tmp_path):
    import hashlib,json
    coverage,bundle,arrays=projection_fixture(tmp_path)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    coverage['contract']=dict(anchors=[dict(id='SOFTWARE',world_xyyaw=[.5,.5,0.])],
        same_three_source_support_rectangles=[rectangle(0.,1.,0.,1.)])
    coverage['source_control_review']={'SOFTWARE_ONLY':True}
    coverage['H3_environment_comparison']={'SOFTWARE_ONLY':True}
    coverage_path=tmp_path/'coverage.json';mission.evidence.write_exclusive(coverage_path,coverage)
    numeric=tmp_path/'recorded_world_mesh_arrays.npz';np.savez(numeric,**arrays)
    bundle.update(contract=dict(source_coverage=str(coverage_path),source_coverage_sha256=sha(coverage_path)),
        input_sha256={str(coverage_path):sha(coverage_path)},output_sha256={str(numeric):sha(numeric)})
    bundle_path=tmp_path/'mesh_geometry_bundle.json';mission.evidence.write_exclusive(bundle_path,bundle)
    old=tmp_path/'mesh_projection_review.json'
    mission.evidence.write_exclusive(old,dict(version='HISTORICAL_SOFTWARE_IMPLEMENTATION_NOT_A_NEW_RESULT',output_sha256={}))
    old_bytes=old.read_bytes();source_bytes=numeric.read_bytes()
    result=mission.export_mesh_projection(coverage_path,bundle_path)
    assert result.parent.name=='corrected_fixed_blockers_v2'
    report=json.loads(result.read_bytes())
    assert report['historical_projection_preserved']['sha256']==sha(old)
    assert old.read_bytes()==old_bytes and numeric.read_bytes()==source_bytes
    assert mission.export_mesh_projection(coverage_path,bundle_path)==result
    for path,value in report['output_sha256'].items():assert sha(mission.Path(path))==value


@pytest.mark.parametrize('fault',['missing','unknown_policy','bad_bound','convex_policy','shifted_points','wrong_topology'])
def test_projection_does_not_hide_missing_or_changed_geometry(tmp_path,fault):
    coverage,bundle,arrays=projection_fixture(tmp_path)
    if fault=='missing':bundle['primitives']=[]
    elif fault=='unknown_policy':bundle['primitives'][0]['geometry']['projection']='UNREVIEWED'
    elif fault=='bad_bound':bundle['primitives'][0]['source_world_aligned_bound']=dict(minimum_m=[0.,0.,0.],maximum_m=[1.,1.,1.])
    elif fault=='convex_policy':bundle['primitives'][0]['geometry']['recorded_physics_approximation']='convexHull'
    elif fault=='shifted_points':arrays['m_points'][:,2]+=100.
    else:arrays['m_counts']=np.array([4])
    with pytest.raises(ValueError):mission.refine_recorded_mesh_grid(coverage,bundle,arrays)


def test_large_original_archive_reader_is_blocked_in_Kit_before_any_file_access(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='OFFLINE ONLY'):
        mission.export_archived_mesh_bundle('/NO_FILE_SHOULD_BE_OPENED','/NO_FILE_SHOULD_BE_OPENED')


def medical_fixture():
    root='/World/Hospital/SM_HospitalBed_SOFTWARE_ONLY'
    rows=[row(root+'/leg_left',minimum=(.2,.2,.001),maximum=(.3,.8,.9)),
        row(root+'/leg_right',minimum=(.7,.2,.001),maximum=(.8,.8,.9)),
        # This section lies above the robot slab but remains part of the
        # asset envelope. Do not silently drop it when combining sections.
        row(root+'/mattress',minimum=(.15,.15,.8),maximum=(.85,.85,1.))]
    prims=[dict(path=r['path'],world_aligned_bound=copy.deepcopy({k:v for k,v in r['world_aligned_bound'].items() if k!='size_m'})) for r in rows]
    bounds=dict(minimum_m=[.15,.15,.001],maximum_m=[.85,.85,1.])
    receipt=dict(actual_assets=[dict(path=root,active=True,loaded=True,unloaded_payload_paths=[],enabled_collision_prims=prims)],
        cached_vs_actual=dict(candidate_count=1,all_cached_enabled_collision_paths_retained=True,
            maximum_cached_vs_actual_union_AABB_difference_m=0.,candidates=[dict(path=root,name_based_family='hospital_bed',
                actual_enabled_collision_count=3,cached_enabled_collision_count=3,same_cached_enabled_collision_paths=True,
                maximum_cached_vs_actual_union_AABB_difference_m=0.,actual_union_AABB=bounds)]))
    grid=build(rows)[0]
    return inventory(rows),receipt,grid


def test_medical_asset_leg_gaps_and_above_slab_sections_are_not_driving_space():
    inv,receipt,grid=medical_fixture();before=copy.deepcopy((inv,receipt,grid))
    report,mask=mission.medical_asset_keepouts(inv,receipt,grid)
    assert (inv,receipt,grid)==before
    assert mask[50,50] and mask[16,16] and not mask[10,10]
    assert report['all_original_medical_roots_verified']==1 and report['in_ROI_full_asset_keepouts']==1
    assert report['same_original_10mm_robot_footprint_clearance_NO_added_asset_inflation']
    assert report['same_map_policy_required_for_all_comparison_methods']
    assert not report['actual_motion_authorized']


@pytest.mark.parametrize('fault',['missing_section','extra_section','changed_bound','nonfinite','unknown_family',
    'stale_union','duplicate_root','duplicate_primitive','inactive','unloaded','unbounded_raster','disabled_count','wrong_slab'])
def test_medical_keepout_never_substitutes_stale_or_incomplete_asset_bounds(fault):
    inv,receipt,grid=medical_fixture()
    asset=receipt['actual_assets'][0];family=receipt['cached_vs_actual']['candidates'][0]
    if fault=='missing_section':asset['enabled_collision_prims'].pop()
    elif fault=='extra_section':asset['enabled_collision_prims'].append(dict(path=asset['path']+'/UNKNOWN',world_aligned_bound={}))
    elif fault=='changed_bound':asset['enabled_collision_prims'][0]['world_aligned_bound']['maximum_m'][0]+=.01
    elif fault=='nonfinite':inv['collision_prims'][0]['world_aligned_bound']['minimum_m'][0]=float('nan')
    elif fault=='unknown_family':family['name_based_family']='ASSUME_ANYTHING_IS_A_BED'
    elif fault=='stale_union':family['actual_union_AABB']['maximum_m'][0]+=.01
    elif fault=='duplicate_root':receipt['actual_assets'].append(copy.deepcopy(asset))
    elif fault=='duplicate_primitive':asset['enabled_collision_prims'].append(copy.deepcopy(asset['enabled_collision_prims'][0]))
    elif fault=='inactive':asset['active']=False
    elif fault=='unloaded':asset['unloaded_payload_paths']=['/UNREAD']
    elif fault=='unbounded_raster':grid['raster']['shape_rows_columns']=[100000,100000]
    elif fault=='disabled_count':inv['collision_prims'][0]['collision_enabled_value']=False
    else:grid['original_height_slab_m']=[.03,.45]
    with pytest.raises((ValueError,KeyError)):mission.medical_asset_keepouts(inv,receipt,grid)


def test_medical_assets_outside_new_ROI_remain_in_the_source_roster_not_a_false_keepout():
    inv,receipt,grid=medical_fixture()
    grid['analysis_roi_xy_m']=[2.,3.,2.,3.];grid['raster']['origin_xy_m']=[2.,2.]
    report,mask=mission.medical_asset_keepouts(inv,receipt,grid)
    assert report['all_original_medical_roots_verified']==1 and report['in_ROI_full_asset_keepouts']==0
    assert not mask.any() and len(report['assets'])==1


def test_medical_review_export_is_separate_pinned_repeatable_and_never_changes_source_maps(tmp_path,monkeypatch):
    import hashlib,json
    from pathlib import Path
    inv,assets,grid=medical_fixture();sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def save(name,value):
        path=tmp_path/name;mission.evidence.write_exclusive(path,value);return path
    fingerprint=dict(row_count=3,sha256='SOFTWARE_FIXTURE_NOT_ACTUAL_ENVIRONMENT')
    raw=save('all_stage.json',dict(inventory=inv,all_4997_protected_environment_rows=fingerprint,retained_result='SOFTWARE_ONLY'))
    assets.update(input_sha256={str(raw):sha(raw)},all_4997_protected_environment_rows=fingerprint,retained_result='SOFTWARE_ONLY')
    asset_path=save('assets.json',assets)
    coverage=save('coverage.json',dict(coverage_grid=grid,
        contract=dict(all_stage_receipt=str(raw),source_4997_row_environment=fingerprint,
            same_three_source_support_rectangles=[rectangle(0.,1.,0.,1.)],anchors=[dict(id='fixture',world_xyyaw=[.5,.5,0.])]),
        input_sha256={str(raw):sha(raw)},output_sha256={}))
    original=tmp_path/'mesh_refined_grid.npz'
    old=np.zeros((100,100),bool);old[20:80,20:30]=True
    np.savez(original,occupied=old,supported_full_cells=np.ones_like(old),obstacle_mesh_projection=old)
    projection=save('projection.json',dict(source_coverage=str(coverage),
        input_sha256={str(raw):sha(raw),str(coverage):sha(coverage)},output_sha256={str(original):sha(original)}))
    before={str(p):sha(p) for p in (raw,asset_path,coverage,projection,original)}
    monkeypatch.setattr(mission,'GATE',tmp_path/'OUTPUT_SOFTWARE_ONLY')
    result=mission.export_medical_keepout_review(coverage,projection,asset_path)
    report=json.loads(result.read_bytes())
    assert report['newly_blocked_furniture_gap_cells']>0 and not report['actual_motion_authorized']
    assert mission.export_medical_keepout_review(coverage,projection,asset_path)==result
    assert all(sha(path)==value for path,value in before.items())
    for path,value in {**report['input_sha256'],**report['output_sha256']}.items():assert sha(path)==value
    with np.load(result.parent/'medical_keepout_grid.npz') as saved:
        assert saved['occupied'][50,50] and np.array_equal(saved['supported_full_cells'],np.ones_like(old))
    # Tamper ONLY with the temporary SOFTWARE fixture. Validation belongs
    # to original inputs too, not just the existence of the output JSON.
    np.savez(original,occupied=~old)
    with pytest.raises(ValueError,match='Original saved source changed'):
        mission.export_medical_keepout_review(coverage,projection,asset_path)

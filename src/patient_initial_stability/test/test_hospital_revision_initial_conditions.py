from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_initial_conditions as initial


def test_six_tasks_have_five_paired_joint_draws_for_B2_P():
    p=initial.definition()
    assert len(p['main_cells'])==60
    pairs={c['paired_condition_id'] for c in p['main_cells']}
    assert len(pairs)==30
    for pair in pairs:
        rows=[c for c in p['main_cells'] if c['paired_condition_id']==pair]
        assert len(rows)==2
        assert rows[0]['initial_joint_angles_deg']==rows[1]['initial_joint_angles_deg']
        assert rows[0]['start_xyyaw']==rows[1]['start_xyyaw']
        assert rows[0]['goal_xyyaw']==rows[1]['goal_xyyaw']
    for c in p['main_cells']:
        assert all(abs(c['initial_joint_angles_deg'][n])<=.5 for n in initial.ACTIVE)
        assert all(abs(c['initial_joint_angles_deg'][n])<=20 for n in initial.CASTERS)
    assert p==initial.definition()


def test_supplement_keeps_prior_negative_trials_and_does_not_claim_new_plan_timing():
    p=initial.definition()
    assert p['old_negative_trials_NOT_removed_replaced_or_relabelled']
    assert p['no_new_independent_planning_time_measurement_claimed']
    assert p['existing_actual_static_and_pre_hold_may_change_angles_and_are_NOT_bypassed']


def test_source_overlay_only_authors_declared_states_and_never_rewrites_original(tmp_path):
    from pxr import Usd
    source=Path(initial.runtime.shared.preparation.old.constructor.ASSET)
    before=initial.runtime.sha(source)
    angles=initial.definition()['main_cells'][0]['initial_joint_angles_deg']
    path=tmp_path/'seed.usda'
    receipt=initial.seed_asset(path,angles)
    assert initial.runtime.sha(source)==before==receipt['original_sha256']
    stage=Usd.Stage.Open(str(path))
    from pxr import PhysxSchema
    for name,value in angles.items():
        prim=stage.GetPrimAtPath(initial.runtime.shared.preparation.old.constructor.OLD_ROBOT+'/joints/'+name)
        state=PhysxSchema.JointStateAPI.Get(prim,'angular')
        assert abs(state.GetPositionAttr().Get()-value)<1e-6
        assert state.GetVelocityAttr().Get()==0
    # Root contains a reference plus the initial-state opinions, not a new
    # mass/collider/friction/drive configuration copied from another robot.
    text=stage.GetRootLayer().ExportToString()
    for token in ('physics:mass','physics:staticFriction','drive:angular:physics:stiffness'):
        assert token not in text


def test_private_source_reference_validates_all_original_mass_geometry_and_drive_settings(tmp_path):
    cell=initial.definition()['main_cells'][0]
    asset=tmp_path/'seed.usda'
    initial.seed_asset(asset,cell['initial_joint_angles_deg'])
    geometry=initial.runtime.read(initial.runtime.read(initial.campaign.PROTOCOL)['tasks'][cell['task']]['original_geometry'])
    context=initial.runtime.context_for(tmp_path/'trial','/World/patient_transport_seed_test',geometry)
    initial.private_source_context(context,asset)
    reference=context.initial_condition_source_helpers.fresh_source_reference(context)
    made=reference['source_constructor_readback']
    assert made['simulation_mass_kg']==265
    assert len(made['revolute_joint_names'])==12
    assert len(made['rigid_bodies'])==13
    assert len(reference['fixed_configuration_rows'])==145
    for name,value in cell['initial_joint_angles_deg'].items():
        states=made['actual_source_authored_joint_seed'][name]
        assert abs(states['state:angular:physics:position']-value)<1e-6

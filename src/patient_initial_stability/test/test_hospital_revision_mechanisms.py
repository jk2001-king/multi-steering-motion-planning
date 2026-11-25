
from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
from dataclasses import asdict
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_mechanisms as mechanism
from patient_initial_stability.steering_lattice import State


@pytest.fixture
def setup():
    profile = dict(geometry=dict(wheelbase_m=1.330, front_length_m=.665, rear_length_m=.665),
        mode_lateral_response_scale={'front_only':1.00542},
        mode_curvature_response_scale={'front_only':.995386})
    kernel = SimpleNamespace(steering_targets=lambda mode,k:(math.atan(1.330*k),0.))
    state = State(1.,2.,.3,'rear_only',.4,-.2,True,-1,.02,.01,.03)
    return state,kernel,profile


@pytest.mark.parametrize('policy,changed', [
    ('carry',set()),('steady_steering',{'front','rear'}),
    ('steady_velocity',{'longitudinal_velocity','lateral_velocity','yaw_rate'}),
    ('steady_both',{'front','rear','longitudinal_velocity','lateral_velocity','yaw_rate'})])
def test_only_declared_predictive_state_components_change(setup,policy,changed):
    state,kernel,profile = setup
    before = asdict(state)
    got = mechanism.projected_entry(state,'front_only',.5,1,kernel=kernel,
        profile=profile,target_speed_m_s=.03,history_policy=policy)
    assert {k for k,v in asdict(got).items() if v!=before[k]} == changed
    assert asdict(state)==before
    if policy=='carry':
        assert got is state
    if policy in ('steady_velocity','steady_both'):
        assert got.longitudinal_velocity==.03  # NOT repeated start from rest


def test_equilibrium_targets_include_same_bias_and_geometry(setup):
    state,kernel,profile=setup
    got=mechanism.projected_entry(state,'front_only',.5,-1,kernel=kernel,
        profile=profile,target_speed_m_s=.03,history_policy='steady_both')
    assert got.front==pytest.approx(math.atan(1.330*.5)-math.radians(1))
    assert got.rear==pytest.approx(-math.radians(1))
    assert got.yaw_rate==pytest.approx(-.03*.995386*(math.tan(got.front)-math.tan(got.rear))/1.330)
    assert got.lateral_velocity==pytest.approx(-.03*1.00542*(math.tan(got.front)+math.tan(got.rear))/2)


def test_baseline_without_dynamic_model_is_never_changed(setup):
    state,kernel,profile=setup
    seen=[]
    def original(s,*args,**kwargs):
        seen.append(s)
        return s,[],1.
    wrapped=mechanism.bind_propagation(original,kernel=kernel,profile=profile,
        history_policy='steady_both')
    wrapped(state,'front_only',.5,1,rigid_body_model=None,target_speed_m_s=.03)
    assert seen[0] is state


def test_new_ablation_does_not_turn_started_false_into_true(setup):
    state,kernel,profile=setup
    from dataclasses import replace
    with pytest.raises(ValueError,match='initialized'):
        mechanism.projected_entry(replace(state,started=False),'front_only',.5,1,
            kernel=kernel,profile=profile,target_speed_m_s=.03,history_policy='steady_both')


def test_resource_history_fields_are_retained_in_augmented_state(setup):
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class ResourceState(State):
        mission_reverse_total_bins:int=8
        mission_reverse_episode_bins:int=3
        mission_direction_changes:int=2
    state,kernel,profile=setup
    augmented=ResourceState(**asdict(state))
    got=mechanism.projected_entry(augmented,'front_only',.5,1,kernel=kernel,
        profile=profile,target_speed_m_s=.03,history_policy='steady_both')
    assert isinstance(got,ResourceState)
    assert (got.mission_reverse_total_bins,got.mission_reverse_episode_bins,got.mission_direction_changes)==(8,3,2)


def test_python_and_native_interventions_match_actual_1330_profile():
    import json
    import numpy as np
    import audit_hospital_paper_readiness as shared
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    root=_ijcas_workspace()
    reference=_ijcas_path(root, 'studies/IJCAS/results/hospital_paper_gate/revision_campaign_20261121/planning/H2_P_dynamics_full_envelope_s03/P_dynamics_full_envelope.json')
    plan=json.loads(reference.read_text())
    profile=plan['geometry_profile']
    connections=shared.unified_wheelbase_connections(profile,include_compiled=True)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**plan['rigid_body_parameter_bounds']))
    state=State(0.,0.,0.,'rear_only',.1,-.2,True,1,.025,.002,.01)
    for policy in mechanism.HISTORY_POLICIES:
        py=mechanism.bind_propagation(connections.python.propagate,kernel=connections.python,
            profile=profile,history_policy=policy)
        native=mechanism.bind_propagation(connections.compiled,kernel=connections.python,
            profile=profile,history_policy=policy)
        for mode in ('front_only','rear_only','coordinated_counterphase'):
            for direction in (-1,1):
                a,poses_a,ta=py(state,mode,.5,direction,rigid_body_model=model,
                    speed_m_s=.05,target_speed_m_s=.03)
                b,poses_b,tb=native(state,mode,.5,direction,rigid_body_model=model,
                    speed_m_s=.05,target_speed_m_s=.03)
                assert ta==pytest.approx(tb,abs=1e-12)
                assert np.asarray(poses_a)==pytest.approx(np.asarray(poses_b),abs=1e-12)
                assert asdict(a)==pytest.approx(asdict(b),abs=1e-12)

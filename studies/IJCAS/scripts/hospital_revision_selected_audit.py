"""Read-only provenance/initialization audit of the final selected n3 block.

Requires the complete block and the already approved camera replacement.
Never dispatches a simulator command or changes an experiment record.
"""
import argparse
import csv
import math
from pathlib import Path

import hospital_revision_selected_results as selected
from hospital_controlled_comparison import validate_profile

runtime = selected.runtime
OUTPUT = selected.OUTPUT


def sample_record(sample, names):
    if sample is None:
        return None
    return dict(time_s=float(sample['time_s']), phase=sample['phase'],
        base_link_world_xyyaw=[float(sample[k]) for k in ('x_m', 'y_m', 'yaw_rad')],
        command_speed_m_s=float(sample['command_speed_m_s']),
        joint_angles_deg={name: math.degrees(float(sample['joint_' + name + '_rad']))
            for name in names if 'joint_' + name + '_rad' in sample})


def audit():
    if not (OUTPUT / 'analysis_completed.json').is_file():
        raise ValueError('Final selected analysis, including the technical replacement, must finish first')
    block = selected.experiment.campaign()
    snapshot = selected.snapshot()
    protocol = runtime.read(block.PROTOCOL)
    cells = {c['cell_id']: c for c in runtime.read(block.EXECUTION_PROTOCOL)['main_cells']}
    if len(snapshot['rows']) != 78 or len(cells) != 78:
        raise ValueError('All declared conditions required')
    observations = []
    pins = {}
    h6 = []
    models = []
    speed_profiles = []
    recomputed = []
    for row in snapshot['rows']:
        cell = cells[row['cell_id']]
        value = runtime.read(row['outcome_path'])
        pins[row['outcome_path']] = runtime.sha(row['outcome_path'])
        summary = runtime.read(row['planning_summary'])
        if summary['plan_success']:
            planned_path = Path(summary['plan'])
            planned = runtime.read(planned_path)
            speed_profiles.append(dict(cell_id=row['cell_id'], plan=str(planned_path),
                plan_sha256=runtime.sha(planned_path), **validate_profile(planned)))
        if row['method'] == 'P_dynamics_full_envelope':
            plan_path = Path(summary['plan'])
            plan = runtime.read(plan_path)
            if plan['fixed_P_lookahead_edges_all_tasks'] != 4:
                raise ValueError('Nonuniform P/history lookahead: ' + row['cell_id'])
            models.append(dict(cell_id=row['cell_id'], plan=str(plan_path),
                plan_sha256=runtime.sha(plan_path), lookahead_edges=4))
        record = dict(cell_id=row['cell_id'],
            paired_condition_id=cell['paired_condition_id'],
            prescribed_source_angles_deg=cell['initial_joint_angles_deg'],
            prescribed_start_xyyaw=cell['start_xyyaw'],
            model_initial_active_steering_deg=[0., 0.],
            selected_outcome=row['outcome_path'], completed=row['completed'],
            approved_technical_replacement=row.get('approved_technical_replacement', False))
        if not row.get('report'):
            record.update(status='NOT_MEASURED_NO_PHYSICAL_ROUTE', first_bound=None,
                first_route_sample=None, first_tracking_sample=None, first_nonzero_command_sample=None)
            observations.append(record)
            continue
        raw = runtime.read(row['report'])
        packet = runtime.read(raw['protocol'])
        if (runtime.sha(raw['protocol']) != raw['protocol_sha256']
                or runtime.sha(row['report']) != value['metrics']['report_sha256']
                or runtime.sha(raw['trace']) != raw['trace_sha256']
                or raw['trace_sha256'] != value['metrics']['trace_sha256']
                or packet['primary_cell'] != cell
                or raw['controller_candidate_version'] != block.control.VERSION
                or raw['predictive_history_policy'] != cell['predictive_history_policy']):
            raise ValueError('Selected report/protocol/trace/controller provenance mismatch: ' + row['cell_id'])
        actual_metrics = block.metrics.metrics(row['report'])
        numeric_fields = ('sample_count', 'tracking_sample_count', 'tracking_duration_s',
            'tracking_position_RMS_mm', 'tracking_yaw_RMS_deg',
            'maximum_tracking_position_error_mm', 'maximum_absolute_tracking_yaw_error_deg',
            'final_position_error_mm', 'final_yaw_error_deg', 'actual_distance_m',
            'controller_elapsed_simulation_s')
        for name in numeric_fields:
            observed, saved = actual_metrics[name], value['metrics'][name]
            if ((observed is None) != (saved is None)
                    or (observed is not None and not math.isclose(observed, saved,
                        rel_tol=1e-12, abs_tol=1e-9))):
                raise ValueError('Recomputed metric differs: ' + row['cell_id'] + '/' + name)
        recomputed.append(dict(cell_id=row['cell_id'], compared_fields=list(numeric_fields),
            report_sha256=actual_metrics['report_sha256'], trace_sha256=actual_metrics['trace_sha256']))
        limits = protocol['tasks'][row['task']]
        if (packet['simulation_budget_s'] != limits['simulation_budget_s']
                or packet['maximum_wall_time_s'] != limits['wall_budget_s']):
            raise ValueError('Nonuniform declared task budget: ' + row['cell_id'])
        if row['task'] == 'H6':
            h6.append(dict(cell_id=row['cell_id'], protocol=raw['protocol'],
                simulation_budget_s=packet['simulation_budget_s'],
                maximum_wall_time_s=packet['maximum_wall_time_s']))
        if not all(raw['checks'].get(k) is True for k in (
                'final_zero_target_sent', 'actual_Pause_confirmed', 'original_gains_targets_restored')):
            raise ValueError('Selected physical route lacks confirmed safe retirement: ' + row['cell_id'])
        initial_path = Path(row['report']).parent.parent / 'static/initial_joint_initialization_readback.json'
        initial = runtime.read(initial_path)
        if (initial['prescribed_new_source_angles_deg'] != cell['initial_joint_angles_deg']
                or initial['passive_initialization_verified'] is not True):
            raise ValueError('Actual initialization not matched to its declared draw: ' + row['cell_id'])
        record['first_bound'] = {k: initial[k] for k in ('time_s',
            'prescribed_new_source_angles_deg', 'first_bound_physics_angles_deg',
            'wrapped_difference_deg', 'passive_initialization_verified')}
        record['first_bound_source'] = str(initial_path)
        pins[str(initial_path)] = runtime.sha(initial_path)
        first = tracking = moving = None
        with Path(raw['trace']).open(newline='') as stream:
            for sample in csv.DictReader(stream):
                if first is None:
                    first = sample
                if tracking is None and sample['phase'] == 'path_tracking':
                    tracking = sample
                if moving is None and abs(float(sample['command_speed_m_s'])) > 1e-9:
                    moving = sample
                if first is not None and tracking is not None and moving is not None:
                    break
        names = tuple(cell['initial_joint_angles_deg'])
        record.update(status='EXTRACTED_NOT_INVENTED',
            first_route_sample=sample_record(first, names),
            first_tracking_sample=sample_record(tracking, names),
            first_nonzero_command_sample=sample_record(moving, names),
            report=row['report'], report_sha256=runtime.sha(row['report']),
            trace=raw['trace'], trace_sha256=raw['trace_sha256'])
        observations.append(record)
    if len(h6) != 12 or len(models) != 24:
        raise ValueError('Expected twelve selected H6 routes and eighteen P plus six history plans')
    original = runtime.read(selected.replacement.primary.OUTPUT /
        'paused_boundary_GET' / (selected.replacement.CELL + '.json'))
    return dict(status='ALL_SELECTED78_PROVENANCE_INITIALIZATION_AND_COMMON_LIMITS_VERIFIED',
        selected_condition_count=78, original_technical_camera_attempt_preserved=True,
        selected_technical_replacements=snapshot['technical_replacements'],
        original_camera_proof_status=original['status'],
        current_controller_version=block.control.VERSION,
        H6_selected_all12_common_limits=h6, P_and_history_all24_common_lookahead=models,
        method_blind30_braking_profile_checks=speed_profiles,
        selected_actual_metrics_recomputed_from_native_traces=recomputed,
        initialization_observations=observations, input_sha256=pins,
        note='Source draw, first-BOUND, route entry and moving states are distinct. '
             'No retrospective exact-angle identity after physical settling is claimed.',
        actual_live_state_requery_NOT_performed_by_this_offline_audit=True,
        simulator_commands=0, new_experiments=0)


def live_get():
    """One post-completion GET of the actual owner, Pause and full tensor."""
    if not (OUTPUT / 'analysis_completed.json').is_file():
        raise ValueError('Do not query a final boundary before the approved queue finishes')
    from hospital_revision_runner import remote
    source = '''import builtins,json
import hospital_revision_camera_replacement as _r
from pxr import UsdPhysics
_r.verify()
_finish=_r.runtime.read(_r.OUTPUT/'completed.json')
_job=getattr(builtins,_r.KEY,None)
assert _job is not None and _job.done() and not _job.cancelled(),'Replacement owner unfinished'
assert _job.exception() is None,'Replacement owner raised an exception'
assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is _job,'Another task owns Isaac'
_raw=_r.runtime.read(_finish['last_actual_report'])
_packet=_r.runtime.read(_raw['protocol'])
_sdk=_r.runtime.retained.runtime.live_sdk()
_stage=_sdk.omni.usd.get_context().get_stage()
_timeline=_sdk.omni.timeline.get_timeline_interface()
_state=_r.runtime.evidence.timeline_state(_timeline)
assert _state==_raw['actual_final_timeline'] and not _state['playing'] and not _state['stopped'],'Actual Pause/clock mismatch'
_view=_sdk.SimulationManager.get_physics_sim_view().create_articulation_view(_packet['actor']+'/base_link')
_tensor=_r.runtime.shared.preparation.old.tensor_snapshot(_view)
assert _tensor==_raw['actual_final_readback']['tensor'],'Current saved full tensor mismatch'
_original=_packet['required_tensor_snapshot']
assert all(_tensor[k]==_original[k] for k in ('names','link_names','gains','joint_position_targets_rad','joint_velocity_targets_rad_s')),'Original motor settings/targets mismatch'
_bodies=[str(p.GetPath()) for p in _stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
assert len(_bodies)==13 and all(p.startswith(_packet['actor']+'/') for p in _bodies),'Retained actor identities changed'
assert _r.runtime.evidence.timeline_state(_timeline)==_state,'Timeline changed during GET'
assert _r.runtime.shared.preparation.old.tensor_snapshot(_view)==_tensor,'Tensor changed during GET'
_proof=dict(status='FINAL_ACTUAL_OWNER_PAUSE_FULL13LINK12JOINT_TENSOR_AND_MOTOR_RESTORATION_GET_VERIFIED',
 current_native_owner=_r.KEY,owner_done=True,owner_exception=None,latest_owner_exact=True,
 actor=_packet['actor'],actual_timeline=_state,body_paths=_bodies,tensor=_tensor,
 exact_saved_tensor=True,original_all12_gains_effort_limits_targets_exact=True,
 final_report=_finish['last_actual_report'],final_report_sha256=_r.runtime.sha(_finish['last_actual_report']),
 independent_4997_environment_requery_NOT_performed_in_this_short_GET=True,
 no_Play_Stop_Pause_reset_camera_scene_gain_target_or_motion_command=True)
_path=_r.OUTPUT/'final_live_GET.json'
_r.runtime.evidence.write_exclusive(_path,_proof)
print(json.dumps(dict(saved=str(_path),status=_proof['status'],timeline=_state,owner=_r.KEY)))
'''
    return remote(source)


def korean_brief(proof):
    """Generate prose from final aggregates, without modifying the manuscript."""
    finish = runtime.read(OUTPUT / 'analysis_completed.json')
    data = runtime.read(finish['detailed_feedback_analysis'])
    groups = data['task_statistics']
    main = [g for g in groups if g['experiment_role'] == 'MAIN']
    methods = selected.experiment.original.design.original.METHODS
    lines = ['교수님 피드백 보완: 고정 버전 3조건 최종 실험 취합', '',
        '범위: 본 비교 72조건 + H2/H3 이력 제거 6조건. 별도 오프라인 외곽 비교 12건.',
        '외부 카메라 중단 1건은 원본을 보존하고, 명시적으로 승인한 동일 조건 대체 1회 결과를 성공 여부와 무관하게 선택했다.',
        '조건 수를 늘리거나 성능 실패를 성공 재시도로 교체하지 않았다.', '',
        '1. 임무 완료 수(각 방법·과제당 3조건; 계획 실패 포함)',
        'Task | B0 | B1 | B2 | P']
    for task in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
        subset = {g['method']: g for g in main if g['task'] == task}
        lines.append(task + ' | ' + ' | '.join(str(subset[m]['completed_n']) + '/3' for m in methods))
    p = data['P_goal']
    rank = data['ranking_counts']
    lines += ['', f"P 합계: {p['completed_n']}/{p['declared_n']} ({100*p['completion_fraction']:.2f}%).",
        'P의 유효 미완료 기록도 삭제하지 않았다. 18/18 또는 모든 지표의 보편적 우수성을 주장하지 않는다.',
        f"P와 비교군이 함께 완료한 조건 {rank['P_with_at_least_one_completed_baseline']}개 중, 완료 방법들 사이에서 위치 RMS 최저는 {rank['P_best_position_when_comparable']}개, yaw RMS 최저는 {rank['P_best_yaw_when_comparable']}개다.",
        'P만 완료한 조건은 완료 능력의 사례이지, 비교군의 부분 RMS보다 낮다는 증거로 집계하지 않았다.', '',
        '2. 동일 속도 조건',
        f"계획이 존재하는 {len(proof['method_blind30_braking_profile_checks'])}조건 모두 방법 비의존 30 mm/s 순항·동일 거리 기반 감속 규칙을 원 계획에서 재검증했다.",
        '실제 관측 속도가 시시각각 동일하다고 가정하지 않았다. 공통 30 mm/s 명령 구간 분석은 별도 사후 민감도 결과이며, 기본 전체 path_tracking RMS를 대체하지 않는다.']
    for pair in data['paired_completed_only']:
        if pair['task'] == 'H2' and pair['both_completed'] and '_B2_' in pair['reference']:
            effect = pair['relative_reductions_percent']
            lines.append(f"H2의 공동 완료 B2/P 1쌍: 위치 RMS {effect['tracking_position_RMS_mm']:.2f}%, yaw RMS {effect['tracking_yaw_RMS_deg']:.2f}% 감소. 이 1쌍의 효과를 세 조건 전체의 독립 효과로 확대하지 않는다.")
    lines += ['', '3. 반복 조건 및 이력 제거 ablation',
        '과제별 초기 몸체 위치·yaw·능동 조향각·네 캐스터 각도를 달리한 세 조건을 모든 방법에 짝지었다.',
        '완료 수는 모든 선언 조건을 분모로 하고, RMS/주행시간 평균±표본 SD는 완료 실행만 포함하며 유효 n을 표에 함께 썼다. n<2의 SD는 0이 아닌 미정으로 표시한다.',
        '세 개의 제한된 초기조건에 대한 기술 통계이며, 통계적 유의성이나 임상 신뢰도 검증을 주장하지 않는다.',
        '이력 제거는 각 예측 primitive 진입 상태를 현재 명령의 정상상태로 투영하는 개입이다. 조향/속도 이력을 함께 제거하며, 물리 로봇 상태를 중간에 초기화하지 않았다.']
    for task in ('H2', 'H3'):
        full = next(g for g in main if g['task'] == task and g['method'] == 'P_dynamics_full_envelope')
        ablated = next(g for g in groups if g['task'] == task and g['experiment_role'] == 'HISTORY_ABLATION')
        lines.append(f"{task}: 전체 P {full['completed_n']}/3, 이력 제거 {ablated['completed_n']}/3.")
        if full['completed_n'] == ablated['completed_n'] == 3:
            effects = [100*(1-full[k]['mean']/ablated[k]['mean']) for k in
                ('completed_tracking_position_RMS_mm', 'completed_tracking_yaw_RMS_deg')]
            lines.append(f"세 완료 실행의 평균 RMS 비율: 위치 {effects[0]:.2f}%, yaw {effects[1]:.2f}% 감소. 개별 조향/속도 이력의 독립 주효과로 해석하지 않는다.")
    constraints = [r for r in data['force_load_constraints'] if r['experiment_role'] == 'MAIN']
    calls = sum(r['candidate_activity'].get('model_primitive_calls', 0) for r in constraints)
    uf = [r['maximum_selected_friction_utilization'] for r in constraints]
    ltr = [r['maximum_selected_abs_LTR'] for r in constraints]
    exceed = {k: sum(r['candidate_activity'].get(k, 0) for r in constraints) for k in
        ('infeasible_result_count', 'friction_limit_exceedance_count', 'LTR_limit_exceedance_count')}
    lines += ['', '4. Full-envelope 및 Force/Load',
        'H2: 점 근사와 endpoint 근사 모두 3조건에서 독립 full-envelope 충돌 표본 0개(차이가 없는 결과도 포함).',
        'H3: 점 근사 후보 3개는 독립 full-envelope 감사에서 충돌 표본 961/469/1107개로 거부됐다. 위험 후보는 실제 주행하지 않았다.',
        'H3 endpoint 근사 3건은 기존 지도/자원 판정에서 수용 후보를 만들지 못했다. 충돌 표본 수는 미측정이며, 거부 원인을 충돌 하나로 단정하지 않는다.',
        f"P 18계획의 선택 primitive 최대값 범위: 마찰 사용률 {min(uf):.9f}–{max(uf):.9f}, 절댓값 LTR {min(ltr):.9f}–{max(ltr):.9f} (각 한도 0.8).",
        '이는 선택 계획의 모델 전파 표본 지표이며, 실제 주행 종단 보정·정지 유지 구간을 포함하지 않는다.',
        f"후보 모델 평가 {calls}회에서 집계한 활동: {exceed}. 이는 물리 주행 반복 수가 아니다.",
        '이 결과로 Force/Load가 개선의 원인 또는 개별 모터/환자 안전성을 보증한다고 주장하지 않는다. 보조 모델 검사로 한정한다. 한도 이하 비용/순위 효과와 screen-off 경로 차이는 별도 통제 비교가 없어 미확인이다.', '',
        '5. 공정성·추적성·남은 원고 작업',
        'H6의 네 방법 모두 600초 시뮬레이션/1500초 실제시간 한도. P/이력 제거 모델의 lookahead는 H1–H6 공통 4이다. B0–B2에 존재하지 않는 beam 파라미터를 동일 적용했다고 쓰지 않는다.',
        f"선택한 실제 실행 {len(proof['selected_actual_metrics_recomputed_from_native_traces'])}건의 RMS·최대 오차·종단 오차·거리·시간을 원본 trace에서 재계산해 저장 지표와 대조했다.",
        '기존 버전·실패·기술 중단은 원본 폴더에 그대로 있으며, 이전 선택 사례/센서 기능 사례를 새 반복 통계에 섞지 않았다.']
    live = OUTPUT / 'final_live_GET.json'
    lines.append('최종 실제 Pause/13링크·12관절/원래 게인·토크 한도·목표 복원: ' +
        ('별도 읽기 전용 SDK 확인 완료 (' + str(live) + ')' if live.is_file() else '별도 SDK 확인 미완료'))
    lines += ['실물 플랫폼 사진은 저자가 실험 종료 후 제공하기로 하여 아직 미제공이다. 사진을 실제 주행 검증으로 설명하지 않는다.',
        '원고에는 최종 표/그림 반영과 10페이지 편집이 남아 있다. 현재 결과 대기 저자 검토본을 제출 최종본이라고 표시하지 않는다.', '',
        '근거 파일:', str(OUTPUT / 'selection_manifest.json'),
        str(OUTPUT / 'selected_evidence_audit.json'),
        str(Path(finish['analysis_folder']) / 'individual.csv'),
        str(Path(finish['analysis_folder']) / 'summary.csv'),
        str(finish['detailed_feedback_analysis'])]
    return '\n'.join(lines) + '\n'


def closeout():
    """Join append-only analysis and the later independent live proof."""
    selected.replacement.verify()
    analysis = runtime.read(OUTPUT / 'analysis_completed.json')
    offline = runtime.read(OUTPUT / 'selected_evidence_audit.json')
    live = runtime.read(OUTPUT / 'final_live_GET.json')
    if (offline['selected_condition_count'] != 78
            or len(offline['selected_actual_metrics_recomputed_from_native_traces']) != 69
            or live['actual_timeline']['playing'] or live['actual_timeline']['stopped']
            or live['exact_saved_tensor'] is not True
            or live['original_all12_gains_effort_limits_targets_exact'] is not True
            or live['latest_owner_exact'] is not True):
        raise ValueError('Final selected data and independently confirmed actual boundary required')
    data = runtime.read(analysis['detailed_feedback_analysis'])
    evidence = [OUTPUT / name for name in ('completed.json', 'analysis_completed.json',
        'selection_manifest.json', 'selected_evidence_audit.json', 'final_live_GET.json',
        'FINAL_RESULT_BRIEF_KO.txt')]
    evidence += [Path(analysis['detailed_feedback_analysis'])]
    evidence += [Path(analysis['analysis_folder']) / name
        for name in ('results.json', 'individual.csv', 'summary.csv')]
    target = OUTPUT / 'FINAL_EXPERIMENT_CLOSEOUT.json'
    runtime.evidence.write_exclusive(target, dict(
        status='ALL_DECLARED_N3_CONDITIONS_AND_FINAL_PROVENANCE_CHECKS_COMPLETE',
        main_conditions=72, history_conditions=6, offline_envelope_comparisons=12,
        approved_external_camera_replacement_executions=1,
        extra_independent_repetitions=0, remaining_experiments_in_this_frozen_block=0,
        method_completion_counts=data['all_declared_counts'], P_result=data['P_goal'],
        metric_comparison_counts=data['ranking_counts'],
        current_native_owner=live['current_native_owner'], actual_timeline=live['actual_timeline'],
        final_independent_live_GET_requirement_now_satisfied=True,
        earlier_append_only_analysis_receipt_NOT_rewritten=True,
        algorithmic_failures_and_original_technical_attempt_preserved=True,
        read_only_CSV_export_schema_repaired_NOT_a_new_experiment=True,
        incomplete_earlier_readonly_exports_NOT_selected_for_final_tables=True,
        actual_platform_photo_NOT_supplied=True, manuscript_final_results_figures_and10pages_pending=True,
        input_sha256={str(p): runtime.sha(p) for p in evidence},
        reporting_source_sha256={str(p): runtime.sha(p) for p in
            (Path(__file__).resolve(), Path(selected.__file__).resolve())}))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-get', action='store_true')
    args = parser.parse_args()
    if args.live_get:
        print(live_get())
    else:
        target = OUTPUT / 'selected_evidence_audit.json'
        proof = audit()
        runtime.evidence.write_exclusive(target, proof)
        with (OUTPUT / 'FINAL_RESULT_BRIEF_KO.txt').open('x', encoding='utf-8') as stream:
            stream.write(korean_brief(proof))
        print(target)

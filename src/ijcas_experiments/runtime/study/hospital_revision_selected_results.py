"""Read-only final selection with one explicitly approved technical replacement.

All original outcomes stay untouched. The only substituted observation must
be the externally interrupted B1 camera case, with identical frozen plan,
condition, controller and limits. The replacement is used whether it succeeds
or fails; selection never depends on improved performance.
"""
import argparse
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import hospital_revision_camera_replacement as replacement
import hospital_revision_final_comparison_analysis as detailed

runtime = replacement.runtime
experiment = replacement.experiment
OUTPUT = replacement.OUTPUT


def label_replacement(rows, selected_path, original_path):
    """Use a uniform CSV schema, including rows that were not replaced."""
    for row in rows:
        row.update(approved_technical_replacement=False, original_technical_outcome=None)
    chosen = next(r for r in rows if r['cell_id'] == replacement.CELL)
    chosen.update(outcome_path=str(selected_path), approved_technical_replacement=True,
        original_technical_outcome=str(original_path))


def snapshot():
    replacement.verify()
    block = experiment.campaign()
    original = block.results_snapshot()
    if original['status'] != 'COMPLETE' or len(original['rows']) != 78:
        raise ValueError('Every original declared outcome must finish first')
    finish = runtime.read(OUTPUT / 'completed.json')
    value_path = Path(finish['outcome'])
    value = runtime.read(value_path)
    cell = replacement.cell_definition()
    old_path = block.OUTPUT / 'outcomes' / (replacement.CELL + '.json')
    old = runtime.read(old_path)
    if (value.get('technical_replacement_of_cell_id') != replacement.CELL
            or value.get('valid_same_condition_replacement') is not True
            or value.get('independent_repetition_increment') != 0
            or value.get('authority') != replacement.AUTHORITY
            or old.get('technical_interruption') is not True
            or old.get('completion_evaluable') is not False):
        raise ValueError('Only the explicit approved technical replacement is selectable')
    raw = runtime.read(value['report'])
    packet = runtime.read(raw['protocol'])
    old_raw = runtime.read(old['report'])
    old_packet = runtime.read(old_raw['protocol'])
    for key in ('primary_cell', 'geometry_file', 'grid_file', 'simulation_budget_s',
                'maximum_wall_time_s', 'common_tracking_policy', 'common_lifecycle_contract'):
        if packet[key] != old_packet[key]:
            raise ValueError('Replacement changed frozen condition: ' + key)
    if (packet['primary_cell'] != cell
            or raw['controller_candidate_version'] != block.control.VERSION
            or raw['predictive_history_policy'] != cell['predictive_history_policy']
            or runtime.sha(Path(value['report']).parent / packet['plan_filename']) !=
                runtime.sha(Path(old['report']).parent / old_packet['plan_filename'])
            or runtime.sha(value['report']) != value['metrics']['report_sha256']
            or runtime.sha(raw['trace']) != raw['trace_sha256']
            or raw['trace_sha256'] != value['metrics']['trace_sha256']):
        raise ValueError('Exact plan/controller/trace provenance required; no success-only selection')
    runtime.retained.runtime.task.verify_pins(packet['input_sha256'])

    def selected_read(path):
        if Path(path).resolve() == old_path.resolve():
            return copy.deepcopy(value)
        return runtime.read(path)

    private_runtime = SimpleNamespace(**dict(vars(runtime), read=selected_read))
    result = runtime.bind(block.results_snapshot, runtime=private_runtime)()
    label_replacement(result['rows'], value_path, old_path)
    record = dict(cell_id=replacement.CELL, paired_condition_id=cell['paired_condition_id'],
        original_technical_outcome=str(old_path), original_outcome_sha256=runtime.sha(old_path),
        original_report=str(old['report']), original_report_sha256=runtime.sha(old['report']),
        original_trace=old['metrics']['trace'], original_trace_sha256=old['metrics']['trace_sha256'],
        original_camera_proof=old['separate_camera_boundary_GET'],
        selected_replacement_outcome=str(value_path), selected_outcome_sha256=runtime.sha(value_path),
        selected_actual_verdict=value['actual_verdict'], selected_report=str(value['report']),
        user_authority=replacement.AUTHORITY, original_record_preserved=True,
        selection_is_external_camera_reason_NOT_success_or_metric=True,
        independent_condition_increment=0, full_original_plan_hash_identical=True)
    result.update(technical_replacements=[record], excluded_technical_attempts=[record],
        completion_denominator_includes_all_declared_algorithmic_failures=True,
        exactly_one_external_camera_interruption_replaced_under_explicit_authority=True,
        selected_conditions_remain_main72_plus_history6=True)
    return result


def export():
    block = experiment.campaign()
    selected = snapshot()
    export_results = runtime.bind(block.export_results, results_snapshot=lambda:copy.deepcopy(selected), OUTPUT=OUTPUT)
    analysis = export_results()
    view = SimpleNamespace(**dict(vars(block), results_snapshot=lambda:copy.deepcopy(selected),
        verify_sources=replacement.verify))
    evidence = detailed.export(view, output_root=OUTPUT)
    runtime.evidence.write_exclusive(OUTPUT / 'selection_manifest.json', dict(
        status='ALL78_SELECTED_CONDITIONS_WITH_ONE_APPROVED_EXTERNAL_TECHNICAL_REPLACEMENT',
        original_protocol=str(block.PROTOCOL), original_protocol_sha256=runtime.sha(block.PROTOCOL),
        selected_condition_count=78, additional_independent_repetitions=0,
        replacements=selected['technical_replacements'],
        original_all78_outcomes_and_all_attempts_preserved=True))
    target = OUTPUT / 'analysis_completed.json'
    runtime.evidence.write_exclusive(target, dict(
        status='FINAL_N3_WITH_APPROVED_TECHNICAL_REPLACEMENT_ANALYZED',
        analysis_folder=str(analysis), detailed_feedback_analysis=str(evidence),
        selection_manifest=str(OUTPUT / 'selection_manifest.json'),
        selected_condition_count=78, excluded_external_technical_attempt_count=1,
        current_native_owner=replacement.KEY, final_independent_live_GET_still_required=True))
    return target


def watch():
    while not (OUTPUT / 'completed.json').exists():
        if (OUTPUT / 'stopped.json').exists():
            raise ValueError('Replacement unresolved; no final selected dataset')
        if (replacement.primary.OUTPUT / 'stopped.json').exists():
            raise ValueError('Primary queue unresolved; no selected dataset')
        time.sleep(15.)
    print(export(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export', 'watch'))
    args = parser.parse_args()
    result = export() if args.action == 'export' else watch()
    if result is not None:
        print(result, flush=True)

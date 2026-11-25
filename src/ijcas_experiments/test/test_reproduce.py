import math
import pytest
from ijcas_experiments.reproduce import describe, same_number


def test_sample_sd_and_missing_statistics():
    assert describe([])==dict(n=0,mean=None,sample_SD=None,ddof=1)
    assert describe([2.])==dict(n=1,mean=2.,sample_SD=None,ddof=1)
    assert describe([1.,3.])['sample_SD']==pytest.approx(math.sqrt(2.))


def test_different_or_missing_values_never_silently_replace_archived_values():
    same_number(1.,1.+1e-13,'same')
    for a,b in ((None,1.),(1.,None),(1.,2.)):
        with pytest.raises(ValueError):same_number(a,b,'changed')


@pytest.mark.parametrize('task', ['H5', 'H6'])
def test_sensor_raw_receipts_and_trace_reproduce_original_figure_data(task):
    from ijcas_experiments.reproduce import raw_sensor_results, sensor_results
    raw, saved = raw_sensor_results(task), sensor_results()[task]
    assert raw['all_physical_and_ROS_completion_checks_passed'] is True
    assert raw['new_trials'] == 0 and raw['paper_results_replaced'] is False
    for kind in ('localization', 'actual_tracking'):
        assert raw[kind]['samples'] == saved[kind+'_samples']
        same_number(raw[kind]['duration_s'], saved[kind+'_duration_s'], task+'/duration')
        for key in ('position_RMS_mm', 'yaw_RMS_deg'):
            same_number(raw[kind][key], saved[kind+'_'+key], task+'/'+kind+'/'+key)


def test_localization_scoring_rejects_a_nearby_nonmatching_timestamp(monkeypatch):
    from ijcas_experiments import reproduce
    reader = reproduce.csv.DictReader
    def changed_stamp(stream, *args, **kwargs):
        rows = list(reader(stream, *args, **kwargs))
        rows[0]['actual_timeline_s'] = str(float(rows[0]['actual_timeline_s']) + 1e-7)
        return rows
    monkeypatch.setattr(reproduce.csv, 'DictReader', changed_stamp)
    with pytest.raises(ValueError, match='Exact same-time'):
        reproduce.raw_sensor_results('H5')

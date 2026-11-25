"""Pure reporting tests: no SDK, experiment files or simulator command."""
import csv
import io

import pytest

import hospital_revision_selected_results as selected


def test_replacement_metadata_has_uniform_csv_columns():
    rows = [dict(cell_id='H1_B0', outcome_path='original-first', completed=False),
        dict(cell_id=selected.replacement.CELL, outcome_path='old-camera', completed=False),
        dict(cell_id='H6_B2', outcome_path='original-last', completed=False)]
    selected.label_replacement(rows, 'replacement', 'preserved-camera')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    assert len(list(csv.DictReader(io.StringIO(stream.getvalue())))) == 3
    assert rows[0]['outcome_path'] == 'original-first'
    assert rows[2]['outcome_path'] == 'original-last'
    assert [r['approved_technical_replacement'] for r in rows] == [False, True, False]


@pytest.mark.parametrize('completed', [False, True])
def test_replacement_label_does_not_depend_on_or_change_performance(completed):
    row = dict(cell_id=selected.replacement.CELL, outcome_path='old-camera',
        completed=completed, tracking_position_RMS_mm=27.4)
    selected.label_replacement([row], 'replacement', 'preserved-camera')
    assert row['completed'] is completed
    assert row['tracking_position_RMS_mm'] == 27.4
    assert row['approved_technical_replacement'] is True
    assert row['original_technical_outcome'] == 'preserved-camera'

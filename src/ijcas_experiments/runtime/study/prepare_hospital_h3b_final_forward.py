#!/usr/bin/env python3
"""Freeze a dedicated offline adapter contract. Never imports or commands Isaac."""

import json

import hospital_h3b_final_forward_pilot as h


def main():
    info = h.load_profile()
    record = h.admission_payload(info)
    h.verify_files(record['input_sha256'])
    if h.ADMISSION.exists():
        if json.loads(h.ADMISSION.read_bytes()) != record:
            raise FileExistsError('Preserve differing admission; no overwrite or bypass')
    else:
        h.ADMISSION.parent.mkdir(parents=True, exist_ok=True)
        with h.ADMISSION.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
    print('=== H3-B CAP30 FINAL-FORWARD ADAPTER PREPARED (NO MOTION) ===')
    print('actual SECOND reverse raw PASS and latest PAUSED continuity verified; all 12 joints retained')
    print('all EIGHT selected MODEL design cases verified; reused cases, NOT independent validation')
    print('fixed reference length mm:', round(info['reference_length_m']*1000,3))
    print('EXACT original goal:', info['mission_goal_xyyaw'])
    print('planned cap30mm/s; same 8deg/s, stop5mm/s, terminal30mm/2deg, strict clearance10mm')
    print('saved:', h.ADMISSION)
    print('Keep PAUSED. Dedicated instrumented FINAL-forward entry rechecks live continuity BEFORE motion.')
    print('No reset/reopen/manual Play/save/CICS or automatic retry. No whole-H3 success implied.')


if __name__ == '__main__':
    main()

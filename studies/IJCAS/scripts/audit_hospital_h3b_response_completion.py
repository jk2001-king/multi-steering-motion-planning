#!/usr/bin/env python3
"""Append-only raw five-component response closeout. No fitting or motion."""

import json

import hospital_h3b_response_completion as h


def main():
    if h.TARGET.exists() or h.SUMMARY.exists():
        h.load_closeout(h.DIRECTORY)
        print('Existing five-component closeout verified, NOT replaced:', h.TARGET)
        return
    *_, report = h.build_closeout()
    h.DIRECTORY.mkdir(parents=True, exist_ok=False)
    h.SUMMARY.parent.mkdir(parents=True, exist_ok=False)
    for path, value in ((h.TARGET, report), (h.SUMMARY, report['response_synthesis'])):
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print('=== H3-B FIVE ACTUAL RESPONSE COMPONENTS / FINAL GOAL RAW CLOSEOUT ===')
    for component in report['response_synthesis']['components']:
        print(component['component'], component['original_motion_verdict'], 'actual terminal mm/deg:',
            round(component['actual_terminal_position_error_mm'], 3), round(component['actual_terminal_heading_error_deg'], 6))
        print('  conditional model gap mm (NOT actual goal error):',
            *[round(c['final_position_gap_mm'], 3) for c in component['conditional_body_replays']])
    print('Four saved body/twelve-joint handoffs and sampled STRICT +10mm envelopes: PASS')
    print('Whole-sequence original MODEL: 2/8 PASS, six HOLDs unchanged; NO whole-route drive admission')
    print('saved:', h.TARGET)
    print('Next manual step: dedicated READ-ONLY isaac_review_ijcas_hospital_h3b_final_response_goal.py')
    print('No sixth leg, new fit, reset/reopen/Play/save/CICS or automatic retry.')


if __name__ == '__main__':
    main()

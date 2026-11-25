#!/usr/bin/env python3
"""Freeze a CONDITIONAL single-route protocol, never substitute for actual static."""

import json
import hospital_h4_path_launch as launch


def main():
    if launch.PROTOCOL.exists():
        launch.load_protocol()
        print('Existing H4 path protocol verified, not overwritten:', launch.PROTOCOL)
        return
    report = launch.prepare_protocol()
    launch.PROTOCOL.parent.mkdir(parents=True, exist_ok=False)
    with launch.PROTOCOL.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 CONDITIONAL SINGLE-ROUTE PILOT PROTOCOL (OFFLINE ONLY) ===')
    print('methods:', report['frozen_native_methods'])
    print('actual H4 static/raw CSV/live twelve-joint continuity/user authorization: REQUIRED')
    print('same v3/50mm_s/8deg_s/30mm/2deg/180s/900s; no CICS or new tuning')
    print('saved:', launch.PROTOCOL)
    print('No actual H4 static or drive admission, scene change or automatic retry.')


if __name__ == '__main__':
    main()

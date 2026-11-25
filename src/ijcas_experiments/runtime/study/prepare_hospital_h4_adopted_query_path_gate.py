"""Conditional route protocol; adoption is NOT physical route authority."""

import json

import hospital_h4_adopted_query_path_launch as launch


def main():
    report = launch.prepare_protocol()
    if launch.PROTOCOL.exists():
        if json.loads(launch.PROTOCOL.read_bytes()) != report:
            raise ValueError('Frozen adopted-query path protocol differs; no overwrite')
    else:
        launch.PROTOCOL.parent.mkdir(parents=True, exist_ok=False)
        launch.preparation.save_exclusive(launch.PROTOCOL, report)
    print('=== H4 ADOPTED QUERY / ORIGINAL V3 CONDITIONAL PATH PROTOCOL ===')
    print('Original raw static FAIL retained; separate adopted support reassessment verified')
    print('Same four plans/control/gains policy/strict map/contact guards/180s sim/900s wall')
    print('ONE selected route still requires explicit user physical-trial authorization')
    print('saved:', launch.PROTOCOL)
    print('No Play/reset/retest/drive or new final comparison result.')


if __name__ == '__main__':
    main()

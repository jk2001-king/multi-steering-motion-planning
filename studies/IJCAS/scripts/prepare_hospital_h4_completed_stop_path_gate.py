"""Freeze new preparation/static identity ONLY; no actual drive admission."""

import json

import hospital_h4_completed_stop_path_launch as launch


def main():
    report = launch.prepare_protocol()
    if launch.PROTOCOL.exists():
        if json.loads(launch.PROTOCOL.read_bytes()) != report:
            raise ValueError('Existing frozen route protocol differs; no overwrite')
    else:
        launch.PROTOCOL.parent.mkdir(parents=True, exist_ok=False)
        launch.preparation.save_exclusive(launch.PROTOCOL, report)
    print('=== H4 ORIGINAL V3 ROUTE / NEW STATIC IDENTITY BINDING (OFFLINE ONLY) ===')
    print('Unchanged four frozen plans, common controller and acceptance')
    print('Actual NEW static JSON/CSV replay + retained body/twelve joints + user launch: REQUIRED')
    print('Missing static cannot be replaced by MODEL 8/8 or fixture/test success')
    print('saved:', launch.PROTOCOL)
    print('No actual path/PhysX run, automatic continuation/retry or final comparison admission')


if __name__ == '__main__':
    main()

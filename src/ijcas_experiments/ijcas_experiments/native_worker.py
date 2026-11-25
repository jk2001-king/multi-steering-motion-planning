"""Run the retained compiled planner in SDK Python, without SimulationApp."""
from pathlib import Path
import json
import sys

from sdk_runner import configure_paths


def main():
    configure_paths()
    from ijcas_experiments.workflow import plan_selected
    if len(sys.argv) != 3:
        raise ValueError('Exactly selection.json and a reserved output required')
    path, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if path != output / 'selection.json':
        raise ValueError('Own reserved selection required')
    spec = json.loads(path.read_text())
    if spec['stage'] not in ('plan', 'run'):
        raise ValueError('A supported explicit planning selection is required')
    if spec['feedback'] == 'amcl_encoder':
        from ijcas_experiments.sensor_runner import audit_frozen
        result = audit_frozen(spec, output)
    else:
        result = plan_selected(spec, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

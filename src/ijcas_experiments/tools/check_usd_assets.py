"""Inspect local USD references without composing remote scenes or starting Kit."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
from pathlib import Path
import json
from pxr import Sdf

ROOT = _ijcas_workspace()
pending = list((_ijcas_path(ROOT, 'isaac')).rglob('*.usd')) + list((_ijcas_path(ROOT, 'isaac/scenes')).glob('*.usda'))
pending += list((_ijcas_path(ROOT, 'src')).rglob('*.usd'))
seen, remote, missing = set(), set(), []
while pending:
    path = pending.pop().resolve()
    if path in seen:
        continue
    seen.add(path)
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        missing.append(str(path))
        continue
    for asset in layer.GetExternalReferences():
        if not asset:
            continue
        if '://' in asset:
            remote.add(asset)
            continue
        target = (path.parent/asset).resolve()
        if not target.exists():
            missing.append([str(path.relative_to(ROOT)),asset])
        elif target.suffix in {'.usd','.usda','.usdc'}:
            pending.append(target)
result = dict(local_layers_checked=len(seen), missing_local_references=missing,
              external_remote_references=sorted(remote), simulator_started=False)
print(json.dumps(result,indent=2))
output=ROOT/'log/ijcas/usd_dependencies.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+'\n')
if missing:
    raise SystemExit(1)

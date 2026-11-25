"""Inspect local USD references without composing remote scenes or starting Kit."""
from pathlib import Path
import json
from pxr import Sdf

ROOT = Path(__file__).resolve().parents[1]
pending = list((ROOT/'isaac').rglob('*.usd')) + list((ROOT/'isaac/scenes').glob('*.usda'))
pending += list((ROOT/'src').rglob('*.usd'))
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
(ROOT/'migration/usd_dependencies.json').write_text(json.dumps(result,indent=2)+'\n')
if missing:
    raise SystemExit(1)

"""Exact Root-layer field diagnosis only; never waive a differing field.

Current layers are copied via native TransferContent. Only historical text is
parsed into a new anonymous layer of its own format. No live ImportFromString,
camera normalization, subtree exclusion or numerical tolerance is applied.
"""

import difflib
import math


def semantic(value):
    if isinstance(value,float) and not math.isfinite(value):
        return dict(type='float',value=str(value))
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    if isinstance(value,dict):
        return {str(k):semantic(value[k]) for k in sorted(value,key=str)}
    if hasattr(value,'GetReal') and hasattr(value,'GetImaginary'):
        return dict(type=type(value).__name__,real=semantic(value.GetReal()),imaginary=semantic(value.GetImaginary()))
    try:
        rows=list(value)
    except TypeError:
        return dict(type=type(value).__name__,value=str(value))
    return dict(type=type(value).__name__,values=[semantic(v) for v in rows])


def snapshot(layer):
    from pxr import Sdf
    paths=set()
    layer.Traverse(Sdf.Path.absoluteRootPath,lambda path:paths.add(str(path)))
    paths.add('/')
    rows={}
    for path in sorted(paths):
        spec=layer.GetObjectAtPath(path)
        if spec is None:
            if Sdf.Path(path).IsTargetPath():
                continue
            raise ValueError('Unexpected missing layer spec: '+path)
        rows[path]=dict(spec_type=type(spec).__name__,
            info={str(k):semantic(spec.GetInfo(k)) for k in sorted(spec.ListInfoKeys())})
    return rows


def copy_layer(layer):
    from pxr import Sdf
    copied=Sdf.Layer.CreateAnonymous('h4_boundary_readonly_copy.usda')
    copied.TransferContent(layer)
    return copied


def read_historical(text):
    from pxr import Sdf
    if text.lstrip().startswith('#usda'):
        suffix='.usda'
    elif text.lstrip().startswith('#sdf'):
        suffix='.sdf'
    else:
        raise ValueError('Historical Root text format is not recognized; no parser/header substitution')
    layer=Sdf.Layer.CreateAnonymous('h4_historical_root'+suffix)
    if not layer.ImportFromString(text):
        raise ValueError('Historical Root cannot be parsed in an isolated copy')
    return layer


def compare(historical_text,current_layer):
    before=snapshot(read_historical(historical_text))
    current=snapshot(copy_layer(current_layer))
    absent={'field_is_not_authored':True}
    changes=[]
    for path in sorted(set(before)|set(current)):
        a,b=before.get(path,{}),current.get(path,{})
        for field in ['__spec_type__',*sorted(set(a.get('info',{}))|set(b.get('info',{})))]:
            old=a.get('spec_type',absent) if field=='__spec_type__' else a.get('info',{}).get(field,absent)
            new=b.get('spec_type',absent) if field=='__spec_type__' else b.get('info',{}).get(field,absent)
            if old!=new:
                changes.append(dict(path=path,field=field,before=old,current=new,automatically_waived=False))
    current_text=current_layer.ExportToString()
    diff='\n'.join(difflib.unified_diff(historical_text.splitlines(),current_text.splitlines(),
        fromfile='actual_completed_Stop_archived_Root',tofile='actual_current_STOPPED_Root',n=3))
    return dict(raw_export_text_equal=historical_text==current_text,
        complete_native_Root_field_values_equal=not changes,changed_field_count=len(changes),changed_fields=changes,
        unified_Root_text_diff=diff,all_fields_compared_no_camera_or_physics_exclusions=True,
        any_field_change_automatically_waived_or_repaired=False,
        comparison_alone_does_NOT_authorize_static_Play_or_drive=True)

"""Pure runtime export checks. No SDK, state mutation or contact exemptions."""

import ast
from pathlib import Path


def audit_exports(namespace,entries):
    report={}
    for entry in entries:
        tree=ast.parse(Path(entry).read_text())
        def key(node):
            return node.slice.value if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                and node.value.id in ('h','helper','rules') and isinstance(node.slice,ast.Constant)
                and isinstance(node.slice.value,str)) else None
        required={key(n) for n in ast.walk(tree)}-{None}
        called={key(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}-{None}
        missing=required-set(namespace)
        not_callable={k for k in called if k in namespace and not callable(namespace[k])}
        if missing or not_callable:
            raise RuntimeError('Runtime interface rejected BEFORE any motion: '+Path(entry).name+
                '; missing='+str(sorted(missing))+'; noncallable='+str(sorted(not_callable)))
        report[Path(entry).name]=dict(required_keys=sorted(required),callable_keys=sorted(called))
    return report


def probe_contact_classifier(classify,root,wheels,support,legacy):
    if not callable(classify):raise RuntimeError('Contact classifier is NOT callable; no Play permitted')
    pairs=[(root+'/'+name+'/collision',support) for name in wheels]
    expected=['wheel_support']*len(wheels)
    pairs.extend([(root+'/base_link/collision',support),(root+'/'+wheels[0]+'/collision','/World/Hospital/test_wall'),
        (root+'/'+wheels[0]+'/collision','/World/unclassified_test')])
    expected.extend(['nonwheel_support','hospital_obstacle','unclassified_external'])
    if legacy:
        pairs.append((root+'/'+wheels[0]+'/collision',legacy[0]));expected.append('unexpected_legacy_support')
    result=[classify(pair,support,legacy) for pair in pairs]
    if result!=expected or [classify(tuple(reversed(pair)),support,legacy) for pair in pairs]!=expected:
        raise RuntimeError('Original wheel/forbidden-contact classification failed preflight; no exemptions or Play')
    return dict(pure_classifier_probes=result,original_forbidden_classes_retained=True,
        no_synthetic_event_injected_into_live_callbacks_or_counters=True)

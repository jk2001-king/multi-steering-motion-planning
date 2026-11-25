"""OFFLINE envelope ablation; never admit or execute a point-only path.

Replace only rectangular body half-extents by zero in a private exact SAT
kernel, retaining the same60mm per-side planning padding. Independently audit
the resulting path using the real full rectangle and unchanged propagation.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import hospital_revision_planning as planning
import hospital_native_envelope as native

runtime=planning.runtime
OUTPUT=planning.campaign.OUTPUT/'point_reference_planning'


def point_checker(base):
    source=native.SOURCE.read_text()
    old='const double a = 1.630 / 2.0 + clearance, b = .965 / 2.0 + clearance;'
    new='const double a = clearance, b = clearance;'
    if source.count(old)!=1:
        raise ValueError('Exact original body-dimension anchor required')
    derived=source.replace(old,new)
    if derived.replace(new,old)!=source:
        raise ValueError('Only body extents may differ in this ablation')
    native.initialize_native()
    directory=tempfile.TemporaryDirectory(prefix='ijcas_point_ablation_')
    target=Path(directory.name)/'point_sat.so'
    built=subprocess.run(['cc',*native.FLAGS,'-x','c','-','-o',str(target),'-lm'],
        input=derived,text=True,capture_output=True,check=False,timeout=60)
    if built.returncode:
        directory.cleanup()
        raise RuntimeError(built.stderr)
    library=ctypes.CDLL(str(target))
    function=library.hospital_sat_batch
    function.argtypes=list(native._NATIVE[2].argtypes)
    function.restype=ctypes.c_int
    class PointChecker(base):
        _private_point_library=(directory,library,function)

        def _batch(self,poses,reference):
            self.check_deadline()
            values=np.asarray(poses if isinstance(poses,np.ndarray) else list(poses),dtype=np.float64)
            if not values.size:return False
            if values.ndim!=2 or values.shape[1]!=3:
                raise ValueError('Finite pose triples required')
            values=np.ascontiguousarray(values)
            counts=np.zeros(3,dtype=np.uint64)
            value=function(self._occupied_pointer,self._integral_pointer,*self.occupancy.shape,
                *self.origin,self.resolution,*reference,self.clearance,
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(values),
                counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
            for key,count in zip(('pose_queries','empty_aabb','exact_sat'),counts):
                self.statistics[key]+=int(count)
            self.statistics['native_batches']+=1
            if value<0:
                raise ValueError('Nonfinite point-reference pose')
            return bool(value)
    PointChecker.ablation_definition=dict(
        body_half_extents_m=[0.,0.],padding_rule='same clearance on each side',
        original_C_sha256=hashlib.sha256(source.encode()).hexdigest(),
        derived_C_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        only_two_body_dimensions_removed=True,cell_SAT_and_boundary_contact_rule_unchanged=True,
        known_colliding_paths_are_NOT_executed=True)
    return PointChecker


def run_cell(cell_id):
    protocol=runtime.read(planning.campaign.PROTOCOL)
    rows=[c for c in protocol['main_cells'] if c['cell_id']==cell_id]
    if len(rows)!=1 or rows[0]['method']!=planning.campaign.planning.P or rows[0]['task'] not in ('H2','H3'):
        raise ValueError('Predeclared H2/H3 P envelope-ablation condition required')
    parent=OUTPUT/'variant_definition.json'
    if not parent.exists():
        runtime.evidence.write_exclusive(parent,dict(
            kind='OFFLINE_REFERENCE_POINT_WITH_SAME60MM_PADDING_ONLY',
            primary_protocol=str(planning.campaign.PROTOCOL),
            primary_protocol_sha256=runtime.sha(planning.campaign.PROTOCOL),
            planner_body_dimensions_m=[0.,0.],independent_audit_body_dimensions_m=[1.630,.965],
            planning_padding_m=.060,independent_planning_audit_padding_m=.060,
            source=str(Path(__file__).resolve()),source_sha256=runtime.sha(Path(__file__)),
            actual_scene_or_physical_experiments=0,unsafe_paths_NOT_executed=True))
        runtime.lc.freeze_workflow_sources(OUTPUT/'variant_sources.zip',
            {str(Path(__file__).resolve()):runtime.sha(Path(__file__))})
    def bindings(*args,**kwargs):
        method,auditor,connections,parameters,activity=planning.planner_bindings(*args,**kwargs)
        original=method.__globals__['HospitalEnvelopeChecker']
        checker=point_checker(original)
        method=runtime.bind(method,HospitalEnvelopeChecker=checker)
        parameters=dict(parameters,collision_planning_ablation=checker.ablation_definition,
                        full_envelope_not_checked_during_this_candidate_search=True)
        return method,auditor,connections,parameters,activity
    run=runtime.bind(planning.run_cell,OUTPUT=OUTPUT,planner_bindings=bindings)
    result=run(cell_id)
    audit_path=OUTPUT/cell_id/'independent_plan_audit.json'
    audit=runtime.read(audit_path) if audit_path.exists() else None
    result=dict(result,variant='P_reference_point_with_same_planning_padding',
        point_plan_found=result['plan_success'],
        full_envelope_independent_audit=audit['verdict'] if audit else 'NOT_RUN_NO_PLAN',
        full_envelope_colliding_samples=audit.get('colliding_samples') if audit else None,
        new_PhysX_trials=0,actual_motion_authorized=False)
    runtime.evidence.write_exclusive(OUTPUT/cell_id/'envelope_ablation_result.json',result)
    print('[ENVELOPE ABLATION]',result,flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell_id')
    run_cell(parser.parse_args().cell_id)

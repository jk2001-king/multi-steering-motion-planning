"""Native batching of the UNCHANGED finite-cell full-envelope SAT equations.

Build an isolated temporary library; no package installation or persisted USD
change. Independent audits still use the original numpy polygon/cell SAT.
The private function clone changes only the checker class, never global modules.
"""

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import types

import numpy as np

from hospital_planning_tools import HospitalEnvelopeChecker


SOURCE=Path(__file__).resolve().with_name("hospital_exact_sat.c")
FLAGS=("-O3","-std=c99","-shared","-fPIC","-ffp-contract=off")
_NATIVE=None


def initialize_native():
    global _NATIVE
    if _NATIVE is None:
        directory=tempfile.TemporaryDirectory(prefix="ijcas_exact_sat_")
        target=Path(directory.name)/"hospital_exact_sat.so"
        try:
            built=subprocess.run(["cc",*FLAGS,str(SOURCE),"-o",str(target),"-lm"],
                capture_output=True,text=True,check=False,timeout=60)
            if built.returncode:
                raise RuntimeError("Native SAT compilation failed: "+built.stderr)
            lib=ctypes.CDLL(str(target))
            function=lib.hospital_sat_batch
            function.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_int64),ctypes.c_int,ctypes.c_int,
                *([ctypes.c_double]*7),ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
            function.restype=ctypes.c_int
            version=subprocess.run(["cc","--version"],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()[0]
            _NATIVE=(directory,lib,function,{"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "compiler":version,"compile_flags":list(FLAGS),"fast_math_used":False,
                "temporary_library_only":True,"model_or_clearance_changed":False})
        except Exception:
            directory.cleanup()
            raise
    return dict(_NATIVE[3])


class NativeHospitalEnvelopeChecker(HospitalEnvelopeChecker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        initialize_native()
        self.occupancy=np.ascontiguousarray(self.occupancy,dtype=bool)
        self.integral=np.ascontiguousarray(self.integral,dtype=np.int64)
        self._occupied_pointer=self.occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._integral_pointer=self.integral.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        self._counts=np.zeros(3,dtype=np.uint64)
        self.statistics["native_batches"]=0

    def _batch(self,poses,reference):
        self.check_deadline()
        values=np.asarray(poses if isinstance(poses,np.ndarray) else list(poses),dtype=np.float64)
        if not values.size:return False
        if values.ndim!=2 or values.shape[1]!=3:
            raise ValueError("Expected finite pose triples")
        values=np.ascontiguousarray(values)
        result=_NATIVE[2](self._occupied_pointer,self._integral_pointer,*self.occupancy.shape,
            *self.origin,self.resolution,*reference,self.clearance,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(values),
            self._counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
        for key,value in zip(("pose_queries","empty_aabb","exact_sat"),self._counts):
            self.statistics[key]+=int(value)
        self.statistics["native_batches"]+=1
        if result<0:raise ValueError("Non-finite hospital pose")
        return bool(result)

    def __call__(self,local_poses):
        return self._batch(local_poses,self.start)

    def world_collision(self,pose):
        return self._batch([pose],(0.,0.,0.))


def cloned_plan_method():
    from plan_hospital_h1 import plan_method
    namespace=dict(plan_method.__globals__)
    namespace["HospitalEnvelopeChecker"]=NativeHospitalEnvelopeChecker
    return types.FunctionType(plan_method.__code__,namespace,plan_method.__name__,plan_method.__defaults__,plan_method.__closure__)

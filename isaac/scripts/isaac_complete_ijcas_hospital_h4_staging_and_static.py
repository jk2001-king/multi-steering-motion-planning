"""One operator launch: remaining staging, then ONE authorized static trial."""

from pathlib import Path
import runpy


if (globals().get('ijcas_hospital_h4_completed_stop_continuation_authorized') is not True
        or globals().get('ijcas_hospital_h4_static_authorized') is not True):
    raise RuntimeError('Both authorized remaining-staging and single-static flags are required')

scripts = Path('/home/jk/Research/IJCAS_ws/isaac/scripts')
runpy.run_path(str(scripts/'isaac_complete_ijcas_hospital_h4_staging_after_stop.py'),
    init_globals={'ijcas_hospital_h4_completed_stop_continuation_authorized': True}, run_name='__main__')
# Any failed staging raises above; no bypass, manual Play or second attempt.
namespace = runpy.run_path(str(scripts/'isaac_run_ijcas_hospital_h4_completed_stop_static_gate.py'),
    init_globals={'ijcas_hospital_h4_static_authorized': True}, run_name='__main__')
ijcas_hospital_h4_static_task = namespace['ijcas_hospital_h4_static_task']

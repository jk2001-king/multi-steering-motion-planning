"""Complete ONLY previously unexecuted staging; no timeline commands."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _complete_staging():
    from pathlib import Path
    import runpy
    import omni.timeline
    import omni.usd

    if globals().get('ijcas_hospital_h4_completed_stop_continuation_authorized') is not True:
        raise RuntimeError('Explicit authorized remaining H4 staging flag required')
    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    c = load(scripts/'hospital_h4_completed_stop_preparation.py')
    report = c['stage_remaining'](omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface())
    print('=== H4 COMPLETED-STOP REMAINING STAGING (NO ADDITIONAL STOP / NO PLAY) ===')
    print('exact reviewed Root fields:', 269, '; old H3 archive / FAIL / HOLD preserved')
    print('new H4 robot / fixed start:', c['ROBOT'], report['start_xyyaw'])
    readback = report['live_new_robot_readback']
    print('mass / bodies / joints:', readback['simulation_mass_kg'], len(readback['rigid_bodies']), len(readback['revolute_joint_names']))
    print('source-authored seed; original material/gains/support/colliders unchanged')
    print('saved:', c['LIVE'])
    print('STOPPED. Staging ONLY; separate single authorized static trial still required.')
    return report


ijcas_hospital_h4_completed_stop_staging = _complete_staging()

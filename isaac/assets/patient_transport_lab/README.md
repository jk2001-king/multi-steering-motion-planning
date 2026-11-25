# Measured laboratory patient-transfer platform

This directory is reserved for the USD imported from the measured laboratory
platform description. It is intentionally separate from
`isaac/assets/patient_transport/patient_transport.usd`, which represents the
older U-shaped design and hospital insertion scenario.

Source URDF:

`src/patient_transport_description/urdf/patient_transport_lab_parametric.urdf`

Canonical photograph-aligned import:

`isaac/assets/patient_transport_lab/patient_transport_lab.usd`

Import settings:

- Merge Fixed Joints: off
- Fix Base Link: off
- Create Physics Scene: on
- Imported root name: `patient_transport_lab_screening`

Before running the Timeline, execute:

`isaac/scripts/isaac_audit_patient_transport_lab_import.py`

The first import and its subsequent long-arm visual patch placed the casters
too far inboard. The model was reimported on 2026-10-06 under the same
canonical filename to update caster joint and rigid-body positions together.
The provisional caster X coordinate is aligned with the crossbeam center
(magnitude 0.660 m), pending remeasurement. Verify the current stage with the
import audit before any dynamics experiment.

The 240 kg mass distribution is provisional. Do not present it as a measured
property of the laboratory platform.

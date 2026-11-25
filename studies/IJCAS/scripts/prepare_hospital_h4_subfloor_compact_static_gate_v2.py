"""Offline exact metadata review and conditional static definition; no Kit."""

import hospital_h4_namespace_reference as reference
import hospital_h4_subfloor_compact_static_gate_v2 as gate


def main():
    p=gate.preparation
    review=reference.build_review()
    if reference.REVIEW.exists():
        if p.read_small(reference.REVIEW)!=review:
            raise ValueError('Recorded exact metadata review differs; no overwrite')
    else:p.write_exclusive(reference.REVIEW,review)
    protocol=gate.prepare_protocol()
    if gate.PROTOCOL.exists():
        if p.read_small(gate.PROTOCOL)!=protocol:
            raise ValueError('Frozen v2 conditional protocol differs; no overwrite')
    else:p.write_exclusive(gate.PROTOCOL,protocol)
    print('Exact eight unvalued Namespace metadata fields independently reproduced')
    print('Saved 145 actor nodes equal immutable source under identical offline USD registry')
    print('v2 requires ALL live actor settings to match a source reference under the SAME running SDK')
    print('Same original static controller / thresholds / 26+4 probes / full 4997 environment rows')
    print('No live stage, Play, retry, physical PASS or drive admission')
    print('saved:',gate.PROTOCOL)


if __name__=='__main__':main()

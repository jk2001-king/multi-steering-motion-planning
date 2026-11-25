#!/usr/bin/env python3
"""Freeze ONE next-forward physical-adapter admission without running Isaac."""

import json
from pathlib import Path

import hospital_h3b_next_forward as h


def main():
    info=h.load_profile();report=h.admission_payload(info)
    h.verify_files(report['input_sha256'])
    h.ADMISSION.parent.mkdir(parents=True,exist_ok=True)
    with h.ADMISSION.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print('=== H3-B ONE NEXT FORWARD EXECUTOR ADMISSION (NO MOTION) ===')
    print('raw reverse PASS / live 12-joint continuity: VERIFIED')
    print('next counterphase forward reference m:',info['reference_length_m'])
    print('frozen model screens: 4/4 PASS; inherited course law and gains unchanged')
    print('STATUS:',report['status']);print('saved:',h.ADMISSION)
    print('Keep SAME reverse-PASS scene PAUSED. Only the dedicated NEXT-forward instrumented entry may be used; no old controller/reset/manual Play.')


if __name__=='__main__':main()

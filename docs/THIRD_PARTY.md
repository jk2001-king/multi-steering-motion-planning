# Third-party components and distribution

## NVIDIA Isaac Sim / Hospital / Grid

Isaac Sim 5.0 SDK와 NVIDIA Hospital/Grid 환경 자산은 별도 설치·접근하는 외부 배포물이다. 이 저장소는 해당 SDK, Hospital/Grid geometry·texture·binary를 포함하지 않는다. `isaac/scenes/hospital_experiment.usda`에는 원본 환경의 외부 USD 참조가 남아 있다. 다운로드와 사용에는 원배포자의 해당 조건을 따른다.

## Robot description

`patient_transport_description`, `wheelchair_description`와 `isaac/assets`에는 기존 로봇 description, mesh와 USD가 있다. 기존 `package.xml`의 라이선스 표기를 유지한다. 이 업로드는 모델·디자인의 제3자 권리를 새롭게 허가하거나 환자이승 하드웨어 사용을 인증하지 않는다.

원본 환자이승 로봇 base USD는 약 77 MB다. 모델 수치를 바꾸지 않기 위해 원본을 유지하며 Hospital/Grid 전체 환경을 이 파일에 합쳐 배포하지 않는다.

## CPython numerical compatibility

`studies/IJCAS/scripts/hospital_mission_planning.py`의 `_cpython311_hypot2`는 CPython 3.11의 두 좌표 norm 계산 순서를 유지하는 수치 호환 구현을 포함한다. 원본 PSF 고지와 수정 요약은 [THIRD_PARTY_NOTICES.txt](../studies/IJCAS/THIRD_PARTY_NOTICES.txt)에 그대로 있다.

## OMPL

설치된 OMPL 1.7의 Reeds–Shepp metric을 선택적 탐색 가이드로 사용하는 어댑터가 있다. OMPL 알고리즘 소스·헤더·바이너리를 이 저장소에 포함하지 않으며 설치된 구성요소의 원래 조건을 따른다. 이 metric은 응답 모델, 전체 외곽 검사 또는 물리 안전 검증의 대체물이 아니다.

## ROS 2 / Nav2 / Python dependencies

ROS 2 Humble과 Nav2는 설치된 시스템 패키지를 사용한다. NumPy, PyYAML, Numba/llvmlite 등은 별도 의존성이다. 이 저장소에는 해당 프로젝트 배포본 전체를 vendoring하지 않는다. 각 의존성의 원래 라이선스가 적용된다.

## Project license boundary

공유 모델 패키지에는 기존 Apache-2.0 표기가 있고 `ijcas_experiments`, `ijcas_nav2_sensor_adapter`에는 `UNLICENSED` 표기가 있다. 별도 동의 없이 저장소 전체에 새로운 MIT/Apache 라이선스를 선언하지 않았다. 공개 열람 가능 여부와 모든 구성요소의 재사용 허가는 구분해야 한다.

## What is not uploaded

원본 실행 로그, raw trace, native sensor frames, calibration evidence, 로컬 이관·백업 기록, colcon 출력과 개발 보관본은 업로드하지 않는다. `docs/data`에는 평가 확인을 위한 소규모 측정 결과 요약과 원본 상대경로·SHA-256 provenance만 제공한다. 원본 자료가 필요한 실행/분석에는 별도 원본 묶음이 필요하다.

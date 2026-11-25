# Experiments and evaluation

## 1. 고정 비교 블록

본 비교는 `revision_final_n3_seed30_v2_20261122`의 H1–H6 × B0/B1/B2/P × 3초기조건, 총 **72조건**이다. H2/H3 상태 이력 제거 **6조건**과 오프라인 외곽 비교 **12조건**은 별도로 집계한다.

조건은 `numpy.default_rng(2026112205)`로 생성한 차체 초기 pose와 전·후 조향륜·네 캐스터의 각도 변화를 함께 포함한다. 각 과제의 같은 조건을 모든 비교 방법에 공통 적용한다. 현재 선택기의 condition 1–3은 이 보관된 조건을 선택하며 새 난수 반복을 생성하지 않는다. 실제 정적 진입과 tracking 진입의 조인트 상태도 원본 기록으로 보존한다.

과거 개발 과정의 선택 사례, 이전 5조건 블록, 새 실행기 검증 결과를 이 고정 블록과 합쳐 성공률을 만들지 않는다. H6/B1의 외부 카메라 조작 중단은 사전에 승인한 동일조건 기술적 대체 실행만 선택했으며 원 중단 기록은 보존했다. 과학적 미완료를 성공 재시도로 바꾸지 않는다.

## 2. 공통 조건과 비교 방법

| 항목 | 최종 본 비교 |
|---|---|
| B0 | 역위상 단독 운동학 |
| B1 | F/R/C 다중 모드 운동학 |
| B2 | B1 + 조향 모드/전환 비용 |
| P | 조향·속도 응답 상태 전파 + 후보 검사/정제 |
| Cruise / acceleration | 모든 방법에 0.030 m/s / 0.020 m/s² |
| P speed levels | 0.003, 0.010, 0.030 m/s |
| P lookahead | 모든 과제에서 4 |
| H1–H2 | 180 simulation s / 900 wall s |
| H3–H4 | 300 simulation s / 900 wall s |
| H5 | 240 simulation s / 900 wall s |
| H6 | 모든 방법에 600 simulation s / 1,500 wall s |
| Actual completion | 위치 30 mm / yaw 2°, 속도·정지 유지·접촉 하중 검사 |

공통 cruise 설정은 실제 각 샘플의 속도가 완전히 같다는 의미가 아니다. 정렬, 모드 전환과 감속이 실제 속도 및 주행 시간을 바꾼다. B2–P 결과는 이 공통 속도 정책 아래의 전체 계획 구성 비교다. 개별 물리 제약이나 모든 응답 파라미터의 독립 인과 효과를 입증하는 결과로 확대하지 않는다.

실행 모델의 HOLD는 사전 예측으로 보존한다. 독립 전체 외곽 검사와 실제 정적 검사 통과 후에는 공통 실제 안전 감시 아래 실행 결과를 측정한다. 모델 HOLD를 PASS로 변경하거나 실제 안전 검사를 해제한 것이 아니다.

## 3. 평가 정의

- 위치/자세는 world/map과 정합된 `base_link` 원점 기준이다.
- Actual tracking은 simulator pose와 **실제 경로 진행에 대응한 reference**의 차이다. 시간표에 따른 가상 reference와 비교하지 않는다.
- Localization은 동시각 estimated pose와 simulator pose의 차이다. Estimated-reference 내부 오차와 actual tracking은 구분한다.
- yaw 차이는 `atan2(sin(delta), cos(delta))`로 wrap한다.
- RMS는 제곱 오차의 시간 적분을 평가 시간으로 나눈 후 제곱근을 취한다. 센서 재계산기는 인접한 두 표본이 모두 `path_tracking`일 때 trapezoidal 적분한다.
- Tracking 평가에는 해당 단계의 정렬·방향 전환이 포함되며 pre-hold, 종단 goal correction, 최종 stationary hold는 제외한다.
- 센서 localization은 timestamp가 정확히 대응된 20 Hz 제어 자료, actual tracking은 약 60 Hz physics 자료를 사용한다. 최근접 시간 대체·보간·held estimate 점수화는 하지 않는다.
- 완료 사례 RMS 통계는 **완주·정지 유지까지 통과한 사례만** 사용한다. 완료율 분모에는 전체 선언 조건을 유지한다.
- 표본 SD는 `ddof=1`이다. 완료 n=1에서는 SD를 계산하지 않으며 n=0은 미측정으로 둔다.
- 계획 시간은 원본 wall-clock 측정값이다. 새 오프라인 재계산 시간을 원본 계획 시간으로 보고하지 않는다.
- 주행 시간은 사전 hold부터 추종·대기·goal correction·최종 hold까지의 simulation time이다. 준비·정적 검사는 제외한다.

완료 기준은 Nav2 action 결과만으로 판단하지 않는다. 5초 stationary hold의 마지막 3초 전체에 대해 위치/yaw, raw PhysX 수평 속도와 약 0.2초 위치창 속도 ≤5 mm/s, 기준 위치 대비 변위 ≤3 mm, 위치 선형회귀 속도 ≤1 mm/s, 여섯 바퀴 평균 수직 접촉력 각각 >1 N, 총 평균 하중 0.8Mg–1.2Mg를 확인한다. 실제 Pause와 원래 게인/목표 복원도 별도 확인한다.

## 4. 본 비교 결과

단위: 계획 시간 s, 위치 RMS mm, yaw RMS deg. 계획 시간은 모든 선언 조건, RMS는 완료 사례의 평균 ± 표본 SD다. n=1에는 SD가 없다. 실행을 시작하지 못했거나 중단된 조건의 짧은 trace RMS를 완주 정확도와 섞지 않는다.

| Task | Method | Completed | Planning time (s) | Position RMS (mm) | Yaw RMS (deg) |
|---|---|---:|---:|---:|---:|
| H1 | B0 | 0/3 | 0.205 ± 0.047 | — | — |
| H1 | B1 | 2/3 | 0.191 ± 0.023 | 35.180 ± 1.741 | 2.195 ± 0.125 |
| H1 | B2 | 3/3 | 0.239 ± 0.009 | 16.692 ± 0.790 | 1.087 ± 0.113 |
| H1 | P | 3/3 | 11.153 ± 0.329 | 15.859 ± 2.181 | 0.602 ± 0.085 |
| H2 | B0 | 1/3 | 0.604 ± 0.260 | 8.509 (n=1) | 1.732 (n=1) |
| H2 | B1 | 3/3 | 0.185 ± 0.008 | 40.042 ± 6.137 | 2.908 ± 0.205 |
| H2 | B2 | 1/3 | 0.254 ± 0.021 | 15.968 (n=1) | 1.039 (n=1) |
| H2 | P | 3/3 | 11.196 ± 0.397 | 13.060 ± 1.244 | 0.557 ± 0.005 |
| H3 | B0 | 0/3 | 18.129 ± 3.929 | — | — |
| H3 | B1 | 0/3 | 180.083 ± 0.004 | — | — |
| H3 | B2 | 0/3 | 169.227 ± 1.068 | — | — |
| H3 | P | 3/3 | 70.749 ± 10.688 | 17.707 ± 2.706 | 1.351 ± 0.174 |
| H4 | B0 | 0/3 | 47.250 ± 1.362 | — | — |
| H4 | B1 | 0/3 | 5.173 ± 1.799 | — | — |
| H4 | B2 | 1/3 | 1.606 ± 0.511 | 33.161 (n=1) | 1.269 (n=1) |
| H4 | P | 2/3 | 105.760 ± 3.525 | 14.619 ± 2.533 | 1.258 ± 0.279 |
| H5 | B0 | 0/3 | 0.480 ± 0.159 | — | — |
| H5 | B1 | 1/3 | 0.815 ± 0.104 | 29.437 (n=1) | 3.147 (n=1) |
| H5 | B2 | 0/3 | 4.117 ± 2.756 | — | — |
| H5 | P | 3/3 | 53.973 ± 1.489 | 16.541 ± 0.621 | 0.934 ± 0.044 |
| H6 | B0 | 0/3 | 2.127 ± 0.562 | — | — |
| H6 | B1 | 0/3 | 1.465 ± 0.249 | — | — |
| H6 | B2 | 0/3 | 14.502 ± 4.134 | — | — |
| H6 | P | 3/3 | 155.563 ± 2.369 | 14.126 ± 1.411 | 0.952 ± 0.037 |

전체 완료 횟수는 B0 1/18, B1 6/18, B2 5/18, P 17/18이다. P의 H4 미완료 1조건을 삭제하거나 완료로 대체하지 않았다. n=3의 제한된 초기조건 민감도 평가이며 임상 환경의 모집단 성공 확률을 추정한 대규모 시험이 아니다.

공통으로 완료한 H2 condition 3의 B2/P 위치 RMS는 각각 **15.967868 / 12.188521 mm**, yaw RMS는 **1.038663 / 0.554787 deg**다. 이는 해당 paired case의 결과이며 전체 조건에 대한 동일 개선율로 일반화하지 않는다.

[summary.csv](data/summary.csv), [individual.csv](data/individual.csv)에 반올림 전 결과와 집계 표본 수가 있다. `summary.csv`는 역할 열로 본 비교와 ablation을 구분한다.

## 5. Dynamic-response history ablation

`P_no_history`는 구간 진입의 조향각·속도 상태를 정상 입력 상태로 투영하는 예측 모델이다. 실제 로봇을 구간 경계마다 초기화하지 않는다. 응답 이력 전파의 묶음 효과를 비교하며, 조향 이력과 속도 이력의 개별 기여까지 각각 분리한 실험은 아니다.

| Task / variant | Completed | Position RMS (mm) | Yaw RMS (deg) | Planning time (s) |
|---|---:|---:|---:|---:|
| H2 / P carry | 3/3 | 13.060 ± 1.244 | 0.557 ± 0.005 | 11.196 ± 0.397 |
| H2 / no history | 3/3 | 32.179 ± 2.177 | 2.869 ± 0.509 | 11.833 ± 0.063 |
| H3 / P carry | 3/3 | 17.707 ± 2.706 | 1.351 ± 0.174 | 70.749 ± 10.688 |
| H3 / no history | 0/3 | — | — | 63.156 ± 5.209 |

원본 반올림 전 값은 [history_ablation.csv](data/history_ablation.csv)와 [summary.csv](data/summary.csv)에 있다. 이 표의 계획 시간은 전체 선언 조건을 포함한다.

## 6. Full-envelope / force-load

오프라인 외곽 비교는 H2/H3의 세 초기조건에 point-only와 endpoint-only 변형을 적용한 12조건이다. 별도 독립 full-envelope validator로 전환 구간까지 다시 검사했다. H3 point-only 세 경로에서 각각 961 / 469 / 1,107개의 충돌 표본이 확인되었다. 위험 경로를 실제로 주행하지 않았으며 **관측된 물리 충돌 횟수로 보고하지 않는다**. Endpoint-only H3는 원래 reference/map/resource 검사에서 거부되었고, 이를 envelope만의 단독 원인으로 분해했다고 주장하지 않는다. [offline_envelope.csv](data/offline_envelope.csv)에 상태와 범위가 있다.

본 비교의 선택 P 경로 최대 friction utilization은 0.019024, 최대 절댓값 LTR은 0.142658이며 설정 한도는 각각 0.8이다. 후보 활동 기록에서도 해당 한도 초과 횟수는 0이다. 따라서 이번 저속 조건에서 **force/load 한도가 경로 선택이나 성공률 개선을 직접 유발했다고 주장하지 않는다**. 근사 지표 계산·검사 기능은 구현 범위로 유지하되 실제 안전성 보증이나 검증된 기여로 확대하지 않는다. [constraints.csv](data/constraints.csv)의 각 primitive 최대값을 선택 경로 전체에서 다시 최대 집계하며 실제 종단 보정/정지 구간은 제외한다.

## 7. 별도 센서 기능 검증과 새 ROS 실행기 확인

H5/H6의 기존 고정 P 경로를 AMCL/엔코더 피드백으로 실행한 각 한 사례다. 본 비교 세 초기조건의 추가 반복으로 세지 않는다. H6 센서 한도는 600 simulation s / 3,000 wall s로 본 비교와 별도의 프로토콜이다.

원본 native scan configuration, known-start AMCL prior와 full GUI fixed timeline 설정을 새 실행기에도 유지했다. 이전 어댑터 불일치 시도는 로컬에 보존하고 최종 비교 통계에 포함하지 않았다. 최종 새 H5/H6 실행은 실제 센서·PhysX 실행이며 보관된 제어/pose 로그를 재생하여 만든 성공이 아니다.

| Task | Fresh run ID | Physics rows / control records |
|---|---|---:|
| H5 | `20261125_123612_853224_H5_sensor_cohort_04_r01` | 8,149 / 2,717 |
| H6 | `20261125_125122_775531_H6_sensor_cohort_03_r01` | 24,577 / 8,193 |

이 호스트에서는 비교 대상 physics 수치 필드, 제어 timestamp/명령, estimated pose와 보고 지표가 원본과 일치했다. 다른 GPU/PhysX 환경, 센서 잡음, 초기화 변경 또는 wall time까지 같다고 보장하지 않는다. [sensor_raw_results.json](data/sensor_raw_results.json)에 RMS·최댓값·표본 수·구간과 원본 입력 hash, [commissioning.json](data/commissioning.json)에 검증 실행의 식별 정보가 있다. 새 ROS 실행기의 fresh H2/P/condition 3 대표 사례도 기록되어 있으나 전체 72조건을 이번에 다시 주행한 것은 아니다.

## 8. 재계산과 자료 범위

```bash
source install/setup.bash
ros2 run ijcas_experiments reproduce_results
```

원본 묶음이 필요하며 새 시뮬레이션은 시작하지 않는다. 출력의 `individual.csv`, `summary.csv`, `history_ablation.csv`, `constraints.csv`, `offline_envelope.csv`, `sensor_raw_results.json`을 확인한다.

공개 `docs/data`는 보관된 2026-11-25 재계산 출력에서 집계 표를 복사하거나 상태·수치 열을 선택한 소규모 문서용 자료다. raw trace, 원본 오류의 로컬 절대경로, 민감한 호스트 설정은 포함하지 않는다. 원본 export의 상대경로·SHA-256은 [provenance.json](data/provenance.json)에 있다. 요약 표만으로 원본 raw 계산을 재검증할 수 있다는 주장은 하지 않는다.

## 9. 남은 한계

- 초기조건 세 가지와 한 병원 환경의 저속 비교이며 P도 18조건 중 1조건을 완료하지 못했다.
- P의 계획 시간은 baseline보다 길다. 정확도·완료율과 계산 비용을 함께 평가해야 한다.
- 동일 cruise 정책과 history ablation만으로 모든 파라미터·안전 제약의 독립 효과가 분리되지는 않는다.
- 근사 force/load 지표는 실제 접촉력·전도 안전 보증이 아니며 이 실험에서는 한도가 활성화되지 않았다.
- 전체 외곽 검사는 유한 표본의 평면 geometry 검사로 연속시간 3D clearance 보증이 아니다.
- 센서 사례는 noise-free 가상 스캔과 작은 known-start prior를 사용한다. 잡음·전역 초기화·실물 검증은 별개다.

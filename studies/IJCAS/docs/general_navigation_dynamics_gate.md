# IJCAS 일반 자율주행 동역학 경로계획 게이트

## 3-LiDAR 인터페이스 및 GPU 운용 판정 (2026-10-23)

HY 장착 위치에 둔 임시 RTX LiDAR는 센서별 단독 시험에서 모두 정상적인 ROS 2
`PointCloud2`와 올바른 frame ID를 발행했다. 사용한 `Example_Rotary`는 정확한
MID-360 광학 모델이 아니므로 이 결과는 관측 인터페이스 시험으로만 해석한다.

| 구성 | 관측 발행률 | frame | 판정 |
|---|---:|---|---|
| 전방 단독 | 약 2.88--3.12 Hz | `livox_front` | PASS |
| 후방 단독 | 약 2.86 Hz | `livox_rear` | PASS |
| 상단 단독 | 약 3.60 Hz | `livox_top` | PASS |

3대 동시 구성은 세 토픽이 일시적으로 약 3.1 Hz를 발행했지만 곧 심각한 렌더링
지연이 발생했고 NVIDIA 드라이버가 다시 응답하지 않았다. 앞선 시험에서도
Xid 31, CUDA error 700 및 PhysX GPU broadphase 오류가 확인되었으므로 **지속
운용 FAIL**로 판정한다. 토픽이 잠시 발행된 사실을 동시 운용 PASS로 보고하지
않는다.

이후 상단 RTX LiDAR 단독 구성도 10,563-point XYZ payload와 장착 TF 검증에는
통과했으나, 지속 운용 중 동일하게 GPU 드라이버가 응답하지 않았다. 따라서 단독
RTX 역시 장시간 mapping 입력으로 사용하지 않는다. USD를 다시 여는 것만으로
RTX/Replicator/CUDA 자원 수명이 초기화되지 않는 문제도 있으므로 RTX 센서 시험
사이에는 Isaac 프로세스를 완전히 종료해야 한다.

따라서 RTX 3070 기반 시뮬레이션의 G2 센서 폐루프는 다음과 같이 제한한다.

- 전방/후방/상단 센서의 ROS 인터페이스와 장착 TF는 각각 단독으로 검증한다.
- 지도 생성, 위치추정 및 Nav2 폐루프는 GPU 렌더링과 무관한 CPU PhysX 360°
  raycast `/scan`으로 수행한다. 이는 장착 센서의 정확한 광학 재현이 아니라
  장애물 관측 기반 폐루프 검증을 위한 대체 센서 모델이다.
- 실제 플랫폼의 3대 MID-360 동시 융합은 물리 플랫폼 단계에서 별도로 검증한다.
- 본 제한은 GPU/Isaac RTX 자원 한계이며 제안 경로계획기의 성능 이점으로
  해석하지 않는다.

재발 방지를 위해 준비 스크립트의 `all` 선택은 기본적으로 차단한다.

## 확정한 연구 범위

침상 정렬·인서트·이탈은 도킹 절차이므로 본 IJCAS의 중심 과제에서 제외한다.
기존 침상 측면 정렬 결과는 조향 구성 전환의 성립성을 확인한 개발 증거로만
보존한다.

본 연구의 중심은 긴 전·후단과 전·후륜 능동조향을 가진 이동 플랫폼이 일반
자율주행 환경을 통과할 때 다음 항목을 경로계획 상태와 비용에 포함하는 것이다.

- 현재 전륜 및 후륜 조향각
- 조향 속도 제한과 구성 전환 시간
- 조향 구성별 실제 곡률 응답
- 차체와 캐스터를 포함한 전체 외곽선
- 경로 길이, 후진, 구성 전환 및 수행시간 비용

환자이승로봇은 검증 플랫폼이지만, 문제 정의는 침상 도킹이나 KTL 시나리오에
종속되지 않는다.

## 비교군

1. `fixed_counterphase`: 기존 전·후륜 역상조향만 사용하며 동일한 동역학을 적용
2. `multimode_kinematic`: 세 조향 구성을 허용하되 조향 전환을 순간적으로 가정
3. `multimode_transition_cost`: 전환 시간은 비용으로만 반영하고 궤적은 이상적으로 전파
4. `proposed_full_dynamics`: 조향 상태·속도 제한·구성별 곡률 응답을 궤적에 직접 전파

각 계획 결과는 동일한 full-dynamics 모델로 오프라인 재생하고, 제안 경로는 별도로
Isaac Sim에서 강체 동역학으로 검증한다. 오프라인 재생은 독립된 물리 검증으로
간주하지 않는다.

## 일반 주행 시나리오

- 직선 주행 직후 90° 코너
- S자 통로
- 좁은 T자 교차로
- 문 통과 직후 회전
- 장애물 우회 후 원 경로 복귀

현재 치수는 성립성 검토용 시뮬레이션 형상이다. 최종 논문에서는 병원·시설 통로의
측정값 또는 명시한 폭 변화 범위로 고정해야 한다.

## 30,000-node 예비 결과

### 직선 후 90° 코너

| 방법 | 계획 | full-dynamics 재생 |
|---|---:|---:|
| 고정 역상조향 | 예산 내 경로 미발견 | - |
| 다중모드 운동학 | 성공, 2.55 m | 실패: 138.4 mm / -4.73° |
| 전환 비용만 반영 | 성공, 2.55 m | 충돌, 264.4 mm / -9.85° |
| 제안 full dynamics (초기 계수) | 성공, 2.55 m | 실패: 19.4 mm / 4.61° |

첫 Isaac 실행 `general90_dyn_r01`은 접촉 없이 51/51 primitive를 끝냈으나,
최종 yaw 오차 4.61°로 실패했다. CSV의 안정 구간에서 실제 yaw/curvature 비율을
재계산한 결과 계획 계수보다 전륜 단독은 약 3.4%, 후륜 단독은 약 5.5% 큰 응답을
보였고, 정지 후 안정화 구간에서 yaw가 추가로 약 0.64° 변했다. 따라서 이 실행은
검증 반복이 아니라 응답 계수 보정 자료로 분류한다. 기존 결과를 성공으로 바꾸지
않는다.

곡률 응답 계수를 적용한 Isaac 실행 `general90_dyn_r02`는 접촉 없이 51/51을
완료했고 yaw 오차는 1.05°로 통과했지만 위치 오차 40.95 mm로 실패했다. 안정 구간의
차체 횡방향 속도 비율은 내부 모델보다 전륜 단독에서 약 5.7% 작고 후륜 단독에서
약 5.1% 컸다. 이 결과도 성공으로 세지 않고 횡방향 응답 계수 보정에 사용한다.

두 응답 보정 후 `calibrated_candidate_r03`에서 새 경로를 생성했다. 90° 코너는
52 primitive, 모드 전환 1회, 후진 0회이며 내부 종단 오차 20.5 mm / -0.35°이다.
T자 교차로는 49 primitive, 모드 전환 1회, 후진 0회이며 내부 종단 오차
12.8 mm / 0.04°이다. 이 값들은 planner 내부 결과이며 실제 Isaac 재생 검증 전이다.

### 좁은 T자 교차로

| 방법 | 계획 | 내부 full-dynamics 오프라인 재생 |
|---|---:|---:|
| 고정 역상조향 | 예산 내 경로 미발견 | - |
| 다중모드 운동학 | 성공, 2.35 m | 실패: 233.1 mm / -8.41° |
| 전환 비용만 반영 | 성공, 2.35 m | 실패: 247.6 mm / -10.05° |
| 제안 full dynamics (초기 계수) | 성공, 2.45 m | 통과: 17.6 mm / -0.29° |

제안 방식은 약 0.10 m 긴 경로를 선택하지만 실제 조향 전환 후에도 실행 가능한
종단 상태를 만든다. 즉 최단 운동학 경로보다 실행 가능성을 우선한 결과이다.

보정된 제안 경로의 Isaac 명목 반복은 현재 1회 통과했다.

| 실행 | 접촉 | 완료 action | 위치 오차 | yaw 오차 | 판정 |
|---|---:|---:|---:|---:|---|
| `generalt_dyn_r01` | 0 | 49/49 | 22.04 mm | 1.068° | 통과 |
| `generalt_dyn_r02` | 0 | 49/49 | 17.57 mm | 0.918° | 통과 |
| `generalt_dyn_r03` | 0 | 49/49 | 13.35 mm | 0.524° | 통과 |

독립 반복 3회가 모두 통과했다. 따라서 두 명목 시나리오의 Isaac 동역학 반복
게이트가 완료되었다. 이는 명목 조건의 반복 재현성 근거이며, 조건 변화에 대한
강건성이나 실물 플랫폼 성능을 의미하지 않는다.

계수 보정 후보의 실제 Isaac 재생 기록은 다음과 같다.

| 실행 | 접촉 | 완료 action | 위치 오차 | yaw 오차 | 동역학 시간 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| `general90_dyn_r01` | 0 | 51/51 | 19.36 mm | 4.606° | - | 실패 (yaw) |
| `general90_dyn_r02` | 0 | 51/51 | 40.95 mm | 1.048° | - | 실패 (위치) |
| `general90_dyn_r03` | 0 | 52/52 | 26.20 mm | -0.006° | 92.43 s (1.067× nominal) | 통과 |
| `general90_dyn_r04` | 0 | 52/52 | 25.89 mm | 0.039° | - | 통과 |
| `general90_dyn_r05` | 0 | 52/52 | 26.07 mm | -0.025° | - | 통과 |

`r01`과 `r02`는 각각 곡률 응답과 횡방향 응답을 보정하는 개발 실행으로 남기며,
성공 횟수에 포함하지 않는다. `r03`은 두 계수를 반영해 다시 계획한 뒤 수행한 첫
실제 동역학 통과이다. r03–r05의 독립 명목 반복 3회가 모두 접촉 0회, 52/52 action,
위치 30 mm 이하, yaw 2° 이하로 통과했다. 이는 이 시나리오의 명목 반복 게이트를
통과한 것이며, 조건 변화에 대한 강건성을 입증한 것은 아니다. planner 요약의 일반
가상 횡방향 경계 여유는 ±3 m 가상 경계 기준이지 씬의 시나리오 벽 접촉 판정이 아니다.
실제 시나리오 충돌체는 별도 PhysX 접촉 모니터에서 0회였다.

S자 통로와 장애물 우회 형상은 100,000-node 예산에서도 모든 방법이 실패했다.
이는 제안 우월성 근거로 사용하지 않으며 형상 타당성 및 탐색 효율을 재검토한다.
문 직후 회전은 기존 목표 자세가 벽과 겹치는 형상 오류를 확인해 오른쪽 벽 시작점을
2.60 m에서 2.90 m로 수정했다. 수정 형상의 공통 100,000-node 비교에서 고정 역상조향과
제안 full dynamics는 내부 재생을 통과했고, 전환 비용만 반영한 경로는 full-dynamics
재생에 실패했으며 다중모드 운동학은 예산 내 경로를 찾지 못했다. 제안 경로는 49 action,
2.45 m, 후진 0회, 모드 전환 1회이다. 첫 Isaac 재생 `door_dyn_r01`은 접촉 0회,
49/49 action, 위치 오차 29.98 mm, yaw 오차 1.052°, 수행시간 88.64 s로 통과했다.
다만 위치 기준 30 mm에 대한 여유가 0.02 mm에 불과하므로 안정적인 통과로 해석하지
않고 동일 조건 독립 반복을 계속한다. 두 번째 실행 `door_dyn_r02`도 접촉 0회,
49/49 action, 위치 오차 29.83 mm, yaw 오차 0.994°로 통과했지만 위치 여유는 여전히
0.17 mm에 불과하다. 세 번째 실행 `door_dyn_r03`은 접촉 0회, 49/49 action,
위치 오차 28.70 mm, yaw 오차 0.965°로 통과했다. 세 실행의 위치 오차는
29.50±0.70 mm, yaw 오차는 1.004±0.044°이다. 명목 반복은 3/3 통과했지만
체계적인 종단 편향과 작은 위치 여유를 강건성 위험요인으로 유지한다.

S자 통로와 장애물 우회는 초기 형상에서 차체 길이와 최대 곡률에 비해 진입거리가
부족했다. 차체 외곽선과 최대 곡률을 기준으로 접근 거리를 늘리고 장애물 거리장
휴리스틱을 공통 적용한 300,000-node 전진 전용 비교를 수행했다. 두 시나리오에서
고정 역상조향과 제안 full dynamics는 내부 재생을 통과했다. 운동학 다중모드와
전환비용만 반영한 다중모드는 계획 경로를 찾았지만 동일 full-dynamics 재생에서
충돌 또는 종단 오차로 실패했다. 이 값은 13.73°/s 초기 조향속도 가정에 따른 후보로,
S자에서 118 action/5.90 m/모드 전환 5회, 장애물 우회에서
95 action/4.75 m/모드 전환 3회이며 둘 다 후진은 없다.

장애물 우회 첫 Isaac 실행 `bypass_dyn_r01`은 접촉 0회와 95/95 action 완료에도
위치 오차 109.43 mm로 실패했다(yaw 0.667°). 마지막 큰 후륜 조향 전환에서 계획은
13.73°/s를 가정했지만 실제 유효 전환은 약 8.3°/s였고, 목표 −17.9°에 대해 실제
−29.4°가 남았다. 이 실행은 성공 반복에 포함하지 않고 큰 각도 조향속도 보정 자료로
분류한다. 보수적인 8.0°/s로 다시 계획한 후보는 96 action, 4.80 m, 후진 0회,
모드 전환 1회이며 내부 full-dynamics 재생을 통과했다. 이 재보정 후보는 Isaac 실행
`bypass_rate8_dyn_r01`에서도 접촉 0회, 96/96 action, 위치 오차 22.35 mm,
yaw 오차 0.251°로 통과했다. 최초 보정 실행보다 위치 오차가 약 79.6% 감소했고
내부 예측 23.29 mm와도 일치했다. 장애물 우회는 일반 Nav2와의 비교를 위한 대조
시나리오로 유지하며, 현재 단계에서 반복 횟수를 늘려 제안법의 핵심 근거로 사용하지
않는다.

동일한 8.0°/s를 별도 S자 통로에 적용한 공통 300,000-node 비교에서는 고정
역상조향과 제안 full dynamics가 내부 재생을 통과했다. 운동학 다중모드는 충돌했고,
전환 비용만 반영한 경로는 위치 오차 199.3 mm로 실패했다. 제안 경로는 119 action,
5.95 m, 후진 0회, 모드 전환 1회이며 내부 종단 오차는 20.28 mm / -0.310°이다.
Isaac 독립 반복 `s_rate8_dyn_r01`–`r03`은 모두 접촉 0회와 119/119 action 완료로
통과했다. 위치 오차는 각각 23.60, 28.65, 24.19 mm이고 yaw 오차는 각각
-0.697, -0.847, -0.664°이다. 평균±표본표준편차는 위치 25.48±2.76 mm,
yaw -0.736±0.097°, 수행시간 197.34±0.60 s이다. 수행시간은 0.03 m/s에서의
명목 병진시간 198.33 s와 거의 같다. 명목 재현성은 확보했지만 최악 위치 오차의
허용여유가 1.35 mm뿐이므로 조건 변화에 대한 강건성은 별도로 검증해야 한다.

동일한 8°/s 모델로 생성한 고정 역상조향 경로도 Isaac 명목 반복 3/3을 접촉 없이
119/119 action으로 통과했다. 고정 역상조향의 평균±표본표준편차는 위치
23.07±2.22 mm, yaw -0.131±0.087°, 수행시간 193.18±0.73 s로, 제안 경로의
25.48±2.76 mm, -0.736±0.097°, 197.34±0.60 s보다 모두 유리했다. 따라서 S자
명목 조건은 모드 전환의 우월성 사례가 아니다. 제한 조건에서 이득이 없으면 기존
역상조향을 유지하는 상위 선택 규칙과, 불필요한 전환을 억제해야 한다는 근거로 사용한다.

## 현재 판정

**90° 코너, 좁은 T자 교차로, 문 직후 회전 및 S자 통로의 명목 반복 게이트는
통과했다.**
90° 코너에서 보정된 제안 경로가 Isaac Sim의 실제 충돌체와 강체 동역학 조건에서
접촉 없이 완료되었고 종단 위치·yaw 기준을 통과했으며, 독립 반복 3회 모두 통과했다.
재보정한 장애물 우회도 첫 Isaac 재생을 통과했다. S자 통로는 독립 반복 3/3을
통과했다. 다음은 기준 설정을 동결하고 강건성 screening으로 이동한다. S자의 작은
최악 위치 여유는 우선 확인할 위험요인으로 유지한다.
아직 통로 폭·마찰·하중·CoM 등의 변화에 대한 강건성이나 기존 방법 대비 우월성을
확정하지 않는다.

아직 다음을 주장할 수 없다.

- 30,000 nodes 내 미발견을 기하학적 불가능이라고 표현
- 내부 전파 모델 재생을 Isaac Sim 동역학 검증이라고 표현
- 두 명목 조건으로 강건성 또는 일반성을 주장
- 현재 Python 탐색시간을 실시간 계획 성능이라고 표현

## 다음 게이트

### 계획–실행 전환 정의 일치화 (screening)

Isaac ROS joint-state 안정 구간에서 목표 0°일 때 실제 조향각 중앙값은 전륜 -0.99°,
후륜 -1.03°였다. 이를 반영해 임시 tracking/zero offset을 두 축 모두 -1.0°로
적용했다. 90° 코너 `general90_bias_prop_r01`과 `r02`는 접촉 없이 각각 54/54,
53/53 action을 완료했지만 위치 오차가 123.72 mm와 138.55 mm여서 실패했다.
두 실행 모두 성공 통계에서 제외한다.

`r02`의 구간별 궤적을 분석한 결과 주행 중 yaw 응답 자체는 측정 조향각 기반 모델과
대체로 일치했지만, 정지 모드 전환 중 계획에 없는 변위가 누적되었다. 두 번째와 세 번째
정지 전환에서만 약 36 mm와 41 mm의 차체 이동이 발생했다. 원인은 lattice가 조향각을
8°/s로 바꾸면서 계속 이동하는 rolling transition을 전파하는 반면, replay executor는
모드 변경 때 정지 조향을 수행한 계획–실행 불일치였다. 또한 실행기는 첫 목표 조향을
미리 맞춘 후 원점을 재설정하지만 기존 planning screen은 첫 구간에서 0°부터 조향했다.

이를 수정해 첫 조향은 정렬 완료 상태에서 계획을 시작하고, 제안 경로의 모드 전환은
lattice와 동일한 저속 rate-limited rolling transition으로 실행하도록 분리했다. 수정
모델의 공통 300,000-node 비교에서 고정 역상조향도 62 action / 3.10 m 경로를 찾았고,
제안법은 52 action / 2.60 m / 모드 전환 2회 경로를 찾았다. 두 내부 재생은 각각
18.7 mm / -0.19°와 23.2 mm / 0.49°로 통과했다. 따라서 현재 기여 후보는 “고정
역상조향이 불가능”이 아니라 **동일 코너에서 조향모드 전환으로 경로 길이를 0.50 m
(16.1%) 줄이는 것**이다. 다음 실행 `general90_rolling_prop_r01`은 이 정의를 적용한
첫 Isaac 검증이며, 성공 전에는 센서 연동이나 논문용 반복 비교로 넘어가지 않는다.

`general90_rolling_prop_r01`은 접촉 없이 52/52 action을 완료하고 yaw 오차 0.097°를
보였으나 위치 오차 71.60 mm로 실패했다. 안정 구간의 차체좌표계 속도를 실제 조향각
기하와 비교하면 횡속도 응답비는 전륜 단독 1.00542, 역상 동시 0.92434, 후륜 단독
0.99375였고, 후륜 단독 yaw-curvature 응답비는 0.96430이었다. 기존 횡속도 응답비
0.9431/1.0/1.0508과의 차이가 전륜 구간에서 약 30 mm의 횡오차를 누적시켰다. 따라서
`r01`은 검증 결과가 아니라 응답계수 식별 자료로 분리한다. 갱신 계수로 다시 계획한
경로는 고정 역상조향 62 action / 3.10 m, 제안법 51 action / 2.55 m이며, 제안법은
0.55 m(17.7%) 짧다. 독립 확인 실행 ID는 `general90_rolling_prop_r02`로 한다.

`general90_rolling_prop_r02`는 접촉 0회, 51/51 action 완료, 위치 오차 13.08 mm,
yaw 오차 0.182°로 모든 기준을 통과했다. 명령 경로는 2.55 m, 수행시간은
86.34 s이고, 이동시간 대비 비율은 1.016이다. 따라서 rolling transition과 갱신한
횡속도 응답을 포함한 제안 경로의 Isaac 동역학 성립성을 확인했다. 다음은 동일
실행 설정의 고정 역상조향 3.10 m 경로를 1회 확인하여 공정한 직접 비교를 닫는다.

고정 역상조향 `general90_rolling_fixed_r01`은 접촉 0회와 62/62 action 완료,
yaw 오차 0.549°를 보였지만 위치 오차 37.75 mm로 30 mm 기준을 초과했다.
수행시간은 105.07 s이다. 동일 조건의 제안법은 2.55 m / 86.34 s / 13.08 mm,
고정 역상조향은 3.10 m / 105.07 s / 37.75 mm이다. 제안법은 경로 0.55 m
(17.7%)와 수행시간 18.73 s(17.8%)를 줄이면서 종단 기준을 통과했고, 비교군은
완주했지만 종단 기준을 통과하지 못했다. 이 실패를 제외하거나 PASS가 나오도록
비교군만 재튜닝하지 않고, 명목 직접 비교 결과로 그대로 기록한다.

고정 역상조향 1회 비교가 통과하면 occupancy map/센서 폐루프 연동으로 이동한다.
대규모 강건성 매트릭스와 논문용 반복은 진행 승인 후로 분리하고, 센서 연동
전에는 직접 비교를 닫는 최소 실행만 수행한다.

1. 기준선 결과와 실행 설정을 동결하고, 각 시나리오의 baseline 후보·geometry·마찰·
   하중/CoM·초기 조향 오차를 분리한 실험표를 확정한다.
2. 먼저 한 요인씩 바꾸는 screening으로 민감한 요인을 찾고, baseline과 극단 조건은
   독립 반복한다. 시나리오·속도·action sequence·제어 gains 등 나머지 값은 고정한다.
3. 성공률(무충돌·완료·종단 허용오차 동시 충족), 최소 실제 장애물 여유, 위치/yaw 오차,
   수행시간 및 경로 길이를 비교한다. 실패 실행도 제외하지 않고 기록한다.
4. 이 게이트 이후 실제 센서 기반 occupancy map/폐루프 planning에 연결하고, 센서 기반
   결과와 현재 고정 경로의 동역학 재생을 구분해 비교한다.

### 교수 보고 전 compact gate

실물 플랫폼 진행 승인 여부를 판단하기 위한 보고 전 범위는 최종 논문 전체 매트릭스와
구분한다. 개발 과제는 `s_corridor`와 `doorway_immediate_turn`, 직접 비교 정책은
`fixed_counterphase`와 `proposed_full_dynamics`로 동결한다.

1. 명목 조건에서 두 정책을 각 3회 재생한다. 기존 S자 제안법 3회는 동일한 동결 경로라
   재사용하고, S자 고정 역상조향 3회와 문 직후 회전 두 정책 각 3회를 추가한다.
2. CICS에서 사용한 시뮬레이션 민감도 범위 중 `290 kg/front x=+0.10 m`와 명목 마찰을
   25% 낮춘 `static/dynamic=0.60/0.45`를 보고 전 극단 조건으로 사용한다. 이는 실측값이
   아니라 명시적인 민감도 조건이다.
3. 두 극단 조건에서 두 정책×두 과제×3회, 총 24회를 수행한다. 명목 추가 9회를 합쳐
   교수 보고 전 추가 실행은 33회이다.
4. 초기 조향 오차, 통로 여유, 명령 지연, 관측 잡음의 전체 강건성 셀과 10회 held-out
   반복은 진행 승인 후 최종 논문 시뮬레이션으로 남긴다. 오프라인 ablation 결과는
   보고 자료에 포함하되 Isaac 반복이 아직 없음을 명시한다.

실행기는 `replay_policy`와 `experiment_condition_id`를 결과 JSON에 기록한다. 질량·CoM과
마찰 조건은 `isaac_prepare_ijcas_robustness_condition.py`가 Session Layer에만 적용하며,
finalizer가 stage와 replay의 조건 ID가 다르면 결과 저장을 거부한다.

### 좁은 T자 교차로 첫 Isaac 반복 실행 순서

1. Isaac Sim에서 저장된 IJCAS USD를 새로 열고 Timeline을 Stop한다.
2. Script Editor에서 `ijcas_collision_scenario = "narrow_t_intersection"`,
   `ijcas_collision_run_id = "generalt_dyn_r01"`을 설정한 뒤
   `isaac_prepare_ijcas_full_envelope_obstacle.py`를 실행한다.
3. 별도 터미널에서 다음을 실행한다.

   ```bash
   cd /home/jk/Research/IJCAS_ws
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ros2 launch patient_initial_stability sim_ijcas_platform.launch.py platform_profile:=lab_hy
   ```

4. Isaac Sim에서 Play를 누른 다음, 다른 터미널에서 아래 replay 명령을 실행한다.

   ```bash
   cd /home/jk/Research/IJCAS_ws
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ros2 launch patient_initial_stability sim_spatial_utility_replay.launch.py \
     scenario:=narrow_t_intersection \
     run_id:=generalt_dyn_r01 \
     speed_m_s:=0.03 \
     mode_transition_settle_s:=2.0 \
     braking_distance_m:=0.03 \
     alignment_preview_speed_m_s:=0.0001
   ```

   터미널 마지막에 `SPATIAL UTILITY REPLAY RESULT`와 `Saved:` 경로가 나와야
   하며 경로는 `studies/IJCAS/results/general_navigation_dynamics_gate/dynamic_replay/`
   아래여야 한다. scenario와 run ID는 contact monitor에 설정한 값과 같아야 한다.
5. ROS launch 종료 후 Isaac Sim에서 Stop을 누르고
   `isaac_finalize_ijcas_obstacle_contact.py`를 실행한다.

## 근거 파일

- `studies/IJCAS/results/general_navigation_dynamics_gate/isaac_replay/corridor_90_turn_general90_dyn_r03_contact.json`
- `studies/IJCAS/results/general_navigation_dynamics_gate/dynamic_replay/corridor_90_turn_general90_dyn_r03.json`
- `studies/IJCAS/results/general_navigation_dynamics_gate/isaac_replay/corridor_90_turn_general90_dyn_r04_contact.json`
- `studies/IJCAS/results/general_navigation_dynamics_gate/isaac_replay/corridor_90_turn_general90_dyn_r05_contact.json`

- `studies/IJCAS/results/general_navigation_dynamics_gate/pilot_turns_30k/general_navigation_dynamics_ablation.csv`
- `studies/IJCAS/results/general_navigation_dynamics_gate/pilot_remaining_30k/general_navigation_dynamics_ablation.csv`
- `studies/IJCAS/results/general_navigation_dynamics_gate/pilot_turns_30k/general_navigation_dynamics_paths.json`
- `studies/IJCAS/results/general_navigation_dynamics_gate/pilot_remaining_30k/general_navigation_dynamics_paths.json`

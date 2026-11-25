# Architecture

## 1. 계획과 실행의 분리

| 구성 | 주요 코드 | 역할 |
|---|---|---|
| ROS 선택기 | `src/ijcas_experiments/ijcas_experiments/selection.py` | 과제·방법·초기조건·피드백·단계 선택 및 원본 입력 검증 |
| 실행 절차 | `src/ijcas_experiments/ijcas_experiments/workflow.py` | 새 출력 생성, 계획/분석/SDK 실행 분기 |
| SDK 어댑터 | `src/ijcas_experiments/ijcas_experiments/sdk_runner.py` | 별도 SimulationApp, 장면·정적 검사·경로 실행·복원 |
| 계획 모델 | `src/ijcas_experiments/runtime/study/hospital_mission_planning.py` | 조향·차체 응답 예측과 제약 평가 |
| 공유 모델 | `src/patient_initial_stability/patient_initial_stability/` | 기하·계획·응답 모델 |
| 원본 실행 호스트 | `src/ijcas_experiments/runtime/study/hospital_revision_final_campaign.py`, `hospital_revision_live.py` 등 | 최종 고정 설정과 실행·안전 판정 |
| 센서 연결 | `src/ijcas_experiments/ijcas_experiments/sensor_runner.py` | 원본 H5/H6 센서 설정 고정, 새 센서·ROS 브리지 연결 |
| ROS 브리지 | `src/ijcas_experiments/runtime/study/hospital_sensor_ros_bridge.py` | AMCL·엔코더·Nav2 실행 및 시간 대응 |
| Native Nav2 plugin | `src/ijcas_nav2_sensor_adapter/src/plugins.cpp` | 제어 제안 검증과 costmap 기반 충돌 검사 |
| 결과 재계산 | `src/ijcas_experiments/ijcas_experiments/reproduce.py` | 고정 선택 검증, 원본 trace/제어 자료의 지표 재계산 |

내부 호스트는 과거 실행 코드와 수치 연산 순서를 보존한다. 파일명에 과거 과제나 버전명이 있더라도 개별 파일을 새 주행 명령으로 직접 실행하지 않는다. 사용 진입점은 `ros2 run ijcas_experiments experiment` 또는 `experiment.launch.py`다.

## 2. 계획 상태와 후보 선택

상태는 `(x, y, yaw, delta_front, delta_rear, u, v, r, mode, direction)`을 포함한다. F/R/C 조향 모드와 전후진 방향에 대해 공간 경로를 만들고, 조향 변화율과 차체 속도 응답을 전파하여 구간을 연결한다. 전체 외곽·근사 force/load·감속·방향 전환 조건을 검사하고 정해진 예산 내 첫 허용 후보를 선택한다. 전역 최적성을 주장하지 않는다.

상태 이력 제거 `P_no_history`는 구간 진입의 조향/속도 상태를 정상 입력 상태로 투영하는 **예측 모델 ablation**이다. 실제 로봇의 pose·조인트·속도를 구간마다 재설정하는 실험이 아니다.

## 3. 피드백 경로

### 본 비교

```text
P 또는 baseline 경로 → 공통 추종기 → 조향/구동 관절
                           ↑
                 simulator base_link pose
```

### 별도 센서 통합

```text
native virtual scan ─────────────→ AMCL ───────┐
joint encoders → encoder odometry / twist ────┤
                                             ↓
fixed P path + modes / directions / speeds → P execution logic
                                             ↓
                              Nav2 FollowPath + custom controller / guard
                                             ↓
                                   steering / drive joints
```

Nav2 plugin 식별자는 `src/ijcas_nav2_sensor_adapter/plugins.xml`에 있다. 이 경로는 일반적인 MPPI/Smac 기반 자유 Goal 주행과 다르다. H5/H6의 보관된 P 경로를 사용하며, 경로 밖의 모드·속도 메타데이터를 버리거나 Nav2에 새 공간 경로 최적화를 위임하지 않는다.

시뮬레이터 데이터는 지도 구성, 정적 진입 검사, 안전 감시와 독립 점수 산출에 사용한다. 센서 피드백 제어에 simulator pose를 대신 공급하는 fallback은 없다.

## 4. Python 환경과 시계

ROS Humble 브리지는 시스템 Python 3.10, Isaac Sim은 SDK Python 3.11의 별도 프로세스다. SDK 실행 시 ROS `PYTHONPATH`를 제거하여 바이너리 모듈 충돌을 피한다. 이것은 별도 `setup_env.sh`를 요구하지 않으며 표준 `install/setup.bash`가 ROS 진입점을 제공한다.

SDK 시작은 원래 full GUI 경험의 `/app/player/useFixedTimeStepping=true`를 유지한다. 기본 Python 경험의 부동소수점 timeline 차이가 제어 경계 통과 시점을 바꿀 수 있으므로 원본 clock 설정을 명시적으로 복원한다. 센서 timestamp나 제어 임계값을 후처리하여 맞추지 않는다.

## 5. 실행 권한과 원본 보호

- `check`, `plan`, `analyze`는 물리 주행 권한을 만들지 않는다.
- `run`은 현재 선택의 명시적 `allow_motion=true`가 필요하다.
- 새 로봇과 원본 정적 검사를 통과해야 실제 경로를 시작한다.
- 원본 로그, 이미 소비된 실행 폴더, 과거 승인 기록을 새 실행 권한으로 재사용하지 않는다.
- 모든 실제 충돌·추종·정지·접촉 하중 판정은 유지한다.
- 센서 오류 시 가짜 free-space return, GT 제어 전환 또는 자동 주행 재시도를 하지 않는다.
- 결과는 새 폴더에 저장하며 본 비교의 고정 선택·분모를 변경하지 않는다.

`src/ijcas_experiments/ijcas_paths.py`는 원본의 과거 절대경로를 현재 워크스페이스의 상대 위치로 해석한다. 원본 파일 내용을 바꾸지 않으며 원본 digest 검증과 분리한다. 과거 workspace 경로 문자열은 이 호환 처리의 일부이지 다른 workspace를 실행하는 fallback이 아니다.

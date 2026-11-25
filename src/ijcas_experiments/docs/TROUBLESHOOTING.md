# Troubleshooting

## `ros2` 패키지 또는 실행기를 찾지 못함

시스템 ROS 환경에서 빌드했는지 확인한 뒤 워크스페이스의 `install/setup.bash`를 source한다.

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install
source install/setup.bash
ros2 pkg executables ijcas_experiments
```

`setup_env.sh`나 Script Editor용 명령을 추가할 필요는 없다.

## 원본 protocol / geometry / map / trace가 없음

공개 코드는 약 14 GB 원본 자료를 포함하지 않는다. [SETUP.md](SETUP.md)의 상대경로로 원본과 이관 증빙을 함께 복원한다. 결과 요약 CSV나 빈 JSON으로 대체하지 않는다. `check`의 missing input은 새 물리 주행 결과가 아니다.

## Python / `rclpy` / `pxr` / Isaac 모듈 오류

ROS는 시스템 Python 3.10, Isaac은 SDK Python 3.11을 사용한다. ROS 셸에서 SDK Python 라이브러리를 수동으로 `PYTHONPATH`에 합치지 않는다. `sdk_python:=/path/to/isaac_env/bin/python`으로 별도 환경을 지정한다. 기본 선택기는 SDK 실행용 ROS 환경 경로를 분리한다.

## `output` 폴더를 사용할 수 없음

소비된 출력 폴더나 원본 자료를 덮어쓰지 않는 것이 정상이다. 기본 timestamp 출력을 쓰거나 존재하지 않는 새 폴더를 지정한다. 이전 FAIL/HOLD 기록을 삭제해서 재실행 가능 상태로 만들지 않는다.

## 센서 모드에서 H1–H4 / 다른 method / condition 2·3을 거부함

현재 등록된 센서 실험은 별도 H5/H6 P 사례 각 하나다. `condition:=1`은 본 비교 초기조건 1과 다르다. 등록되지 않은 조건을 자동 생성하거나 ground truth로 대체하지 않는다.

## AMCL 초기 위치 또는 센서 검사가 HOLD

원본 prior는 위치 3 mm / yaw 0.1°다. 이전 probe의 넓은 prior를 쓰면 원본 진입 검사를 통과하지 못할 수 있다. `bridge/configuration_parity.json`과 `native/scan_configuration_parity.json`을 확인한다. 스캔은 해당 원본 실행의 native configuration을 사용해야 한다.

native hit의 nonfinite 값, ray 이탈 또는 원본 삼각면 불일치는 실제 센서 입력 오류로 보존한다. 거리값 수선, 가짜 무장애 return, 임계값 확대 또는 GT 제어 전환으로 통과시키지 않는다.

## 같은 설정인데 GUI 실행과 출발 시각이 달라짐

`simulation_clock_configuration.json`에서 `/app/player/useFixedTimeStepping=true`를 확인한다. Isaac 5.0의 full GUI와 Python 경험의 기본 timeline arithmetic이 다를 수 있다. 현재 SDK 어댑터는 원래 GUI clock 모드를 명시적으로 유지한다. timestamp 반올림이나 정지/출발 임계값 변경으로 숨기지 않는다.

## Nav2 action은 성공인데 임무 완료가 아님

action 결과와 simulator 기반 완료 기준을 따로 확인한다. 최종 위치/yaw, 5초 정지 유지의 마지막 3초, raw/위치 기반 속도·드리프트·접촉 하중 조건을 모두 만족해야 한다. 단위시험 성공이나 모델 감사 통과를 실제 임무 완료로 바꾸지 않는다.

## 센서 실행이 중단되어 RMS 계산 오류가 남음

부족한 trace에서 동시각 paired metric을 만들 수 없으면 `sensor_metrics_error.json`을 남기며 원래 HOLD를 유지한다. 실패 구간을 성공으로 바꾸거나 임의 보간해 완주 수치를 만들지 않는다. PASS로 보고된 실행의 지표 오류는 정상 완료로 무시하지 않는다.

## 과거 전체 테스트 중 일부가 실패함

일부 과거 fixture는 SDK USD 라이브러리나 보관된 과거 소스/경로를 요구한다. 최종 실행 인터페이스 75개 통과와 모든 역사적 테스트 통과는 다른 주장이다. 대상 테스트, Python 환경, 원본 묶음 유무를 함께 기록한다. fixture를 없애 실제 안전 검증을 통과한 것처럼 표시하지 않는다.

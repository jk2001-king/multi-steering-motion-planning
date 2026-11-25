# Setup

## 1. 요구 환경

- Ubuntu 22.04, ROS 2 Humble, colcon/rosdep
- NVIDIA Isaac Sim 5.0과 해당 버전이 지원하는 GPU·드라이버
- ROS: 시스템 Python 3.10; Isaac: 별도 SDK Python 3.11
- ROS의 Nav2 / map server / lifecycle / TF / 메시지 의존성
- Python 수치 라이브러리: NumPy, PyYAML; JIT 경로에는 Numba와 호환 llvmlite
- Hospital/Grid 외부 USD 자산 접근 및 로봇 모델의 로컬 USD 참조

Isaac/드라이버는 NVIDIA의 해당 버전 설치 절차를 따른다. 시스템 ROS Python 환경에 SDK 바이너리를 합치거나, conda Python으로 ROS 패키지를 빌드하지 않는다.

## 2. 복제와 ROS 빌드

```bash
git clone https://github.com/jk2001-king/multi-steering-motion-planning.git ~/IJCAS_ws
cd ~/IJCAS_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --base-paths src --symlink-install
source install/setup.bash
ros2 run ijcas_experiments experiment --list
```

이미 빌드한 PC의 새 터미널에서는 워크스페이스의 `source install/setup.bash`를 사용한다. 프로젝트 전용 셸 소싱 파일은 필요하지 않다. 로컬 작업 경로가 `~/Research/IJCAS_ws`이면 해당 경로를 사용해도 된다.

## 3. 코드와 원본 입력의 구분

Git에는 실행 코드·설정·로봇 모델·측정 결과 요약만 있다. 약 16.5 GB의 원본 실행 자료와 이관 증빙은 로컬에 보존되어 있으며 Git 배포본에는 없다. 실행 또는 원본 수치 재계산을 하려면 소유자가 제공하는 원본 자료를 다음 상대경로로 복원해야 한다.

```text
IJCAS_ws/archive/original/
├── studies/IJCAS/results/       # fixed definitions, plans, geometry/grid, raw runs
├── studies/IJCAS/reports/       # referenced technical and sensor evidence
├── studies/CICS26/              # calibration evidence used by archived definitions
├── migration/                  # original source hashes and relocation receipts
└── data/                       # referenced original technical data
```

이 밖에 해시 검증용 정리 전 소스·USD 사본을 포함한 `archive/original`과 `archive/layout_20261126/{before.json,source_manifest.json,usd_references.json}`도 원본 자료 묶음에서 함께 복원한다. 일부 폴더만 복사하면 저장된 source ZIP의 호환성 검증을 통과할 수 없다. 이 목록의 임의 빈 파일이나 `docs/data`의 요약 표로 누락 입력을 대체하면 안 된다. 공개 데이터 다운로드 링크는 현재 제공하지 않는다.

```bash
ros2 run ijcas_experiments check
ros2 launch ijcas_experiments experiment.launch.py \
  scenario:=H2 method:=P condition:=3 stage:=check
```

입력이 없으면 확인 명령은 경로를 보고하고 실패한다. 이는 실행 완료나 주행 실패의 측정 결과가 아니다.

## 4. SDK Python 지정

```bash
ros2 launch ijcas_experiments experiment.launch.py \
  scenario:=H2 method:=P condition:=3 stage:=plan \
  sdk_python:=/path/to/isaac_env/bin/python
```

탐색 순서는 명시 `sdk_python`, `ISAAC_SIM_PYTHON`, 설치된 `isaacsim` 실행기, 기존 로컬 SDK 환경이다. 다른 PC에서는 명시 경로를 사용하는 편이 분명하다. SDK 안에 수치 패키지가 없으면 해당 SDK 환경의 Python으로 설치한다. 기존 검증 환경과 버전이 다르면 새 결과를 별도 검증해야 한다.

## 5. 실행 단계

| 단계 | 동작 | 새 주행 |
|---|---|---|
| `check` | 선택된 원본 조건과 입력 확인 | 없음 |
| `plan` | 본 비교: 새 오프라인 계획 / 센서: 고정 경로 독립 감사 | 없음 |
| `analyze` | 선택된 원본 실행 지표 재계산 | 없음 |
| `run` | 새 source·정적 검사·조건부 한 경로 주행 | 명시 승인 필요 |

```bash
# 본 비교 한 조건
ros2 launch ijcas_experiments experiment.launch.py \
  scenario:=H2 method:=P condition:=3 feedback:=ground_truth \
  stage:=run allow_motion:=true headless:=true \
  sdk_python:=/path/to/isaac_env/bin/python

# 별도 센서 사례; H6는 scenario만 변경
ros2 launch ijcas_experiments experiment.launch.py \
  scenario:=H5 method:=P condition:=1 feedback:=amcl_encoder \
  stage:=run allow_motion:=true headless:=true \
  sdk_python:=/path/to/isaac_env/bin/python
```

기존 Isaac 실행 객체를 이어받는 구조가 아니다. 장면과 원본 준비 규칙을 새 SDK 프로세스에서 구성한다. 물리 주행 중에는 장면·조인트·설정을 수동 변경하지 않는다. 창 없는 실행도 실제 PhysX 주행이며 단순 trace 재생이 아니다.

## 6. 결과 위치와 검증

- 새 실행: `log/ijcas/experiments/<timestamp>_<cell>/`
- 원본 수치 재계산: `log/ijcas/reproduced/<timestamp>/`
- 새 센서 지표: 해당 실행의 `sensor_metrics/`
- 로컬 원본 보존 검사: 소싱 후 `ros2 run ijcas_experiments check --full` — 전체 원본 묶음 필요
- USD 참조 검사: SDK Python에 `src/ijcas_experiments`를 PYTHONPATH로 지정하고 `src/ijcas_experiments/tools/check_usd_assets.py` 실행 — 출력은 `log/ijcas/usd_dependencies.json`

```bash
python3 -m pytest src/ijcas_experiments/test -q
ros2 run ijcas_experiments reproduce_results
```

단위시험·오프라인 계획 성공은 실제 정지·접촉·전체 임무 성공과 별개다. 자세한 판정과 현재 검증 범위는 [실험 문서](EXPERIMENTS.md)를 참고한다.

경로 이전 소스/모델의 증빙 대조에는 `archive/layout_20261126`의 명세도 원본 자료 묶음과 함께 복원한다. 원본 SHA가 다르거나 현재 소스가 검증 명세와 다르면 실행 호환성 검사는 실패하며 자동 허용하지 않는다.

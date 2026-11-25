# HY 실물 플랫폼 shadow-mode planning gate

## 목적

실제 연구실 `_HY` 플랫폼의 localization, Nav2 costmap, goal을 사용해 IJCAS
조향모드 계획기를 실행하되 구동 명령은 내보내지 않는다. 이 단계는 센서 기반
온라인 계획 인터페이스와 계산시간을 검증하는 단계이며 실물 주행시험이 아니다.

## 안전 경계

`lab_hy_shadow_planning.launch.py`가 시작하는 노드는 다음 두 개뿐이다.

- `patient_transport_platform_profile`
- `lab_hy_shadow_occupancy_lattice_planner`

출력은 `/ijcas/shadow/plan`, `/ijcas/shadow/path`,
`/patient_transport/platform_profile`뿐이다. `/cmd_vel`, `/mode`,
`/docking/cmd_vel`, `/sim/joint_command` publisher나 실행기는 시작하지 않는다.

## 사전 확인

실물 HY bringup을 실행한 뒤 다음 타입과 frame을 확인한다.

```bash
ros2 topic info /global_costmap/costmap_raw -v
ros2 topic info /localization -v
ros2 topic info /goal_pose -v
ros2 topic echo /global_costmap/costmap_raw --once --field header
ros2 topic echo /localization --once --field header
```

기본 계약은 다음과 같다.

- map: `/global_costmap/costmap_raw`, `nav_msgs/msg/OccupancyGrid`, frame `map`
- pose: `/localization`, `nav_msgs/msg/Odometry`, frame `map`
- goal: `/ijcas/shadow/goal`, `geometry_msgs/msg/PoseStamped`, frame `map`
- steering: `/steer_angle_deg`, `std_msgs/msg/Float32MultiArray`; 첫 두 값은
  `[front_actual_deg, rear_actual_deg]`

실물 Nav2의 navigation action이나 기존 `/goal_pose`와 분리하기 위해 shadow 전용
goal topic을 사용한다. RViz의 Nav2 Goal 버튼은 누르지 않는다.

실제 topic 이름이 다르면 launch argument로만 바꾸고 소스의 기본값을 임의로
수정하지 않는다. 세 입력 frame이 같지 않으면 planner는 계획을 거부한다.

## 실행

```bash
cd /home/jk/Research/IJCAS_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch patient_initial_stability lab_hy_shadow_planning.launch.py \
  platform_profile:=lab_hy \
  run_id:=lab_hy_shadow_r01 \
  map_topic:=/global_costmap/costmap_raw \
  odometry_topic:=/localization \
  goal_topic:=/ijcas/shadow/goal \
  steering_state_topic:=/steer_angle_deg \
  speed_m_s:=0.03 \
  robust_clearance_m:=0.20 \
  occupied_threshold:=99 \
  replan_period_s:=1.0
```

별도 터미널에서 실제로 actuator topic을 소유하지 않는지 확인한다.

```bash
ros2 node info /lab_hy_shadow_occupancy_lattice_planner
ros2 topic info /ijcas/shadow/plan -v
ros2 topic info /ijcas/shadow/path -v
ros2 topic echo /ijcas/shadow/plan --once --full-length
```

RViz에서 `/ijcas/shadow/path`를 추가한다. 목표는 Nav2 Goal action이 아니라
`/ijcas/shadow/goal`에 `PoseStamped`로 별도 발행한다. 계획 결과가 나오더라도
로봇은 움직이면 안 된다.

## 기록

첫 acceptance bag은 다음 항목만 기록한다.

```bash
ros2 bag record -o studies/IJCAS/results/physical_shadow/lab_hy_shadow_r01 \
  /patient_transport/platform_profile \
  /localization \
  /odom \
  /steer_angle_deg \
  /global_costmap/costmap_raw \
  /ijcas/shadow/goal \
  /ijcas/shadow/plan \
  /ijcas/shadow/path \
  /tf \
  /tf_static
```

## 통과 기준

- profile이 `lab_hy`이다.
- map/pose/goal frame이 모두 `map`이다.
- 목표 입력 후 shadow plan과 path가 발행된다.
- 계획 JSON의 `planning_mode`가 `physical_shadow`이다.
- planner가 `/cmd_vel`, `/mode`, `/docking/cmd_vel`, `/sim/joint_command`를
  발행하지 않는다.
- 정지 플랫폼에서 1 Hz 재계획이 지속되고 계획시간과 action sequence가 기록된다.
- 이 단계 동안 플랫폼이 움직이지 않는다.

`costmap_raw`의 inflation 비용과 계획기 여유를 중복 적용하지 않도록 기본
`occupied_threshold=99`는 lethal/inscribed 셀만 장애물로 사용한다. 현재
`0.20 m` clearance는 측정된 최종 불확실성 예산이 아니라, 기존 point-cloud
p95 처리지연과 0.2 m/s 제동 가정을 포함한 보수적 shadow 기본값이다.

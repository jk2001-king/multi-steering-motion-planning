# IJCAS patient-transport simulation platform

## Decision

IJCAS work uses a persistent simulation/ROS 2 interface rather than rebuilding
an Action Graph for every trial. The CICS initial-motion experiments remain a
separate completed study. The IJCAS topic is the effect of front/rear active
steering state and platform dynamics on feasible path planning and execution.

The platform is divided into three reproducible layers:

1. **Immutable robot asset** — the measured-geometry laboratory USD.
2. **Derived IJCAS stage** — persistent floor/contact/actuator baseline, Action
   Graph, and simulation sensors.
3. **ROS 2 experiment layer** — TF completion, planners, controllers, parameter
   files, rosbag recording, and batch execution.

## One-time build

1. Start Isaac Sim with the same `ROS_DOMAIN_ID` used by ROS 2.
2. Enable `isaacsim.ros2.bridge` and `isaacsim.sensors.physics`.
3. Open `isaac/assets/patient_transport_lab/patient_transport_lab.usd` fresh.
4. Keep the Timeline stopped.
5. Run `isaac/scripts/isaac_build_patient_transport_ijcas_stage.py` in the
   Script Editor.
6. Run `isaac/scripts/isaac_validate_patient_transport_ijcas_stage.py`.

The builder creates
`isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd`. It refuses
to overwrite an existing derived stage. The source path and SHA-256 are stored
in the derived root layer for traceability. The previously repeated 120 Hz
floor, compliant-contact, and powered-idle preparation is also stored once in
this derived stage. Its values remain explicitly labelled `provisional_v1`.
The Action Graph is triggered by `OnPhysicsStep`, so command consumption,
odometry, joint state, clock, and IMU publication execute once per physics step
rather than once per rendered frame.

## Normal experiment startup

After the one-time build, do not run a graph-construction script for each
experiment.

```bash
cd /home/jk/Research/IJCAS_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=26
ros2 launch patient_initial_stability sim_ijcas_platform.launch.py \
  platform_profile:=lab_hy
```

Open the derived IJCAS USD and press Play. Verify the interface:

```bash
ros2 topic hz /clock
ros2 topic hz /joint_states
ros2 topic hz /sim/ground_truth/odom
ros2 topic hz /sim/imu
ros2 topic echo /patient_transport/platform_profile --once
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link sim_imu
```

Then run the automated five-second acceptance check:

```bash
ros2 run patient_initial_stability ijcas_interface_check --ros-args -p duration_s:=5.0
```

Do not start planning experiments until it reports `VERDICT: PASS`.

The fixed contract is documented in
`studies/IJCAS/config/patient_transport_sim_platform.yaml`.

## TF ownership

- Isaac Sim publishes the dynamic `odom -> base_link` transform.
- `robot_state_publisher` publishes `base_link -> robot links` from the URDF and
  `/joint_states`.
- The bring-up launch publishes `map -> odom` as identity only until a
  localization system owns that transform.
- The launch publishes `base_link -> sim_imu` for the ideal simulation-only
  IMU. The URDF also publishes the recovered HY `livox_frame ->
  livox_front/livox_rear/livox_top -> imu_link` chain for interface testing.

This division avoids two publishers claiming the same transform.

## Sensor policy

The `_HY` mount file identifies the low-profile laboratory platform.
Front/rear LiDAR centre height is 0.23 m and top LiDAR centre height is 0.35 m
in the ground-based `base_link` frame. The file explicitly marks X/Y and RPY
as provisional field-platform assumptions, so those components support
software/TF integration but are not calibrated physical extrinsics.

The enabled `/sim/imu` is an ideal body-frame observation used only while the
planner/controller interface is developed. It is not labelled as the physical
top MID-360 IMU. RTX point emission is also disabled. Enabling the physical
sensor model requires:

- HY-specific measurement of the provisional X/Y and RPY components;
- a reviewed MID-360 RTX profile (360-degree horizontal coverage, vertical
  field, range, rate, and non-repetitive sampling approximation);
- update rate, timestamp convention, noise, and latency;
- self-occlusion by the patient-transfer structure.

Once measured, add the sensors in a new schema/versioned stage rather than
silently changing this baseline.

## Laboratory versus field startup

There are two independent selectors and they must not be confused:

1. `tools/lidar_net_mode.sh unicon|rbio` selects the LiDAR **network
   topology**. The laboratory setup is the historical one-hub `unicon` mode;
   the field platform uses three directly connected NICs in `rbio` mode. The
   LiDAR IP table is shared, so an IP alone does not identify the platform.
2. `ros2 topic pub -1 /mode std_msgs/Int32 "{data: 1}"` selects the motor
   **drive mode** (`1=Ackermann`) after motor initialization. It does not select
   laboratory versus field geometry.

On the physical laboratory computer, run from the checked-out source repo:

```bash
sudo tools/lidar_net_mode.sh unicon if=enP8p1s0
tools/lidar_net_mode.sh status
```

Then launch the simulation/research interface with an explicit locked profile:

```bash
ros2 launch patient_initial_stability sim_ijcas_platform.launch.py \
  platform_profile:=lab_hy
```

Any other `platform_profile` fails immediately. This guards the IJCAS dataset
against accidental use of field-platform geometry. The exact NIC name must be
confirmed with `ip -br link`; do not copy `enP8p1s0` blindly to another PC.
The selected value is also published on
`/patient_transport/platform_profile` with transient-local durability and once
per second so that it is captured in experiment bags.

## Immediate research sequence

1. Build and validate the persistent platform.
2. Confirm topic rates, TF uniqueness, joint-name agreement, and command signs.
3. Reproduce a short zero-steer straight/turn trial and save a rosbag as the
   platform acceptance record.
4. Implement the independent IJCAS baseline: geometric path planning that uses
   pose only and ignores front/rear steering state.
5. Implement the proposed state-aware planner using
   `[x, y, yaw, v, yaw_rate, front_steer, rear_steer]` and separate front/rear
   steering feasibility constraints.
6. Compare matched paths under steering-state, load/CoM, speed, and friction
   perturbations using completion rate, tracking error, clearance violation,
   steering effort/rate, jerk, and traversal time.
7. Freeze parameters before held-out scenarios, then repeat on the physical
   platform after sensor extrinsics and dynamics are measured.

The first physical connection is deliberately read-only. Follow
`studies/IJCAS/docs/lab_hy_shadow_planning_gate.md`; it consumes the HY global
costmap, localization, and goal but publishes only `/ijcas/shadow/plan` and
`/ijcas/shadow/path`. No controller or trajectory executor is launched.

The two earlier diagnostic gates showed small but repeatable dynamic residuals;
they did not establish a planning contribution by themselves. The next claim
must therefore be demonstrated at task scale by fewer failures, smaller
tracking/clearance violations, or a better time/comfort/effort trade-off than
the pose-only planner.

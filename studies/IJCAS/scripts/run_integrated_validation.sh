#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_ID CONDITION_ID [PLAN_PATH]" >&2
  exit 2
fi

IJCAS_RUN_ID=$1
IJCAS_CONDITION_ID=$2
IJCAS_WORKSPACE=/home/jk/Research/IJCAS_ws
IJCAS_LEGACY_PLAN="$IJCAS_WORKSPACE/studies/IJCAS/results/rigid_body_edge_gate/corridor_90_hierarchical_rigid_body_envelope.json"
IJCAS_PLAN_PATH=${3:-$IJCAS_LEGACY_PLAN}

if [[ ! -f "$IJCAS_PLAN_PATH" ]]; then
  echo "missing plan: $IJCAS_PLAN_PATH" >&2
  exit 2
fi

IJCAS_SCENARIO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["scenario"])' \
  "$IJCAS_PLAN_PATH")
if [[ -z "$IJCAS_SCENARIO" ]]; then
  echo "plan has no scenario: $IJCAS_PLAN_PATH" >&2
  exit 2
fi

cd "$IJCAS_WORKSPACE"
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

exec ros2 launch patient_initial_stability sim_spatial_utility_replay.launch.py \
  scenario:="$IJCAS_SCENARIO" \
  run_id:="$IJCAS_RUN_ID" \
  replay_policy:=external_integrated_plan \
  external_plan_path:="$IJCAS_PLAN_PATH" \
  experiment_condition_id:="$IJCAS_CONDITION_ID" \
  speed_m_s:=0.08 \
  mode_transition_policy:=rolling_rate_limited \
  mode_transition_settle_s:=2.0 \
  braking_distance_m:=0.03 \
  minimum_braking_speed_m_s:=0.005 \
  primitive_completion_tolerance_m:=0.002 \
  stop_lead_distance_m:=0.015 \
  stopped_speed_threshold_m_s:=0.005 \
  stopped_hold_s:=0.25 \
  terminal_stop_timeout_s:=30.0 \
  terminal_pose_control_enabled:=true \
  terminal_control_position_tolerance_m:=0.015 \
  terminal_control_target_tolerance_m:=0.005 \
  terminal_control_heading_tolerance_deg:=2.0 \
  terminal_settle_abort_position_m:=0.030 \
  terminal_settle_abort_heading_deg:=3.0 \
  terminal_control_max_speed_m_s:=0.010 \
  terminal_control_min_speed_m_s:=0.003 \
  terminal_control_velocity_gain:=0.8 \
  path_tracking_feedback_enabled:=true \
  path_tracking_lateral_gain:=4.0 \
  path_tracking_heading_gain:=1.0 \
  path_tracking_max_curvature_correction:=0.25 \
  alignment_preview_speed_m_s:=0.0001

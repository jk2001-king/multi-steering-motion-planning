#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {kinematic|dynamics} PAIR_ID" >&2
  exit 2
fi

IJCAS_VARIANT=$1
IJCAS_PAIR_ID=$2
IJCAS_WORKSPACE=/home/jk/Research/IJCAS_ws
IJCAS_PLAN_DIR="$IJCAS_WORKSPACE/studies/IJCAS/results/rigid_body_edge_gate/operational_dynamics_causality_r01/isaac_plans"

case "$IJCAS_VARIANT" in
  kinematic)
    IJCAS_RUN_ID="general90_causal_kin_${IJCAS_PAIR_ID}"
    IJCAS_PLAN_PATH="$IJCAS_PLAN_DIR/corridor_90_multimode_kinematic.json"
    ;;
  dynamics)
    IJCAS_RUN_ID="general90_causal_dyn_${IJCAS_PAIR_ID}"
    IJCAS_PLAN_PATH="$IJCAS_PLAN_DIR/corridor_90_proposed_full_dynamics.json"
    ;;
  *)
    echo "variant must be 'kinematic' or 'dynamics'" >&2
    exit 2
    ;;
esac

cd "$IJCAS_WORKSPACE"
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

echo "=== IJCAS operational dynamics causal replay ==="
echo "variant: $IJCAS_VARIANT"
echo "run id:  $IJCAS_RUN_ID"
echo "plan:    $IJCAS_PLAN_PATH"

exec ros2 launch patient_initial_stability sim_spatial_utility_replay.launch.py \
  scenario:=corridor_90_turn \
  run_id:="$IJCAS_RUN_ID" \
  replay_policy:=external_integrated_plan \
  external_plan_path:="$IJCAS_PLAN_PATH" \
  experiment_condition_id:=nominal_265_center \
  speed_m_s:=0.03 \
  mode_transition_policy:=rolling_rate_limited \
  mode_transition_settle_s:=2.0 \
  steering_rate_limit_deg_s:=8.0 \
  braking_distance_m:=0.03 \
  minimum_braking_speed_m_s:=0.005 \
  primitive_completion_tolerance_m:=0.002 \
  stop_lead_distance_m:=0.015 \
  stopped_speed_threshold_m_s:=0.005 \
  stopped_hold_s:=0.25 \
  terminal_stop_timeout_s:=8.0 \
  terminal_neutral_alignment_enabled:=true \
  terminal_pose_control_enabled:=false \
  terminal_control_position_tolerance_m:=0.030 \
  terminal_control_target_tolerance_m:=0.005 \
  terminal_control_heading_tolerance_deg:=2.0 \
  terminal_settle_abort_position_m:=0.040 \
  terminal_settle_abort_heading_deg:=3.0 \
  path_tracking_feedback_enabled:=false \
  alignment_preview_speed_m_s:=0.0001

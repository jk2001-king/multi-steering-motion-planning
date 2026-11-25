"""Publish a CPU-PhysX 2D LaserScan for the IJCAS sensor closed-loop gate.

This intentionally avoids RTX LiDAR, render products and Replicator.  The
local RTX 3070/driver/Isaac combination repeatedly produced NVIDIA Xid 31 and
CUDA error 700 during sustained RTX LiDAR operation.  Rays originate just
outside the measured platform envelope so the robot cannot scan itself.

Open the persistent IJCAS USD with the Timeline stopped, run this script once,
start the normal ROS platform interface, and then press Play.
"""

import math
import gc
from pathlib import Path

import carb
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation
from pxr import PhysxSchema, Usd, UsdPhysics


WORKSPACE = Path("/home/jk/Research/IJCAS_ws")
TARGET_USD = (
    WORKSPACE / "isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd"
).resolve()
ARTICULATION_ROOT = "/World/patient_transport_lab_screening/base_link"
TOPIC = "/scan"
FRAME = "base_link"
GRAPH = "/World/IJCAS/CPULaserScanGraph"

PUBLISH_RATE_HZ = 5.0
RAY_COUNT = 360
ANGLE_MIN = -math.pi
ANGLE_INCREMENT = 2.0 * math.pi / RAY_COUNT
RANGE_MIN_M = 0.10
RANGE_MAX_M = 12.0
SCAN_HEIGHT_M = 0.35

# Caster-inclusive measured envelope is approximately 1.63 x 0.965 m.
# Start each ray beyond the rectangular envelope to suppress robot self-hits.
HALF_LENGTH_M = 0.815
HALF_WIDTH_M = 0.4825
ENVELOPE_MARGIN_M = 0.03


def _yaw_from_wxyz(quaternion):
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _distance_to_envelope(direction_x, direction_y):
    x_limit = (
        HALF_LENGTH_M / abs(direction_x)
        if abs(direction_x) > 1.0e-9
        else math.inf
    )
    y_limit = (
        HALF_WIDTH_M / abs(direction_y)
        if abs(direction_y) > 1.0e-9
        else math.inf
    )
    return min(x_limit, y_limit) + ENVELOPE_MARGIN_M


def _require_extension(extension_id):
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(extension_id):
        manager.set_extension_enabled_immediate(extension_id, True)
    if not manager.is_extension_enabled(extension_id):
        raise RuntimeError(f"Failed to enable extension: {extension_id}")


def prepare():
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise RuntimeError("Press Stop before preparing the CPU LaserScan")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open")
    root_path = Path(stage.GetRootLayer().realPath).resolve()
    if root_path != TARGET_USD:
        raise RuntimeError(f"Open the persistent IJCAS USD first: {TARGET_USD}")
    if not stage.GetPrimAtPath(ARTICULATION_ROOT).IsValid():
        raise RuntimeError(f"Missing articulation root: {ARTICULATION_ROOT}")
    existing_subscription = globals().get("ijcas_cpu_scan_subscription")
    graph_is_current = stage.GetPrimAtPath(GRAPH).IsValid()
    if existing_subscription is not None and graph_is_current:
        print("=== IJCAS CPU LaserScan already prepared ===")
        print("Current stage graph and physics callback are both valid.")
        return
    if existing_subscription is not None and not graph_is_current:
        # File -> Open replaces the USD stage but does not reset Script Editor
        # globals. Drop the callback owned by the previous stage before making
        # a publisher for the current one.
        globals()["ijcas_cpu_scan_subscription"] = None
        globals()["ijcas_cpu_scan_robot"] = None
        globals()["ijcas_cpu_scan_depth_attribute"] = None
        globals()["ijcas_cpu_scan_timestamp_attribute"] = None
        globals()["ijcas_cpu_scan_state"] = None
        globals()["ijcas_cpu_scan_graph"] = None
        del existing_subscription
        gc.collect()
        print("Discarded stale CPU LaserScan callback from the previously opened USD.")
    if stage.GetPrimAtPath(GRAPH).IsValid():
        raise RuntimeError("CPU LaserScan graph already exists; reopen the USD")

    # Keep all physics on CPU. This script does not enable any RTX sensor.
    physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(physics_scenes) != 1:
        raise RuntimeError(f"Expected one physics scene, found {len(physics_scenes)}")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scenes[0])
    physx_scene.CreateEnableGPUDynamicsAttr(False).Set(False)
    physx_scene.CreateBroadphaseTypeAttr().Set("MBP")

    _require_extension("isaacsim.core.nodes")
    _require_extension("isaacsim.ros2.bridge")

    keys = og.Controller.Keys
    # The graph itself is session-only. The C++ publisher avoids importing the
    # system ROS Humble rclpy (Python 3.10) into Isaac's Python 3.11 process.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        og.Controller.edit(
            {"graph_path": GRAPH, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                    ("PublishScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                ],
                keys.SET_VALUES: [
                    ("PublishGate.inputs:step", 12),
                    ("PublishScan.inputs:topicName", TOPIC),
                    ("PublishScan.inputs:frameId", FRAME),
                    ("PublishScan.inputs:azimuthRange", [-180.0, 180.0]),
                    ("PublishScan.inputs:depthRange", [RANGE_MIN_M, RANGE_MAX_M]),
                    ("PublishScan.inputs:horizontalFov", 360.0),
                    ("PublishScan.inputs:horizontalResolution", 360.0 / RAY_COUNT),
                    ("PublishScan.inputs:numCols", RAY_COUNT),
                    ("PublishScan.inputs:numRows", 1),
                    ("PublishScan.inputs:rotationRate", PUBLISH_RATE_HZ),
                    (
                        "PublishScan.inputs:linearDepthData",
                        np.full(RAY_COUNT, np.inf, dtype=np.float32),
                    ),
                    (
                        "PublishScan.inputs:intensitiesData",
                        np.zeros(RAY_COUNT, dtype=np.uint8),
                    ),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishGate.inputs:execIn"),
                    ("PublishGate.outputs:execOut", "PublishScan.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishScan.inputs:timeStamp",
                    ),
                ],
            },
        )

    session_layer_identifier = stage.GetSessionLayer().identifier
    robot = SingleArticulation(ARTICULATION_ROOT)
    scene_query = omni.physx.get_physx_scene_query_interface()
    state = {"elapsed": 0.0, "initialized": False, "published": 0}

    def on_physics_step(dt):
        # Script Editor callbacks can outlive a File -> Open stage operation.
        # Never use a graph attribute (or articulation wrapper) owned by the
        # previous stage/session after that happens.
        current_stage = omni.usd.get_context().get_stage()
        if current_stage is None:
            return
        if (
            current_stage.GetSessionLayer().identifier
            != session_layer_identifier
            or not current_stage.GetPrimAtPath(GRAPH).IsValid()
        ):
            return

        state["elapsed"] += float(dt)
        if state["elapsed"] + 1.0e-12 < 1.0 / PUBLISH_RATE_HZ:
            return
        state["elapsed"] %= 1.0 / PUBLISH_RATE_HZ
        if not state["initialized"]:
            robot.initialize()
            state["initialized"] = True

        position, orientation = robot.get_world_pose()
        yaw = _yaw_from_wxyz(orientation)
        ranges = []
        for index in range(RAY_COUNT):
            local_angle = ANGLE_MIN + index * ANGLE_INCREMENT
            local_dx = math.cos(local_angle)
            local_dy = math.sin(local_angle)
            world_angle = yaw + local_angle
            world_dx = math.cos(world_angle)
            world_dy = math.sin(world_angle)
            start_distance = _distance_to_envelope(local_dx, local_dy)
            origin = carb.Float3(
                float(position[0]) + world_dx * start_distance,
                float(position[1]) + world_dy * start_distance,
                SCAN_HEIGHT_M,
            )
            direction = carb.Float3(world_dx, world_dy, 0.0)
            remaining = max(0.0, RANGE_MAX_M - start_distance)
            hit = scene_query.raycast_closest(origin, direction, remaining, True)
            if hit.get("hit", False):
                measured = start_distance + float(hit["distance"])
                ranges.append(measured if measured >= RANGE_MIN_M else math.inf)
            else:
                ranges.append(math.inf)

        # Reacquire the graph attribute from the active graph rather than
        # caching a Python AttributeObj across physics callbacks. Cached
        # wrappers become invalid when Isaac replaces/reloads the stage.
        depth_attribute = og.Controller.attribute(
            f"{GRAPH}/PublishScan.inputs:linearDepthData"
        )
        og.Controller.set(depth_attribute, np.asarray(ranges, dtype=np.float32))
        state["published"] += 1

    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
        on_physics_step
    )
    globals()["ijcas_cpu_scan_graph"] = GRAPH
    globals()["ijcas_cpu_scan_session_layer"] = session_layer_identifier
    globals()["ijcas_cpu_scan_robot"] = robot
    globals()["ijcas_cpu_scan_state"] = state
    globals()["ijcas_cpu_scan_subscription"] = subscription

    print("=== IJCAS CPU-PhysX LaserScan prepared ===")
    print("persistent USD unchanged; no RTX sensor/render product created")
    print("topic/frame:", TOPIC, FRAME)
    print("rays/rate/range:", RAY_COUNT, PUBLISH_RATE_HZ, RANGE_MAX_M)
    print("scan plane height [m]:", SCAN_HEIGHT_M)
    print("self-return exclusion envelope [m]:", 2*HALF_LENGTH_M, 2*HALF_WIDTH_M)
    print("physics: CPU dynamics, MBP broadphase")
    print("Next: start the ROS platform interface, press Play, and check /scan.")


prepare()

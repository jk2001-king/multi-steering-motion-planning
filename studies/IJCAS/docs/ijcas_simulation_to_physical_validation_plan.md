# IJCAS simulation-to-physical validation plan

## Current decision

The project has passed the full-envelope collision-truth development gate and
the planning-level necessity screen.  The accepted footprint is 1.630 x
0.965 m, including rotated passive casters.  In `fullenv_r03`, both constrained
end tasks completed all 14 actions without simulator contact.

A stronger fixed-counterphase baseline was then allowed forward and reverse
motion and a 100,000-node search budget.  It solved the front-end task with a
1.60 m path and one direction reversal, whereas configuration selection solved
the same task with a 0.70 m forward-only path.  The rear-end task was not
solved by fixed counterphase within the declared budget and was solved by the
proposed planner.  These results justify continuing the study, but do not
prove global geometric infeasibility of counterphase steering.

The immediate remaining risk is task-selection bias: the two development
goals were derived while studying the single-steering trajectories.  Before
sensor integration or a large robustness matrix, the benefit must therefore
be reproduced in operational tasks whose geometry and goal are defined without
reference to the selected steering policy.

The research sequence is:

1. finish the professor simulation gate;
2. report the contribution, comparison design, and quantitative evidence to
   the professor and obtain approval to continue the IJCAS submission;
3. freeze the planner/controller parameters;
4. validate the same scenarios first in physical shadow mode and then at low
   speed on the laboratory platform;
5. compare paired simulation and physical runs and write the paper.

This keeps the CICS initial-motion study separate. The IJCAS claim concerns
configuration-dependent swept volume, steering-transition feasibility, and
path planning/execution in constrained space.

## Why the work has looked shorter than CICS

CICS used many manually prepared runs and inspected several initial-motion
methods, load conditions, and repeated transient responses. The present IJCAS
work has so far used only two development tasks, one nominal dynamics setting,
one speed, ideal ground-truth pose/map, and effectively deterministic trials.
Automation reduces operator time, but it does not reduce the evidence needed.

The missing evidence is:

- a collision-enabled physical obstacle scene rather than virtual wall checks;
- a simulated-LiDAR-to-map closed loop rather than a ground-truth map;
- matched dynamic runs for all four policies;
- randomized repeats and uncertainty reporting;
- state/transition-cost ablations;
- load/CoM, friction, initial steering, clearance, latency, and noise screens;
- held-out scenarios frozen before evaluation;
- physical and simulation pairs.

## Gates

### G0 — feasibility (completed)

- two constrained-end tasks;
- online map/goal lattice planning;
- dynamic execution through the ROS 2 steering controller;
- provisional time, terminal-error, and virtual-clearance gates.

### G1 — collision truth

Replace the virtual corridor-only collision verdict with collision-enabled wall
and obstacle geometry in Isaac Sim. Log both planner clearance and simulator
contact/collision events. The planner footprint must use the measured 0.965 m
caster-inclusive envelope (or a validated articulated footprint), never the
0.760 m crossbeam width. A path is successful only when planner and simulator
collision verdicts agree.

**Status: completed for the two development tasks (`fullenv_r03`).**

### G1.5 — steering-configuration necessity and operational-task independence

Compare the proposal with a fixed coordinated-counterphase planner that may
drive both forward and reverse.  Record path length, direction reversals,
planning cost and search effort rather than treating a small node-budget
failure as geometric impossibility.

Then freeze at least three operational tasks from environment and task
requirements alone: bedside approach/exit, doorway or elevator entry and
alignment, and narrow-bay entry/exit.  Starts, goals and obstacles must not be
generated from any steering mode trajectory.  The proposal advances if it
either solves an otherwise failed task or reduces path length or task time by
at least 20% without adding a direction reversal, while satisfying collision
and terminal-error limits.

**Status: strong-baseline comparison completed; operational-task independence
pending.** See `steering_configuration_necessity_gate.md`.

### G2 — sensor closed loop

Mount three simulated 3D LiDARs at the recovered HY locations, publish ROS 2
`PointCloud2`, create the occupancy representation from those observations,
and plan from that map. Because Isaac Sim has no installed MID-360 asset, this
gate is an observation-coverage and ROS-interface approximation; it must not
be described as an exact MID-360 optical model.

### G3 — matched policy comparison

Run the same start, goal, obstacles, speed limit, footprint, random seed, and
stopping rule for:

1. fixed coordinated counterphase;
2. fixed front-only steering;
3. fixed rear-only steering;
4. proposed state-aware mode selection.

Report failures, not only successful-run averages. Primary outcomes are task
success, collision, minimum clearance, task time, and terminal pose error.

### G4 — ablation

Compare the full proposal with (a) measured initial steering removed and (b)
transition cost removed. This distinguishes the contribution from merely
adding three motion primitive types.

### G5 — compact robustness screen

Use six deliberately selected corner cells spanning mass/CoM, friction,
initial steering, corridor margin, command latency, and observation noise.
Run paired seeds for the strongest fixed baseline and the proposed method.
Three repeats per cell are sufficient for the professor gate; the final paper
uses ten repeats and confidence intervals on frozen held-out cases.

The physical mass is currently unresolved. The earlier 240/265/290 kg values
are simulation sensitivity cases and must not be labelled as measurements.

### G6 — professor review package

Prepare one page containing:

- the exact claim and its boundary from CICS;
- four-policy comparison plots;
- ablation results;
- robustness success/clearance/time trade-off;
- sensor closed-loop diagram;
- remaining physical measurements and risks;
- the proposed real-platform safety sequence.

The decision request is whether the demonstrated planning contribution is
sufficient to proceed to physical validation and an IJCAS manuscript.

### G7–G9 — final simulation, physical validation, sim-to-real

Freeze parameters before the held-out simulation set. On the physical robot,
run read-only shadow planning first, then bounded low-speed execution with an
independent stop operator. Match scenario geometry, command limits, and metrics
between simulation and reality. Report the performance gap and failure modes;
do not tune separately on every physical test case.

## Minimum run scale

The professor gate is expected to contain substantially more evidence than the
current two runs. A practical minimum is:

- four policies × four tasks × three paired seeds = 48 runs;
- two ablations × four tasks × three paired seeds = 24 runs;
- two selected policies × six robustness cells × two development tasks ×
  three paired seeds = 72 runs.

That is 144 automated simulation runs before the professor package. The final
paper adds ten-repeat held-out tests rather than reusing these development
trials as final evidence. The count is not a goal by itself; paired conditions,
failure accounting, and frozen parameters are the important parts.

---
type: "query"
date: "2026-08-14T02:37:20.703362+00:00"
question: "Create>Robots>Franka Emika Panda Arm这个在isaaclab上存在的资源，能不能搭建在m1上"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Configuration for Unitree Go2 robot for PVCNN project.  This module provides rob"]
---

# Q: Create>Robots>Franka Emika Panda Arm这个在isaaclab上存在的资源，能不能搭建在m1上

## Answer

Expanded from graph vocab: [robot, configuration]. Yes in simulation: use Isaac Sim Robot Assembler to attach the Franka Panda articulation to M1 BASE_LINK and preferably produce one single articulation. The local Isaac Lab provides FRANKA_PANDA_CFG with seven arm joints, two finger joints, gravity, collision, and actuator gains/limits. The combined M1+Panda articulation would have 25 DOFs. Existing M1 actions explicitly select its 16 joints, but joint position/velocity observations currently select all joints, so checkpoints become shape-incompatible unless observations remain scoped to original M1 joints and arm control is separate. Real hardware feasibility is not established: Panda is about 17.8 kg with 855 mm reach, large relative to the roughly 41 kg M1 URDF.

## Outcome

- Signal: useful

## Source Nodes

- Configuration for Unitree Go2 robot for PVCNN project.  This module provides rob
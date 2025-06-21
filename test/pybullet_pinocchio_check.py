#!/usr/bin/env python3
import pybullet as p
import time
import numpy as np
from panda_robot import PandaRobot
from pyb_utils.frame import debug_frame_world
import pinocchio as pin


# Initialize simulation
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./240.)  # 240Hz simulation

# Load robot
panda = PandaRobot(include_gripper=False)

# =============================================
# 1. SET INITIAL CONFIGURATION (CUSTOM POSITIONS)
# =============================================
initial_positions = [
    0.0,    # joint 0 
    -0.785, # joint 1 (45 degrees down)
    0.0,    # joint 2
    -2.356, # joint 3 (135 degrees down)
    0.0,    # joint 4
    1.571,  # joint 5 (90 degrees up)
    0.785,  # joint 6 (45 degrees up)
    0.01,   # gripper finger 1
    0.01    # gripper finger 2
][:panda.dof]  # Automatically adjusts for with/without gripper

# Reset to initial configuration
for j, pos in zip(panda.joints, initial_positions):
    p.resetJointState(panda.robot_id, j, targetValue=pos)
    
r,q = panda.link_pose()
# debug_frame_world(0.2, list(r), orientation=q, line_width=3)


# =============================================
# 2. POSITION CONTROL PHASE (FIRST 3 SECONDS)
# =============================================
p.setJointMotorControlArray(
    bodyUniqueId=panda.robot_id,
    jointIndices=panda.joints,
    controlMode=p.POSITION_CONTROL,
    targetPositions=initial_positions,
    forces=[200.0] * panda.dof,  # Max torque for position control
    positionGains=[0.1] * panda.dof  # PID position gain
)

# =============================================
# 3. MAIN SIMULATION LOOP
# =============================================
switch_time = 1000.0  # Switch after 3 seconds
start_time = time.time()
reset = False

# Compare mass properties
print("PyBullet masses:")
for j in range(p.getNumJoints(panda.robot_id)):
    print(p.getDynamicsInfo(panda.robot_id, j)[0])
# Check base link separately (index -1)
try:
    base_mass = p.getDynamicsInfo(panda.robot_id, -1)[0]
    print(f"Base link: mass={base_mass:.3f} kg")
except p.error:
    print("Base link mass not accessible via index -1 in this PyBullet version")

# Check all regular joints
for j in range(p.getNumJoints(panda.robot_id)):
    info = p.getJointInfo(panda.robot_id, j)
    mass = p.getDynamicsInfo(panda.robot_id, j)[0]
    print(f"Joint {j} ('{info[1].decode()}'): mass={mass:.3f} kg")
    
    
print("\nPinocchio masses:")
for inertia in panda.pin_model.inertias[1:]:  # Skip universe
    print(inertia.mass)

# for j in range(-1, p.getNumJoints(panda.robot_id)):  # -1 = base link
#     info = p.getJointInfo(panda.robot_id, j)
#     mass = p.getDynamicsInfo(panda.robot_id, j)[0]
#     print(f"Link {j}: '{info[12].decode()}' (mass={mass:.3f} kg)")
    
for frame in panda.pin_model.frames:
    if frame.type == pin.FrameType.BODY:
        inertia = panda.pin_model.inertias[frame.parent]
        print(f"Link '{frame.name}': mass={inertia.mass:.3f} kg")
        
        
        
# Corrected Pinocchio mass check
print("Pinocchio Proper Mass Mapping:")
for i, inertia in enumerate(panda.pin_model.inertias[1:]):  # Skip universe
    link_name = panda.pin_model.names[i+1]  # +1 to skip universe
    print(f"Link {i} ('{link_name}'): mass={inertia.mass:.6f} kg")

# Verify frame-parent relationships
print("\nFrame-Parent Relationships:")
for frame in panda.pin_model.frames:
    if frame.type == pin.FrameType.BODY:
        parent_name = panda.pin_model.names[frame.parent]
        print(f"Frame '{frame.name}' -> Parent '{parent_name}'")
#!/usr/bin/env python3
import pybullet as p
import time
import numpy as np
from panda_robot import PandaRobot
from panda_robot import computed_torque_ftip
from panda_robot import KalmanFilter
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
    -0.1516, # joint 1 (45 degrees down)
    0.0,    # joint 2
    -2.1603, # joint 3 (135 degrees down)
    0.0,    # joint 4
    2.0304,  # joint 5 (90 degrees up)
    0.84287,  # joint 6 (45 degrees up)
    0.01,   # gripper finger 1
    0.01    # gripper finger 2
][:panda.dof]  # Automatically adjusts for with/without gripper

# Reset to initial configuration
for j, pos in zip(panda.joints, initial_positions):
    p.resetJointState(panda.robot_id, j, targetValue=pos)
    
r_d,q_d = panda.link_pose()
debug_frame_world(0.2, list(r_d), orientation=q_d, line_width=3)
r_d.flags.writeable = False  # Make immutable
#NOTE: the returned r and q are reference, even if we don't read them again in the main loop, it changes itself.
##JUST IN ORDER TO GET THE PINOCCHIO QUTERNION
panda.fk_pin(initial_positions,[0.0] * panda.dof)
_,q_d = panda.link_pose_pin()
#print(q_d)#[ 0.91244918 -0.40904623  0.00990046 -0.00443833]


# =============================================
# 2. POSITION CONTROL PHASE 
# =============================================
p.setJointMotorControlArray(
    bodyUniqueId=panda.robot_id,
    jointIndices=panda.joints,
    controlMode=p.POSITION_CONTROL,
    targetPositions=initial_positions,
    forces=[200.0] * panda.dof,  # Max torque for position control
    positionGains=[0.1] * panda.dof  # PID position gain
)


# Compare mass properties
print("PyBullet masses:")
for j in range(p.getNumJoints(panda.robot_id)):
    print(p.getDynamicsInfo(panda.robot_id, j)[0])
# PyBullet masses:
# 2.3599995791
# 2.379518833
# 2.6498823337
# 2.6948018744
# 2.9812816864
# 1.1285806309
# 0.4052912465
# 1.0

print("\nPinocchio masses:")
for inertia in panda.pin_model.inertias[1:]:  # Skip universe
    print(inertia.mass)
# 2.3599995791
# 2.379518833
# 2.6498823337
# 2.6948018744
# 2.9812816864
# 1.1285806309
# 0.4052912465

kf = KalmanFilter(1./240., initial_positions, [0.0] * panda.dof)


# Print complete dynamics info for all joints
print("PyBullet joint dynamics info:")
for j in range(p.getNumJoints(panda.robot_id)):
    info = p.getDynamicsInfo(panda.robot_id, j)
    print(f"Joint {j} - Name: {p.getJointInfo(panda.robot_id, j)[1].decode('utf-8')}")
    print(f"  Mass: {info[0]}")
    print(f"  COM: {info[3]}")
    print(f"  Inertia: {info[2]}\n")

# Check base link mass
base_mass = p.getDynamicsInfo(panda.robot_id, -1)[0]
print(f"Base link mass: {base_mass}")

# PyBullet joint dynamics info:
# Joint 0 - Name: panda_joint1
#   Mass: 2.3599995791
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.019728142576632503, 0.015239274376996055, 0.009795094358871718)

# Joint 1 - Name: panda_joint2
#   Mass: 2.379518833
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.02011985919843313, 0.00987721452871454, 0.015591851263185243)

# Joint 2 - Name: panda_joint3
#   Mass: 2.6498823337
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.01387052818268983, 0.016003826430586343, 0.01521725860976921)

# Joint 3 - Name: panda_joint4
#   Mass: 2.6948018744
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.014362618013849918, 0.01553400453768732, 0.016569452983826477)

# Joint 4 - Name: panda_joint5
#   Mass: 2.9812816864
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.034941426638242234, 0.02922380258431487, 0.012415174987071875)

# Joint 5 - Name: panda_joint6
#   Mass: 1.1285806309
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.0028815142250094537, 0.004323742373619749, 0.0050800738585029)

# Joint 6 - Name: panda_joint7
#   Mass: 0.4052912465
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (0.0007065644851465846, 0.0007076622952390152, 0.0011641404432251791)

# Joint 7 - Name: panda_joint8
#   Mass: 1.0
#   COM: (0.0, 0.0, 0.0)
#   Inertia: (6.666666666666666e-07, 6.666666666666666e-07, 6.666666666666666e-07)

# Base link mass: 0.0
# Disable all dynamics and collision for this link
joint_index = 7
# Method 1: Set near-zero mass (recommended)
p.changeDynamics(
    panda.robot_id, 
    joint_index,
    mass=1e-6,  # Minimum practical mass
    localInertiaDiagonal=[1e-9, 1e-9, 1e-9]  # Tiny inertia
)
# =============================================
# 2. COMPARE ZERO CONFIGURATIONS
# =============================================

pos, vel = panda.get_position_and_velocity()
pos = [0.0] * panda.dof
vel = [0.0] * panda.dof
acc = [0.0] * panda.dof
panda.fk_pin(pos,vel)
r,q = panda.link_pose_pin()


dyn_torques = panda.calculate_inverse_dynamics(pos, vel, acc)
print(dyn_torques)
dyn_torques_pin = panda.inv_dyn_pin(pos, vel, acc)
print(dyn_torques_pin)
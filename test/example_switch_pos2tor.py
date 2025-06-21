#!/usr/bin/env python3
import pybullet as p
import time
import numpy as np
from panda_robot import PandaRobot

# Initialize simulation
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./240.)  # 240Hz simulation

# Load robot
panda = PandaRobot(include_gripper=True)

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
switch_time = 1.5  # Switch after 3 seconds
start_time = time.time()

reset = False
try:
    while True:
        current_time = time.time() - start_time
        
        # Position control phase
        if current_time < switch_time:
            pass  # Already set up above
        
        # Torque control phase (after switch_time)
        else:
            # First disable position control
            if(not reset):
                print("CONTROL SWITCH")
                panda.reset_state()
                reset = True
            # if current_time - switch_time < 0.001:  # Only do this once
                # print("SWITCHING TO TORQUE CONTROL!")
                p.setJointMotorControlArray(
                    bodyUniqueId=panda.robot_id,
                    jointIndices=panda.joints,
                    controlMode=p.VELOCITY_CONTROL,
                    forces=[0.0] * panda.dof  # Disable all motors
                )
            
            # Compute gravity compensation torques
            pos, vel = panda.get_position_and_velocity()
            desired_acc = [0.0] * panda.dof
            torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            
            # Apply torques
            panda.set_torques(torques)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
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
debug_frame_world(0.2, list(r), orientation=q, line_width=3)


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
switch_time = 1.0  # Switch after 3 seconds
start_time = time.time()
reset = False

try:
    while True:
        current_time = time.time() - start_time
        
        pos, vel = panda.get_position_and_velocity()
        desired_acc = [0.0] * panda.dof
        panda.fk_pin(pos)
        # Position control phase
        if current_time < switch_time:
            pass  # Already set up above
        
        # Torque control phase (after switch_time)
        else:
            # First disable position control
            # if current_time - switch_time < 0.001:  # Only do this once
            #     print("SWITCHING TO TORQUE CONTROL!")
            #     p.setJointMotorControlArray(
            #         bodyUniqueId=panda.robot_id,
            #         jointIndices=panda.joints,
            #         controlMode=p.VELOCITY_CONTROL,
            #         forces=[0.0] * panda.dof  # Disable all motors
            #     )
            if current_time > switch_time and not reset:  # Only do this once
                print("SWITCHING TO TORQUE CONTROL!")
                reset = True
                p.setJointMotorControlArray(
                    bodyUniqueId=panda.robot_id,
                    jointIndices=panda.joints,
                    controlMode=p.VELOCITY_CONTROL,
                    forces=[0.0] * panda.dof  # Disable all motors
                )
            
            # Compute gravity compensation torques
            # pos, vel = panda.get_position_and_velocity()
            # desired_acc = [0.0] * panda.dof
            torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            # print("pybullet", torques)
            # torques = panda.inv_dyn_pin(pos, vel, desired_acc)
            # print("pinocchio", torques)
            
            # Apply torques
            panda.set_torques(torques)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
    
    #debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)
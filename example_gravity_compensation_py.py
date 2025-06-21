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
# initial_positions = [
#     0.0,    # joint 0 
#     -0.785, # joint 1 (45 degrees down)
#     0.0,    # joint 2
#     -2.356, # joint 3 (135 degrees down)
#     0.0,    # joint 4
#     1.571,  # joint 5 (90 degrees up)
#     0.785,  # joint 6 (45 degrees up)
#     0.01,   # gripper finger 1
#     0.01    # gripper finger 2
# ][:panda.dof]  # Automatically adjusts for with/without gripper
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
pin_gravity = pin.Model.gravity # For Pinocchio
print(f"Pinocchio gravity: {pin_gravity}")
# # Convert to Pinocchio
# q_d_pin = pin.Quaternion(*q_d)  # Direct conversion
# print(q_d_pin)
# #(x,y,z,w) =   -0.409046  0.00990046 -0.00443833    0.912449
# Convert to scalar-last (Pinocchio's format)
# q_d_reordered = [q_d[1], q_d[2], q_d[3], q_d[0]]  # Extract x,y,z then append w
# print(q_d_reordered)
# # Result: [-0.40904623, 0.00990046, -0.00443833, 0.91244918]

# # Now create Pinocchio quaternion
# q_d_pin = pin.Quaternion(*q_d_reordered)
# print(q_d_pin)  # Correctly shows (x,y,z,w) = -0.409046 0.00990046 -0.00443833 0.912449
#q_d = [0.91244918, -0.40904623, 0.00990046, -0.00443833]  # [w,x,y,z]

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

# Compare mass properties
# print("PyBullet masses:")
# for j in range(p.getNumJoints(panda.robot_id)):
#     print(p.getDynamicsInfo(panda.robot_id, j)[0])

# print("\nPinocchio masses:")
# for inertia in panda.pin_model.inertias[1:]:  # Skip universe
#     print(inertia.mass)

kf = KalmanFilter(1./240., initial_positions, [0.0] * panda.dof)



try:
    while True:
        current_time = time.time() - start_time
        
        pos, vel = panda.get_position_and_velocity()
        #pybullet return numpy
        #print("vel",vel)
        desired_acc = [0.0] * panda.dof
        panda.fk_pin(pos,vel)
        r,q = panda.link_pose_pin()
        q_list = [q[0], q[1], q[2], q[3]]  # x,y,z,w components
        # debug_frame_world(0.2, list(r), orientation=q_list, line_width=3)
        # Position control phase
        
        # KF Prediction
        kf.predict()
        # KF Update
        x_est = kf.update(pos)
        pos = list(x_est[:7]) #pybullet receive list , otherwise core dump
        vel = list(x_est[7:])
        
        #print(pos, vel)
        
        
        if current_time < switch_time:
            pass  # Already set up above
        
        # Torque control phase (after switch_time)
        else:
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
            dyn_torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            #pybullet [4.6916841675233994e-08, -23.42250814604298, -0.00642574046228889, 
            # 21.94718832280108, -0.0005010789818168336, 1.234923928349398, -3.856155517247863e-11]

            dyn_torques_pin = panda.inv_dyn_pin(pos, vel, desired_acc)
            #pybullet [ 9.69737838e-08 -1.84555971e+01 -2.67164366e-03  1.73038047e+01
            #1.91641924e-04  3.49800105e-01  2.77426290e-10]

            #print("pybullet", dyn_torques_pin)

            total_torques = dyn_torques
            total_torques = dyn_torques_pin
            
            # # Apply torques
            panda.set_torques(total_torques)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
    

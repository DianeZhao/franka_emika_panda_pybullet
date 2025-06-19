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
            # torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            # #print("pybullet", torques)
            # torques = panda.inv_dyn_pin(pos, vel, desired_acc)
            #print("pinocchio", torques)
            #r_d,q_d = panda.link_pose_pin()
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
            dyn_torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            #print(r_d,q_d) #(x,y,z,w) =    0.912449   -0.409046  0.00989627 -0.00443645
            #print("r_d outside",r_d)
            task_torques = computed_torque_ftip(panda, r_d, q_d, pos, vel, np.array(initial_positions))
            # #print("pybullet", dyn_torques)
            # # torques = panda.inv_dyn_pin(pos, vel, desired_acc)
            # #print("task_torque", task_torques.flatten())
            # #At first iteration
            # # pybullet [3.0814596433944182e-18, -23.428008753319755, 7.378079251313554e-12, 21.943791412584993, 9.209060387693747e-12, 1.235640858425368, 4.865011792986825e-33]
            # # pinocchio [ 3.88206111e-13  1.39064904e-08 -5.37094574e-13  2.00786088e-08
            # # -3.26213842e-14 -5.91074187e-09  2.55575426e-13]

            total_torques = dyn_torques + task_torques.flatten()
            
            # # Apply torques
            panda.set_torques(total_torques)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
    

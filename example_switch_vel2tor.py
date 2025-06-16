#!/usr/bin/env python3
import pybullet as p
import time
from panda_robot import PandaRobot

# Initialize simulation
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./240.)  # Standard 240Hz simulation

# Load robot (with gripper if needed)
panda = PandaRobot(include_gripper=True)

# Disable default velocity controllers (critical!)
p.setJointMotorControlArray(
    bodyUniqueId=panda.robot_id,
    jointIndices=panda.joints,
    controlMode=p.VELOCITY_CONTROL,
    forces=[0.0] * panda.dof  # Disable default motors
)

# Timing control
start_time = time.time()
switch_time = 3.0  # Switch to torque control after 3 seconds
use_velocity_control = True
reset = False
# Simulation loop: Apply zero velocity at EVERY step
try:
    while True:
        current_time = time.time() - start_time
        # Continuously enforce zero velocity
         # Phase 1: Velocity Control (first 3 seconds)
        if use_velocity_control and current_time < switch_time:
            p.setJointMotorControlArray(
                bodyUniqueId=panda.robot_id,
                jointIndices=panda.joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * panda.dof,
                forces=[100.0] * panda.dof  # Holding torque
            )
            
        else:
            if(not reset):
                print("CONTROL SWITCH")
                panda.reset_state()
                reset = True
            # Example: Gravity compensation
            pos, vel = panda.get_position_and_velocity()
            desired_acc = [0.0] * panda.dof
            torques = panda.calculate_inverse_dynamics(pos, vel, desired_acc)
            
            panda.set_torques(torques)
        
# sets the maximum torque the motor can apply to achieve the target velocity.
# forces =0, the velocity command cannot work, it just falls due to grvaity
# forces =50, when the robot arm folded, it cannot standstill again
# forces = 100, can stand still in any configuration
# forces = 300, harder to move
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()

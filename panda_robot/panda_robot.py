import os
import math
import pybullet as p
import numpy as np
import pinocchio as pin

DOF=7
URDF_PATH = "/home/danningzhao/franka_emika_panda_pybullet/panda_robot/model_description/panda.urdf"
class PandaRobot:
    """"""

    def __init__(self, include_gripper):
        """"""
        
        
        # =============================================
        # 1. PYBULLET SIMULATION PART
        # =============================================
        p.setAdditionalSearchPath(os.path.dirname(__file__) + '/model_description')
        panda_model = "panda_with_gripper.urdf" if include_gripper else "panda.urdf"
        self.robot_id = p.loadURDF(panda_model, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # Set maximum joint velocity. Maximum joint velocity taken from:
        # https://s3-eu-central-1.amazonaws.com/franka-de-uploads/uploads/Datasheet-EN.pdf
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=0, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=1, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=2, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=4, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=5, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=6, maxJointVelocity=180 * (math.pi / 180))
        # After loading URDF in PyBullet:
        #p.changeDynamics(self.robot_id, -1, mass=0)  # Set base mass to 0
        # Set DOF according to the fact that either gripper is supplied or not and create often used joint list
        self.dof = DOF #p.getNumJoints(self.robot_id) - 1
        self.joints = range(self.dof)
        self.tool_joint_name = "panda_joint8" #

        
        
        # build a dict of all joints, keyed by name
        self.joints_dict = {}
        self.links_dict = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self.joints_dict[joint_name] = info
            self.links_dict[link_name] = info
            
        self.tool_idx = self.joints_dict[self.tool_joint_name][0]
        # get the indices for the actuated joints
        # if actuated joints are not named, we take all of the non-fixed joints
        # self.robot_joint_indices = []
        # if actuated_joints is None:
        #     for joint in self.joints.values():
        #         # joint type == 4 means a fixed joint, skip these
        #         if joint[2] == 4:
        #             continue
        #         self.robot_joint_indices.append(joint[0])
        # else:
        #     for name in actuated_joints:
        #         idx = self.joints[name][0]
        #         self.robot_joint_indices.append(idx)
        
        # Reset Robot
        self.reset_state()
        
        
        # =============================================
        # 2. PINOCCHIO KINEMATICS/DYNAMICS PART
        # =============================================
        
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH) #only kinematic
        self.pin_data = self.pin_model.createData()
        self.tool_frame_name = "panda_EndEffector"#for pinocchio
        self.tool_frame_id = self.pin_model.getFrameId(self.tool_frame_name)
        
        
        
        
    def fk_pin(self, pos, vel):
        """Apply forward kinematics to pinocchio model
        """
        pos = np.array(pos)
        vel = np.array(vel)
        pin.forwardKinematics(self.pin_model, self.pin_data, pos, vel)
        pin.updateFramePlacements(self.pin_model, self.pin_data) 
        pin.computeAllTerms(self.pin_model, self.pin_data, pos, vel)
        
    def inv_dyn_pin(self, pos, vel, acc):
        """THe pinocchio version inverse dynamics
        INPUT should be NUMPY ARRAY
        """
        pos = np.array(pos)
        vel = np.array(vel)
        acc = np.array(acc)
        
        tau_inv_dyn = pin.rnea(self.pin_model, self.pin_data, pos, vel, acc)
        return tau_inv_dyn
    
    def link_pose_pin(self):
        self.ee_pose = self.pin_data.oMf[self.tool_frame_id]
        ee_pos = self.ee_pose.translation
        ee_ort = pin.Quaternion(self.ee_pose.rotation)
        return ee_pos, ee_ort
    
    def jacobian_pin(self, pos):
        pos = np.array(pos)
        jacobian = pin.computeFrameJacobian(self.pin_model, self.pin_data, pos, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        #print(jacobian)
        return jacobian
    
    def link_pose(self, link_idx=None):
        """Get the pose of a particular link in the world frame.

        It is the pose of origin of the link w.r.t. the world. The origin of
        the link is the location of its parent joint.

        If no link_idx is provided, defaults to that of the tool.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        state = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
        pos, orn = state[4], state[5]
        return np.array(pos), np.array(orn)
    
    def reset_state(self):
        """"""
        # for j in range(self.dof):
        #     p.resetJointState(self.robot_id, j, targetValue=0)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for _ in self.joints])

    def get_dof(self):
        """"""
        return self.dof

    def get_joint_info(self, j):
        """"""
        return p.getJointInfo(self.robot_id, j)

    def get_base_position_and_orientation(self):
        """"""
        return p.getBasePositionAndOrientation(self.robot_id)

    def get_position_and_velocity(self):
        """"""
        joint_states = p.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel

    def calculate_inverse_kinematics(self, position, orientation):
        """"""
        return p.calculateInverseKinematics(self.robot_id, self.dof, position, orientation)

    def calculate_inverse_dynamics(self, pos, vel, desired_acc):
        """"""
        assert len(pos) == len(vel) and len(vel) == len(desired_acc)
        vector_length = len(pos)

        # If robot set up with gripper, set those positions, velocities and desired accelerations to 0
        if self.dof == 9 and vector_length != 9:
            pos = pos + [0., 0.]
            vel = vel + [0., 0.]
            desired_acc = desired_acc + [0., 0.]

        simulated_torque = list(p.calculateInverseDynamics(self.robot_id, pos, vel, desired_acc))

        # Remove unnecessary simulated torques for gripper if robot set up with gripper
        if self.dof == 9 and vector_length != 9:
            simulated_torque = simulated_torque[:7]
        return simulated_torque

    def set_target_positions(self, desired_pos):
        """"""
        # If robot set up with gripper, set those positions to 0
        if self.dof == 9:
            desired_pos = desired_pos + [0., 0.]
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=desired_pos)

    def set_torques(self, desired_torque):
        """"""
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=desired_torque)

    def link_pose(self, link_idx=None):
        """Get the pose of a particular link in the world frame.

        It is the pose of origin of the link w.r.t. the world. The origin of
        the link is the location of its parent joint.

        If no link_idx is provided, defaults to that of the tool.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        state = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
        pos, orn = state[4], state[5]
        return np.array(pos), np.array(orn)
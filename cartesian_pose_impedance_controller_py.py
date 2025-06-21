#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import pinocchio as pin
#from .parameter import Mlist, Glist, Slist
# from ur10e_force_msgs.msg import ForceCmd  # Assuming this message exists
import modern_robotics as mr  # Must be installed separately
from geometry_msgs.msg import PoseStamped

JOINT_SIZE = 7 #7+2
gravity = np.array([0.0, 0.0, -9.8])
urdf_path = "/home/danningzhao/franka-impedance-ws/src/franka_interactive_controllers/panda.urdf"
FRAME_NAME = "panda_EE"
JOINTS_TO_LOCK = ["panda_finger_joint1", "panda_finger_joint2"]
DEFAULT_CONFIGURATION = np.array([0, -0.151628, 0,
                                  -2.160299,0.0063, 2.0304,
                                  0.8428, 0.01, 0.01])
DEFAULT_CONFIGURATION_REDUCED = np.array([0, -0.151628, 0,
                                  -2.160299,0.0063, 2.0304,
                                  0.8428])

class ForceControlClientSubscriber:
    def __init__(self):
        
        #model = pin.buildModelsFromUrdf(urdf_path) returns a tuple, kinematic model, collsision model and visual model
        #########################USE PINnOCHIO#######################################
        self.original_model = pin.buildModelFromUrdf(urdf_path) #only kinematic
        self.original_data = self.original_model.createData()
        self.locked_joint_ids = [self.original_model.getJointId(name) for name in JOINTS_TO_LOCK]
        # self.model = pin.buildReducedModel(self.original_model, self.locked_joint_ids, DEFAULT_CONFIGURATION)
        self.model = pin.buildReducedModel(self.original_model, self.locked_joint_ids, DEFAULT_CONFIGURATION)
        self.data = self.model.createData()
        
        
        self.frame_id = self.model.getFrameId(FRAME_NAME)
        
        self.joint_position = DEFAULT_CONFIGURATION_REDUCED #np.zeros(JOINT_SIZE)
        self.joint_velocity = np.zeros(JOINT_SIZE)

        self.thetalist = DEFAULT_CONFIGURATION_REDUCED#np.zeros(JOINT_SIZE)#current state
        self.dthetalist = np.zeros(JOINT_SIZE)
        self.ddthetalist = np.zeros(JOINT_SIZE)

        self.thetalistd = np.zeros(JOINT_SIZE)
        self.dthetalistd = np.zeros(JOINT_SIZE)
        self.ddthetalistd = np.zeros(JOINT_SIZE)
        self.Ftip = np.zeros(JOINT_SIZE)
        #Integrated error
        self.eint = np.zeros(JOINT_SIZE)
    
    

        ##TODO, MAKE IT AS MATRIX
        self.Kp = 10
        self.Ki = 0
        self.Kd = 5
        ###################TODO: define cartesian_stiffness, cartesian_damping#############################
        # Assuming this is defined elsewhere:
        #self.cartesian_stiffness_target_array = np.array([300, 300, 300, 50, 50, 50])
        self.cartesian_stiffness_target_array = np.array([300, 300, 300, 50, 50, 50])
        self.cartesian_stiffness_target = np.diag(self.cartesian_stiffness_target_array)
        self.cartesian_damping_target_array = 2.0 * np.sqrt(self.cartesian_stiffness_target_array)
        self.cartesian_damping_target = np.diag(self.cartesian_damping_target_array)
        # print(self.cartesian_stiffness_target)
        # print(self.cartesian_damping_target)
        ###################TODO: nullspace ################################
        self.nullspace_stiffness = 0.01
        self.q_d_nullspace = None #can be read from yaml, if no, then read from initial robot state
        ###################TODO: ###########################################
        self.ee_pose = None # to hold the pinocchio pose
        self.ee_pos =np.zeros(3)
        self.ee_pos_d =None
        self.ee_ort =None
        self.ee_ort_d =None
        
        
        self.gravity = np.array([0, 0, -9.81])
        ####Module depended cosntant params
        # self.Mlist = Mlist
        # self.Glist = Glist
        # self.Slist = Slist
        # ##

        rospy.Subscriber("/joint_states", JointState, self.panda_callback)
        rospy.Subscriber("/cartesian_impedance_controller/desired_pose", PoseStamped, self.desired_pose_callback) #TODO: receive the target pose of the interactive marker

        # self.publishers = [
        #     rospy.Publisher("/ur10e/shoulder_pan_joint_effort_controller/command", Float64, queue_size=10),
        #     rospy.Publisher("/ur10e/shoulder_lift_joint_effort_controller/command", Float64, queue_size=10),
        #     rospy.Publisher("/ur10e/elbow_joint_effort_controller/command", Float64, queue_size=10),
        #     rospy.Publisher("/ur10e/wrist_1_joint_effort_controller/command", Float64, queue_size=10),
        #     rospy.Publisher("/ur10e/wrist_2_joint_effort_controller/command", Float64, queue_size=10),
        #     rospy.Publisher("/ur10e/wrist_3_joint_effort_controller/command", Float64, queue_size=10),
        # ]

    def panda_callback(self, msg):
        
        # self.joint_position = np.array([
        #     msg.position[0], msg.position[1], msg.position[2],
        #     msg.position[3], msg.position[4], msg.position[5],
        #     msg.position[6], msg.position[7], msg.position[8]
        # ])
        # self.joint_velocity = np.array([
        #     msg.velocity[0], msg.velocity[1], msg.velocity[2],
        #     msg.velocity[3], msg.velocity[4], msg.velocity[5],
        #     msg.velocity[6], msg.velocity[7], msg.velocity[8]
        # ])
        self.joint_position = np.array([
            msg.position[0], msg.position[1], msg.position[2],
            msg.position[3], msg.position[4], msg.position[5],
            msg.position[6]
        ])
        self.joint_velocity = np.array([
            msg.velocity[0], msg.velocity[1], msg.velocity[2],
            msg.velocity[3], msg.velocity[4], msg.velocity[5],
            msg.velocity[6]
        ])
        
    def desired_pose_callback(self, msg: PoseStamped):
        # Extract translation
        position = msg.pose.position
        self.ee_pos_d = np.array([position.x, position.y, position.z])

        # Extract orientation (quaternion)
        orientation = msg.pose.orientation
        # self.ee_ort_d = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        # Construct Pinocchio quaternion from [x, y, z, w]
        self.ee_ort_d = pin.Quaternion(np.array([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]))
        #rospy.logwarn(f"target: pos = {self.ee_pos_d}, quat = {self.ee_ort_d}")


    def run(self):
        rate = rospy.Rate(1000)
        while not rospy.is_shutdown():
            
            self.thetalist = self.joint_position
            #rospy.logwarn(f"thetalist: {self.thetalist}")
            self.dthetalist = self.joint_velocity
            ##Q: how the subscriber works
            pin.forwardKinematics(self.model, self.data, self.thetalist)
            pin.updateFramePlacements(self.model, self.data) 
            self.ee_pose = self.data.oMf[self.frame_id]
            self.ee_pos = self.ee_pose.translation
            self.ee_ort = pin.Quaternion(self.ee_pose.rotation)
            if(self.ee_ort_d is None or self.ee_pos_d is None):
                self.ee_ort_d = pin.Quaternion(self.ee_pose.rotation)
                self.ee_pos_d = self.ee_pose.translation
            if(self.q_d_nullspace is None):
                self.q_d_nullspace = self.thetalist
            
            # rospy.logwarn(f"ee-pose: {self.ee_pose}")
            # print(self.ee_pos)# (x,y,z) note: not very exact, error : several  centimeter
            # print(self.ee_ort)# (x,y,z,w)
            
            tau = self.computed_torque_ftip()

            # for i in range(JOINT_SIZE):
            #     self.publishers[i].publish(Float64(tau[i]))

            #rospy.logwarn(f"Torques: {tau}")
            rate.sleep()

    
    
    def computed_torque_ftip(self):
            # e = self.thetalistd - self.thetalist
            # self.eint += e  # Integrate error

            # #########################################FEEDFOWARD OF JOINT SPACE#########################################
            # # Mass matrix
            # M = pin.crba(self.model, self.data, self.thetalist)
            # # Control acceleration (PD+I law)
            # tau_ff2 = M @ (self.Kp * e + self.Kd * (self.dthetalistd - self.dthetalist) + self.Ki * self.eint)
            # ###########################################DYNAMIC COMPENSTTION#########################################
            # # Nonlinear effects (Coriolis + Gravity)
            tau_inv_dyn2 = pin.rnea(self.model, self.data, self.thetalist, self.dthetalist, np.zeros(JOINT_SIZE))
            # rospy.logwarn(f"coriolis: {tau_inv_dyn2}")
            # tau = tau_ff1 + tau_inv_dyn1
            #####################################Cartesian Impedance Control#########################################
            #rospy.logwarn(f"error pos: {self.ee_pos}")
            # rospy.logwarn(f"error pos: {self.ee_pos_d}")##Q:it seems like at the first callback, self.ee_pos_d is not updated yet by marker
            error_pos = self.ee_pos - self.ee_pos_d
            error_rot_base = compute_orientation_error(self.ee_ort, self.ee_ort_d, self.ee_pose) 
            error_pos = error_pos.reshape(3, 1)
            error_rot_base = error_rot_base.reshape(3, 1)
            error = np.vstack((error_pos, error_rot_base))
            #rospy.logwarn(f"current error: {error}")#translation error not good, some 
            jacobian = pin.computeFrameJacobian(self.model, self.data, self.thetalist, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            #Q:WHY?
            #print("jacobian",jacobian)
            # 1. Compute end-effector spatial velocity
            velocity = jacobian @ self.dthetalist  # shape (7,)
            #print("dthetalist", self.dthetalist) 
            #the 7th joint has large velocity when the gripper stays near to the target, why???
            #BECAUSE THEY ARE DEFINED AS POSITION! when they are initialized, typo
            velocity = velocity.reshape((6,1))
            #rospy.logwarn(f"current velocity: {velocity}")
            # 2. Compute desired Cartesian wrench
            F_ee_des = -self.cartesian_stiffness_target @ error - self.cartesian_damping_target @ velocity  # shape (6,), the result is 6,6
            
            #print("stiffness",-self.cartesian_stiffness_target @ error)
            #print("velocity",velocity) # the last one is super huge, why?
            #print("damping",-self.cartesian_damping_target @ velocity)
            #rospy.logwarn(f"F_ee_des: {F_ee_des}")
            jacobian_t = jacobian.T #7,6
            tau_task = jacobian_t @ F_ee_des
            #rospy.logwarn(f"current velocity norm: {np.linalg.norm(velocity[:3])}")
            rospy.logwarn(f"linear control force: {np.linalg.norm(F_ee_des[:3])}")
            rospy.logwarn(f"angular control force: {np.linalg.norm(F_ee_des[3:])}")
            #rospy.logwarn(f"tau_task: {tau_task}")
            ######################################TODO: NullSpace##############################################################
            jacobian_transpose_pinv = pseudo_inverse(jacobian_t, damped=True)
            #print(jacobian_transpose_pinv.shape) #6,7
            # Compute nullspace projection matrix
            I = np.eye(7)
            nullspace_projection = I - jacobian_t @ jacobian_transpose_pinv  # 7x7
            #print(nullspace_projection.shape) #(7,7)
            # PD control in nullspace (critically damped)
            q_d_nullspace = self.q_d_nullspace.reshape((7, 1))

            theta = self.thetalist.reshape((7, 1))
            dtheta = self.dthetalist.reshape((7, 1))

            vector = self.nullspace_stiffness * (q_d_nullspace - theta) - 2.0 * np.sqrt(self.nullspace_stiffness) * dtheta# 7,1
            
            #print(vector)
            # tau_nullspace = nullspace_projection @ (
            #     self.nullspace_stiffness * (q_d_nullspace - self.thetalist) -
            #     2.0 * np.sqrt(self.nullspace_stiffness) * self.dthetalist)#OTHERWISE the dim of tau will be 7x7
            tau_nullspace = nullspace_projection @ vector
            
            #print("Nullspace torques:", tau_nullspace.T)
            ######################################TODO: Tau tool###############################################################
            
            ## Q: how to make sure they are actually read before we call this function?????self.ee_pose is an SE3 object
            # rospy.logwarn(f"error pos: {error_pos}")
            #rospy.logwarn(f"error orient: {error_rot_base}")
            # Compute full nonlinear effects: C(q,v)·v + g(q)
            nle = pin.nonLinearEffects(self.model, self.data, self.thetalist, self.dthetalist)
            #rospy.logwarn(f"nle: {nle.T}")
            # Compute gravity alone: g(q)
            gravity = pin.computeGeneralizedGravity(self.model, self.data, self.thetalist)
            #rospy.logwarn(f"gravity: {gravity.T}") #THe value of inv dyn mainly comes from here, large value
            # Subtract gravity to get Coriolis term
            coriolis = nle - gravity
            rospy.logwarn(f"coriolis: {coriolis.T}")
            tau = tau_task.flatten() + tau_nullspace.flatten() + tau_inv_dyn2#coriolis.flatten()
            # tau_task = tau_task.flatten()
            # tau = tau_task + tau_inv_dyn2
            rospy.logwarn(f"Desired control toruqe: {tau}")
            
            #tau = tau_inv_dyn2
            #return tau
        
def compute_orientation_error(orientation: pin.Quaternion, orientation_d: pin.Quaternion, transform: pin.SE3):
        # Ensure shortest rotation (same hemisphere)
        # if orientation_d.coeffs().dot(orientation.coeffs()) < 0.0:
        #     orientation.coeffs()[:] *= -1.0  # Flip quaternion, but readonly
        if orientation_d.coeffs().dot(orientation.coeffs()) < 0.0:
            orientation = pin.Quaternion(-orientation.coeffs())

        # Compute the difference quaternion: q_err = inv(q) * q_des
        q_error = orientation.inverse() * orientation_d
        #rospy.logwarn(f"q error: {q_error}")
        # Take the imaginary part (x, y, z) of the error quaternion
        x,y,z,w = q_error.coeffs()
        error_rot = np.array([x,y,z])

        # Transform the error to the base frame: -R * error
        error_rot_base = -transform.rotation @ error_rot
        # This works as expected, because NumPy treats a (3,) array like a column vector when used with matrix multiplication (@). 
        # So if transform.rotation is a (3, 3) matrix, then: @ error_rot is valid
        # The result is a (3,) array (still flat)

        return error_rot_base

def pseudo_inverse(M, damped=True):
    """
    Compute the (optionally damped) Moore-Penrose pseudoinverse of matrix M.
    Equivalent to the Eigen C++ version using JacobiSVD.
    """
    lambda_ = 0.2 if damped else 0.0

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Compute damped inverse of singular values
    S_inv_damped = np.array([
        s / (s**2 + lambda_**2) if s > 1e-8 else 0.0 for s in S
    ])
    
    # Reconstruct pseudo-inverse
    S_inv_mat = np.diag(S_inv_damped)
    M_pinv = Vt.T @ S_inv_mat @ U.T
    
    return M_pinv

if __name__ == "__main__":
    rospy.init_node("ur10e_force_control_client")
    node = ForceControlClientSubscriber()
    node.run()




